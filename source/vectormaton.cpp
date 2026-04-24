#include "vectormaton.h"

void VectorMaton::set_vectors(const std::vector<float>& vectors, int dimension) {
    vecs = vectors;
    dim = dimension;
    num_elements = dim == 0 ? 0 : static_cast<int>(vecs.size()) / dim;
}

void VectorMaton::set_strings(const std::vector<std::string>& strings) {
    strs = strings;
}

void VectorMaton::build_gsa() {
    LOG_DEBUG("Building Generalized Suffix Automaton (GSA)");
    unsigned long long start_time = currentTime();
    for (int i = 0; i < num_elements; i++) {
        gsa.add_string(i, strs[i]);
    }
    LOG_DEBUG("GSA built in ", timeFormatting(currentTime() - start_time).str());
    LOG_DEBUG("Total GSA states: ", std::to_string(gsa.size()), ", total string IDs in GSA: ", std::to_string(gsa.size_tot()));

    // Print GSA statistics
    // auto stats = gsa.get_statistics();
    // for (size_t depth = 0; depth < stats.size(); ++depth) {
    //     const auto &stat = stats[depth];
    //     LOG_DEBUG("Depth ", depth, ": num states = ", stat.sizes.size(),
    //               ", median vector set size = ", stat.mid,
    //               ", average vector set size = ", stat.avg);
    // }
}

void VectorMaton::clear_gsa() {
    for (int i = 0; i < gsa.st.size(); i++) {
        gsa.st[i].ids = std::vector<uint32_t>();
    }
}

void VectorMaton::insert(const std::vector<float>& vec, const std::string& str) {
    if (static_cast<int>(vec.size()) != dim) return;
    vecs.insert(vecs.end(), vec.begin(), vec.end());
    strs.emplace_back(str);
    num_elements++;
    gsa.add_string(num_elements - 1, str);
    // Expand inherit_states, size_ids, candidate_ids and hnsws for the new states
    while (candidate_ids.size() < gsa.st.size()) {
        int new_state = candidate_ids.size(), num_ids = gsa.st[new_state].ids.size();
        if (inherit_states.size() > 0) inherit_states.emplace_back(-1);
        candidate_ids.emplace_back(std::vector<int>());
        hnsws.emplace_back(nullptr);
    }
    for (int state : gsa.affected_states) {
        if (candidate_ids[state].empty()) {
            // For brand new states, construct index directly (without inheriting from children)
            candidate_ids[state].resize(gsa.st[state].ids.size());
            for (int j = 0; j < candidate_ids[state].size(); j++) {
                candidate_ids[state][j] = gsa.st[state].ids[j];
            }
            gsa.st[state].ids = std::vector<uint32_t>();
            if (candidate_ids[state].size() >= min_build_threshold) {
                int M = 16, ef_construction = 200;
                hnsws[state] = new hnswlib::HierarchicalNSW<float>(space, candidate_ids[state].size(), vecs.data(), M, ef_construction);
                for (int id : candidate_ids[state]) {
                    hnsws[state]->addPoint(id);
                }
            }
        }
        else if (inherit_states.size() == 0 || inherit_states[state] == -1) {
            // For old states without inheritance, if it is not processed before, add the new vector to candidate list and index
            if (candidate_ids[state].back() != num_elements - 1) {
                candidate_ids[state].emplace_back(num_elements - 1);
                gsa.st[state].ids = std::vector<uint32_t>();
                if (hnsws[state]) {
                    hnsws[state]->resizeIndex(candidate_ids[state].size());
                    hnsws[state]->addPoint(num_elements - 1);
                }
                else if (candidate_ids[state].size() >= min_build_threshold) {
                    int M = 16, ef_construction = 200;
                    hnsws[state] = new hnswlib::HierarchicalNSW<float>(space, candidate_ids[state].size(), vecs.data(), M, ef_construction);
                    for (int id : gsa.st[state].ids) {
                        hnsws[state]->addPoint(id);
                    }
                }
            }
        }
        else {
            // For old states with inheritance, if it is not processed before, process it and its inherited states recursively
            if (candidate_ids[state].back() != num_elements - 1) {
                int now = state;
                std::vector<int> to_process = {now};
                while (inherit_states[now] != -1 && gsa.st[now].ids.size() > 0 && gsa.st[now].ids.back() == num_elements - 1) {
                    now = inherit_states[now];
                    to_process.emplace_back(now);
                }
                for (int i = to_process.size() - 1; i >= 0; i--) {
                    int s = to_process[i];
                    if (gsa.st[s].ids.size() > 0 && gsa.st[s].ids.back() == num_elements - 1) {
                        gsa.st[s].ids = std::vector<uint32_t>();
                        if (inherit_states[s] != -1 && candidate_ids[inherit_states[s]].size() > 0 && candidate_ids[inherit_states[s]].back() == num_elements - 1) {
                            continue;
                        }
                        candidate_ids[s].emplace_back(num_elements - 1);
                        if (hnsws[s]) {
                            hnsws[s]->resizeIndex(candidate_ids[s].size());
                            hnsws[s]->addPoint(num_elements - 1);
                        }
                        else if (candidate_ids[s].size() >= min_build_threshold) {
                            int M = 16, ef_construction = 200;
                            hnsws[s] = new hnswlib::HierarchicalNSW<float>(space, candidate_ids[s].size(), vecs.data(), M, ef_construction);
                            for (int id : gsa.st[s].ids) {
                                hnsws[s]->addPoint(id);
                            }
                        }
                    }
                }
            }
        }
    }
}

void VectorMaton::build_parallel(int cores) {
    build_gsa();
    gsa.build_reverse();

    // Smart build will inherit info from children
    inherit_states.assign(gsa.st.size(), -1);
    candidate_ids.assign(gsa.st.size(), std::vector<int>());
    int* largest_state = new int[gsa.st.size()];
    for (int i = 0; i < gsa.st.size(); i++) {
        largest_state[i] = -1;
    }

    hnsws.assign(gsa.st.size(), nullptr);
    for (int i = 0; i < gsa.st.size(); i++) {
        hnsws[i] = nullptr;
    }
    if (!space) {
        space = new hnswlib::L2Space(dim);
    }
    MPMCQueue q(1 << (int(log2(gsa.st.size())) + 1));
    std::atomic<int> num_init = 0;
    #pragma omp parallel for num_threads(cores)
    for (int i = 0; i < gsa.st.size(); i++) {
        if (gsa.deg[i] == 0) {
            if (!q.enqueue(i)) {
                LOG_WARN("Enqueue failed for GSA state ", i);
            }
            num_init++;
        }
    }
    LOG_DEBUG("Initial boundary states: ", num_init, ", initial queue size: ", q.size(), ", match: ", num_init == q.size() ? "yes" : "no");

    int cur = 0, ten_percent = gsa.size_tot() / 10, tot_vertices = gsa.size_tot();
    std::atomic<int> consumed = 0, built_vertices = 0;
    std::mutex mtx;
    #pragma omp parallel num_threads(cores)
    {
        int i = 0;
        while (true) {
            int c = consumed.load(std::memory_order_acquire);
            if (c >= gsa.st.size())
                break;

            if (q.dequeue(i)) {
                int prev = consumed.fetch_add(1, std::memory_order_acq_rel);
                if (prev < gsa.st.size()) {
                    auto& st = gsa.st[i];
                    int prev_built = built_vertices.fetch_add(st.ids.size(), std::memory_order_acq_rel);
                    if (prev_built >= cur) {
                        mtx.lock();
                        if (prev_built >= cur) {
                            cur += ten_percent;
                            LOG_DEBUG("Built states ", prev, "/", gsa.st.size(), " Built vertices: ", prev_built, "/", tot_vertices);
                        }
                        mtx.unlock();
                    }
                    if (st.ids.size() < min_build_threshold) {
                        int num_ids = st.ids.size();
                        candidate_ids[i].resize(num_ids);
                        for (int j = 0; j < num_ids; j++) {
                            candidate_ids[i][j] = st.ids[j];
                        }
                        st.ids = std::vector<uint32_t>();
                        // Enqueue
                        for (auto prev : gsa.reverse_next[i]) {
                            int val = gsa.deg[prev].fetch_sub(1);
                            if (val == 1) {
                                q.enqueue(prev);
                            }
                        }
                        continue;
                    }
                    // First find a successor with largest built graph
                    int target_sc = -1;
                    for (auto ch : st.next) {
                        if (largest_state[ch.second] != -1 && (target_sc == -1 || candidate_ids[largest_state[ch.second]].size() > candidate_ids[target_sc].size())) {
                            target_sc = largest_state[ch.second];
                        }
                    }
                    inherit_states[i] = target_sc;
                    if (target_sc == -1) {
                        // No successor has built a graph, built the graph with all vector ids
                        int M = 16, ef_construction = 200;
                        hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), vecs.data(), M, ef_construction);
                        int num_ids = st.ids.size();
                        candidate_ids[i].resize(num_ids);
                        for (int j = 0; j < num_ids; j++) {
                            int id = st.ids[j];
                            hnsws[i]->addPoint(id);
                            candidate_ids[i][j] = id;
                        }
                        largest_state[i] = i;
                        st.ids = std::vector<uint32_t>();
                    }
                    else {
                        // Found the largest successor, inherit from this successor
                        inherit_states[i] = target_sc;
                        largest_state[i] = target_sc;
                        // Find remaining vertices
                        int l = 0, r = 0, cnt = 0;
                        int num_ids = st.ids.size() - candidate_ids[target_sc].size();
                        candidate_ids[i].resize(num_ids);
                        while (l < st.ids.size() || r < candidate_ids[target_sc].size()) {
                            if (r == candidate_ids[target_sc].size()) {
                                candidate_ids[i][cnt++] = st.ids[l++];
                            }
                            else {
                                if (st.ids[l] == candidate_ids[target_sc][r]) {
                                    l++, r++;
                                }
                                else {
                                    candidate_ids[i][cnt++] = st.ids[l++];
                                }
                            }
                        }
                        // Only build when meeting requirements
                        if (candidate_ids[i].size() >= min_build_threshold) {
                            int M = 16, ef_construction = 200;
                            hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, candidate_ids[i].size(), vecs.data(), M, ef_construction);
                            for (int j = 0; j < candidate_ids[i].size(); j++) {
                                int id = candidate_ids[i][j];
                                hnsws[i]->addPoint(id);
                            }
                            if (candidate_ids[i].size() > candidate_ids[target_sc].size()) {
                                // Update largest state
                                largest_state[i] = i;
                            }
                        }
                        st.ids = std::vector<uint32_t>();
                    }

                    // Enqueue
                    for (auto prev : gsa.reverse_next[i]) {
                        int val = gsa.deg[prev].fetch_sub(1);
                        if (val == 1) {
                            q.enqueue(prev);
                        }
                    }
                }
            } else {
                #pragma omp flush
            }
        }
    }
}

void VectorMaton::build_smart() {
    build_gsa();
    
    // Smart build will inherit info from children
    inherit_states.assign(gsa.st.size(), -1);
    candidate_ids.assign(gsa.st.size(), std::vector<int>());
    int* largest_state = new int[gsa.st.size()];
    for (int i = 0; i < gsa.st.size(); i++) {
        largest_state[i] = -1;
    }

    hnsws.assign(gsa.st.size(), nullptr);
    for (int i = 0; i < gsa.st.size(); i++) {
        hnsws[i] = nullptr;
    }
    if (!space) {
        space = new hnswlib::L2Space(dim);
    }
    int cur = 0, ten_percent = gsa.size_tot() / 10, built_vertices = 0, tot_vertices = gsa.size_tot();
    auto topo_order = gsa.topo_sort();
    for (int i = topo_order.size() - 1; i >= 0; i--) {
        if (built_vertices >= cur) {
            cur += ten_percent;
            LOG_DEBUG("Building HNSW for state ", gsa.st.size() - i, "/", gsa.st.size(), " Built vertices: ", built_vertices, "/", tot_vertices);
        }
        built_vertices += gsa.st[i].ids.size();
        auto& st = gsa.st[i];
        if (st.ids.size() < min_build_threshold) {
            candidate_ids[i].resize(st.ids.size());
            for (int j = 0; j < st.ids.size(); j++) {
                candidate_ids[i][j] = st.ids[j];
            }
            st.ids = std::vector<uint32_t>();
            continue;
        }
        // First find a successor with largest built graph
        int target_sc = -1;
        for (auto ch : st.next) {
            if (largest_state[ch.second] != -1 && (target_sc == -1 || candidate_ids[largest_state[ch.second]].size() > candidate_ids[target_sc].size())) {
                target_sc = largest_state[ch.second];
            }
        }
        inherit_states[i] = target_sc;
        if (target_sc == -1) {
            // No successor has built a graph, built the graph with all vector ids
            int M = 16, ef_construction = 200;
            hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), vecs.data(), M, ef_construction);
            int num_ids = st.ids.size();
            candidate_ids[i].resize(num_ids);
            for (int j = 0; j < num_ids; j++) {
                int id = st.ids[j];
                hnsws[i]->addPoint(id);
                candidate_ids[i][j] = id;
            }
            largest_state[i] = i;
            st.ids = std::vector<uint32_t>();
        }
        else {
            // Found the largest successor, inherit from this successor
            inherit_states[i] = target_sc;
            largest_state[i] = target_sc;
            // Find remaining vertices
            int l = 0, r = 0, cnt = 0;
            int num_ids = st.ids.size() - candidate_ids[target_sc].size();
            candidate_ids[i].resize(num_ids);
            while (l < st.ids.size() || r < candidate_ids[target_sc].size()) {
                if (r == candidate_ids[target_sc].size()) {
                    candidate_ids[i][cnt++] = st.ids[l++];
                }
                else {
                    if (st.ids[l] == candidate_ids[target_sc][r]) {
                        l++, r++;
                    }
                    else {
                        candidate_ids[i][cnt++] = st.ids[l++];
                    }
                }
            }
            // Only build when meeting requirements
            if (candidate_ids[i].size() >= min_build_threshold) {
                int M = 16, ef_construction = 200;
                hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, candidate_ids[i].size(), vecs.data(), M, ef_construction);
                for (int j = 0; j < candidate_ids[i].size(); j++) {
                    int id = candidate_ids[i][j];
                    hnsws[i]->addPoint(id);
                }
                if (candidate_ids[i].size() > candidate_ids[target_sc].size()) {
                    // Update largest state
                    largest_state[i] = i;
                }
            }
            st.ids = std::vector<uint32_t>();
        }
    }

    delete [] largest_state;
    // clear_gsa();
}

void VectorMaton::build_full() {
    build_gsa();
    candidate_ids.assign(gsa.st.size(), std::vector<int>());

    // Build graph index
    hnsws.assign(gsa.st.size(), nullptr);
    for (int i = 0; i < gsa.st.size(); i++) {
        hnsws[i] = nullptr;
    }
    if (!space) {
        space = new hnswlib::L2Space(dim);
    }
    int cur = 0, ten_percent = gsa.size_tot() / 10, built_vertices = 0, tot_vertices = gsa.size_tot();
    for (int i = gsa.st.size() - 1; i >= 0; i--) {
        if (built_vertices >= cur) {
            cur += ten_percent;
            LOG_DEBUG("Building HNSW for state ", gsa.st.size() - i, "/", gsa.st.size(), " Built vertices: ", built_vertices, "/", tot_vertices);
        }
        built_vertices += gsa.st[i].ids.size();
        auto& st = gsa.st[i];
        candidate_ids[i].resize(st.ids.size());
        for (int j = 0; j < st.ids.size(); j++) {
            candidate_ids[i][j] = st.ids[j];
        }
        int M = 16, ef_construction = 200;
        hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), vecs.data(), M, ef_construction);
        for (auto id : st.ids) {
            hnsws[i]->addPoint(id);
        }
        st.ids = std::vector<uint32_t>();
    }
    
    // clear_gsa();
}

void VectorMaton::load_index(const char* input_folder) {
    namespace fs = std::filesystem;
    fs::path in_path(input_folder);

    fs::path gsa_file = in_path / "gsa.in";
    LOG_DEBUG("Loading automaton data from ", gsa_file.string());
    std::string tmp = gsa_file.string();
    std::vector<char> filename(tmp.begin(), tmp.end());
    filename.push_back('\0');
    gsa = GeneralizedSuffixAutomaton(filename.data());

    fs::path internal_file = in_path / "internal.in";
    LOG_DEBUG("Loading VectorMaton internal data from ", internal_file.string());
    std::ifstream f(internal_file.string());
    inherit_states.assign(gsa.st.size(), -1);
    for (int i = 0; i < gsa.st.size(); i++) {
        f >> inherit_states[i];
    }
    int* size_ids = new int[gsa.st.size()];
    for (int i = 0; i < gsa.st.size(); i++) {
        f >> size_ids[i];
    }
    candidate_ids.assign(gsa.st.size(), std::vector<int>());
    for (int i = 0; i < gsa.st.size(); i++) {
        candidate_ids[i].resize(size_ids[i]);
        for (int j = 0; j < size_ids[i]; j++) {
            f >> candidate_ids[i][j];
        }
    }
    delete [] size_ids;
    f.close();

    LOG_DEBUG("Loading HNSW data");
    space = new hnswlib::L2Space(dim);
    hnsws.assign(gsa.st.size(), nullptr);
    for (int i = 0; i < gsa.st.size(); i++) {
        std::string s = "hnsw";
        s += std::to_string(i);
        std::vector<char> buf(s.begin(), s.end());
        buf.push_back('\0');
        fs::path hnsw_file = in_path / buf.data();
        std::string tmp = hnsw_file.string();
        if (fs::exists(hnsw_file)) {
            hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, tmp, vecs.data());
        }
        else {
            hnsws[i] = nullptr;
        }
    }
}

void VectorMaton::save_index(const char* output_folder) {
    namespace fs = std::filesystem;
    fs::path out_path(output_folder);

    if (!fs::exists(out_path)) {
        fs::create_directories(out_path);  // safer than create_directory
    }

    fs::path gsa_file = out_path / "gsa.in";
    LOG_DEBUG("Saving automaton data from ", gsa_file.string());
    std::string tmp = gsa_file.string();
    std::vector<char> filename(tmp.begin(), tmp.end());
    filename.push_back('\0');
    gsa.dump(filename.data());

    fs::path internal_file = out_path / "internal.in";
    LOG_DEBUG("Saving VectorMaton internal data from ", internal_file.string());
    std::ofstream f(internal_file.string());
    for (int i = 0; i < gsa.st.size(); i++) {
        f << inherit_states[i] << " ";
    }
    f << "\n";
    for (int i = 0; i < gsa.st.size(); i++) {
        f << candidate_ids[i].size() << " ";
    }
    f << "\n";
    for (int i = 0; i < gsa.st.size(); i++) {
        for (int j = 0; j < candidate_ids[i].size(); j++) {
            f << candidate_ids[i][j] << " ";
        }
        f << "\n";
    }
    f.close();

    LOG_DEBUG("Saving HNSW data");
    for (int i = 0; i < gsa.st.size(); i++) {
        if (hnsws[i]) {
            std::string s = "hnsw";
            s += std::to_string(i);
            std::vector<char> buf(s.begin(), s.end());
            buf.push_back('\0');
            fs::path hnsw_file = out_path / buf.data();
            std::string tmp = hnsw_file.string();
            hnsws[i]->saveIndex(tmp);
        }
    }
}

size_t VectorMaton::size() {
    size_t total_size = 0;
    size_t hnsw_size = 0;
    for (int i = 0; i < gsa.st.size(); i++) {
        auto hnsw = hnsws[i];
        if (!hnsw) continue;
        hnsw_size += hnsw->indexFileSize();
    }
    LOG_DEBUG("HNSW size: ", hnsw_size, " bytes.");
    total_size += hnsw_size;
    size_t sa_size = 0;
    for (auto& s : gsa.st) {
        sa_size += s.next.bucket_count() * sizeof(void*);
        sa_size += (sizeof(std::pair<char, int>) + sizeof(void*)) * s.next.size(); // size of adjacency map
        sa_size += sizeof(s.len) + sizeof(s.link); // size of len and link
    }
    LOG_DEBUG("Suffix automaton size: ", sa_size, " bytes.");
    total_size += sa_size;
    size_t string_size = 0, vector_size = 0;
    for (int i = 0; i < num_elements; i++) {
        string_size += sizeof(std::string) + strs[i].capacity(); // size of each string
        vector_size += sizeof(float) * dim; // size of each vector
    }
    // LOG_DEBUG("String size: ", string_size, " bytes.");
    // LOG_DEBUG("Vector size: ", vector_size, " bytes.");
    // total_size += string_size + vector_size;
    // Auxiliary components
    size_t aux_size = sizeof(int) * gsa.st.size() * 3;
    for (int i = 0; i < gsa.st.size(); i++) {
        aux_size += sizeof(int) * candidate_ids[i].size();
    }
    LOG_DEBUG("Auxiliary components' size: ", aux_size, " bytes.");
    total_size += aux_size;
    return total_size;
}

size_t VectorMaton::vertex_num() {
    size_t total_vertices = 0;
    for (int i = 0; i < gsa.st.size(); i++) {
        if (!hnsws[i]) continue;
        total_vertices += candidate_ids[i].size();
        if (candidate_ids[i].size() != hnsws[i]->max_elements_) {
            LOG_ERROR("Vertex number for state ", i, " does not match!");
        }
    }
    return total_vertices;
}

void VectorMaton::set_ef(int ef) {
    for (int i = 0; i < gsa.st.size(); i++) {
        if (hnsws[i]) hnsws[i]->setEf(ef);
    }
}

void VectorMaton::set_min_build_threshold(int threshold) {
    min_build_threshold = threshold;
}

std::vector<int> VectorMaton::query(const float* vec, const std::string &s, int k) {
    int i = gsa.query(s);
    if (i == -1) return {};
    std::vector<std::pair<float, hnswlib::labeltype>> local_res;
    if (!hnsws[i]) {
        // No graph built on this state, brute-force
        for (int j = 0; j < candidate_ids[i].size(); j++) {
            int id = candidate_ids[i][j];
            local_res.emplace_back(distance(vecs.data() + id * dim, vec, dim), id);
        }
        std::sort(local_res.begin(), local_res.end());
        if (local_res.size() > k) local_res.resize(k);
    }
    else {
        hnsws[i]->external_data_ = (const char*)vecs.data();
        local_res = hnsws[i]->searchKnnCloserFirst(vec, k);
    }
    std::vector<std::pair<float, hnswlib::labeltype>> inherit_res;
    if (inherit_states.size() > 0 && inherit_states[i] != -1) {
        if (!hnsws[inherit_states[i]]) {
            LOG_ERROR("HNSW for state ", i, "'s inherited state ", inherit_states[i], " should have been built but is not built!");
        }
        hnsws[inherit_states[i]]->external_data_ = (const char*)vecs.data();
        inherit_res = hnsws[inherit_states[i]]->searchKnnCloserFirst(vec, k);
    }
    std::vector<int> results;
    int l = 0, r = 0;
    while ((l < local_res.size() || r < inherit_res.size()) && results.size() < k) {
        if (l >= local_res.size()) {
            results.emplace_back(inherit_res[r++].second);
        }
        else if (r >= inherit_res.size()) {
            results.emplace_back(local_res[l++].second);
        }
        else {
            if (local_res[l].first < inherit_res[r].first) {
                results.emplace_back(local_res[l++].second);
            }
            else {
                results.emplace_back(inherit_res[r++].second);
            }
        }
    }
    return results;
}

VectorMaton::~VectorMaton() {
    for (int i = 0; i < gsa.st.size(); i++) {
        if (hnsws[i]) delete hnsws[i];
    }
    delete space;
}

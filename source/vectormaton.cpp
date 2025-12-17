#include "vectormaton.h"

void VectorMaton::set_vectors(float** vectors, int dimension, int num_elems) {
    vecs = vectors;
    dim = dimension;
    num_elements = num_elems;
}

void VectorMaton::set_strings(std::string* strings) {
    strs = strings;
}

void VectorMaton::build_gsa() {
    LOG_DEBUG("Building Generalized Suffix Automaton (GSA)");
    unsigned long long start_time = currentTime();
    for (int i = 0; i < num_elements; i++) {
        gsa.add_string(i, strs[i]);
    }
    LOG_DEBUG("GSA built in ", timeFormatting(currentTime() - start_time).str());

    // Print GSA statistics
    auto stats = gsa.get_statistics();
    for (size_t depth = 0; depth < stats.size(); ++depth) {
        const auto &stat = stats[depth];
        LOG_DEBUG("Depth ", depth, ": num states = ", stat.sizes.size(),
                  ", median vector set size = ", stat.mid,
                  ", average vector set size = ", stat.avg);
    }
}

void VectorMaton::build_smart() {
    build_gsa();
    
    // Smart build will inherit info from children
    inherit_states = new int[gsa.st.size()];
    built = new bool[gsa.st.size()];
    candidate_ids = new std::vector<int>[gsa.st.size()];

    // Build graph index
    #if USE_HNSW
        hnsws = new hnswlib::HierarchicalNSW<float>*[gsa.st.size()];
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
            if (st.ids.size() < min_build_threshold) {
                continue;
            }
            // First find a children with largest materialized build
            int target_ch = -1;
            for (auto ch : st.next) {
                if (built[ch.second] && (target_ch == -1 || candidate_ids[ch.second].size() > candidate_ids[target_ch].size())) {
                    target_ch = ch.second;
                }
            }
            inherit_states[i] = target_ch;
            if (target_ch == -1) {
                int M = 16, ef_construction = 200;
                hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), M, ef_construction);
                for (auto id : st.ids) {
                    hnsws[i]->addPoint(vecs[id], id);
                    candidate_ids[i].emplace_back(id);
                }
                built[i] = true;
            }
            else {
                // Find remaining vertices
                int l = 0, r = 0;
                while (l < st.ids.size() || r < candidate_ids[target_ch].size()) {
                    if (r == candidate_ids[target_ch].size()) {
                        candidate_ids[i].emplace_back(st.ids[l++]);
                    }
                    else {
                        if (st.ids[l] == candidate_ids[target_ch][r]) {
                            l++, r++;
                        }
                        else {
                            candidate_ids[i].emplace_back(st.ids[l++]);
                        }
                    }
                }
                // Only build when meeting requirements
                if (candidate_ids[i].size() >= min_build_threshold) {
                    int M = 16, ef_construction = 200;
                    hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, candidate_ids[i].size(), M, ef_construction);
                    for (auto id : candidate_ids[i]) {
                        hnsws[i]->addPoint(vecs[id], id);
                    }
                    built[i] = true;
                }
            }
        }
    #else
        // TODO: implement
    #endif
}

void VectorMaton::build_full() {
    build_gsa();
    built = new bool[gsa.st.size()];

    // Build graph index
    #if USE_HNSW
        hnsws = new hnswlib::HierarchicalNSW<float>*[gsa.st.size()];
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
            if (st.ids.size() < min_build_threshold) {
                continue;
            }
            int M = 16, ef_construction = 200;
            hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), M, ef_construction);
            for (auto id : st.ids) {
                hnsws[i]->addPoint(vecs[id], id);
            }
            built[i] = true;
        }
    #else
        // TODO: implement
    #endif
}

size_t VectorMaton::size() {
    size_t total_size = 0;
    #if USE_HNSW
        for (int i = 0; i < gsa.st.size(); i++) {
            auto hnsw = hnsws[i];
            if (!hnsw) continue;
            total_size += hnsw->max_elements_ * hnsw->size_data_per_element_; // size of data_level0_memory_
            total_size += sizeof(void*) * hnsw->max_elements_; // size of linkLists_
            // Ignore the other variables since they are relatively small
        }
    #else
        for (int i = 0; i < gsa.st.size(); i++) {
            auto nsw = nsws[i];
            if (!nsw) continue;
            for (auto& node : nsw->nodes) {
                total_size += sizeof(node.id);
                total_size += sizeof(int) * node.neighbors.capacity();
            }
        }
    #endif
    for (auto& s : gsa.st) {
        total_size += sizeof(uint32_t) * s.ids.size(); // size of ids vector
        total_size += sizeof(std::pair<char, int>) * s.next.size(); // size of adjacency map
        total_size += sizeof(s.len) + sizeof(s.link); // size of len and link
    }
    for (int i = 0; i < num_elements; i++) {
        total_size += sizeof(std::string) + strs[i].capacity(); // size of each string
        total_size += sizeof(float) * dim; // size of each vector
    }
    return total_size;
}

size_t VectorMaton::vertex_num() {
    size_t total_vertices = 0;
    for (int i = 0; i < gsa.st.size(); i++) {
        if (!built) continue;
        if (candidate_ids) {
            total_vertices += candidate_ids[i].size();
        }
        else {
            total_vertices += gsa.st[i].ids.size();
        }
    }
    return total_vertices;
}

void VectorMaton::set_ef(int ef) {
    #if USE_HNSW
        for (int i = 0; i < gsa.st.size(); i++) {
            hnsws[i]->setEf(ef);
        }
    #else
        // TODO: implement
    #endif
}

void VectorMaton::set_min_build_threshold(int threshold) {
    min_build_threshold = threshold;
}

std::vector<int> VectorMaton::query(const float* vec, const std::string &s, int k) {
    int i = gsa.query(s);
    if (i == -1) return {};
    if (gsa.st[i].ids.size() < min_build_threshold) {
        // Search in brute-force
        std::vector<std::pair<float, int>> distances;
        for (auto id : gsa.st[i].ids) {
            distances.emplace_back(distance(vecs[id], vec, dim), id);
        }
        std::sort(distances.begin(), distances.end());
        std::vector<int> results;
        for (int j = 0; j < std::min(k, static_cast<int>(distances.size())); j++) {
            results.push_back(distances[j].second);
        }
        return results;
    }
    #if USE_HNSW
        std::vector<std::pair<float, hnswlib::labeltype>> local_res;
        if (built[i]) {
            local_res = hnsws[i]->searchKnnCloserFirst(vec, k);
        }
        else {
            // Brute force
            for (auto id : candidate_ids[i]) {
                local_res.emplace_back(distance(vecs[id], vec, dim), id);
            }
            std::sort(local_res.begin(), local_res.end());
            if (local_res.size() > k) local_res.resize(k);
        }
        std::vector<std::pair<float, hnswlib::labeltype>> inherit_res;
        if (inherit_states && inherit_states[i] != -1) {
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
    #else
        // TODO: implement
    #endif
}
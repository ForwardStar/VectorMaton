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
    LOG_DEBUG("Total GSA states: ", std::to_string(gsa.size()), ", total string IDs in GSA: ", std::to_string(gsa.size_tot()));

    // Print GSA statistics
    auto stats = gsa.get_statistics();
    for (size_t depth = 0; depth < stats.size(); ++depth) {
        const auto &stat = stats[depth];
        LOG_DEBUG("Depth ", depth, ": num states = ", stat.sizes.size(),
                  ", median vector set size = ", stat.mid,
                  ", average vector set size = ", stat.avg);
    }
}

void VectorMaton::clear_gsa() {
    for (int i = 0; i < gsa.st.size(); i++) {
        gsa.st[i].ids = std::vector<uint32_t>();
    }
}

void VectorMaton::build_smart() {
    build_gsa();
    
    // Smart build will inherit info from children
    inherit_states = new int[gsa.st.size()];
    size_ids = new int[gsa.st.size()];
    candidate_ids = new int*[gsa.st.size()];
    int* largest_state = new int[gsa.st.size()];
    for (int i = 0; i < gsa.st.size(); i++) {
        inherit_states[i] = -1, largest_state[i] = -1, size_ids[i] = 0, candidate_ids[i] = nullptr;
    }

    // Build graph index
    #if USE_HNSW
        hnsws = new hnswlib::HierarchicalNSW<float>*[gsa.st.size()];
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
                size_ids[i] = st.ids.size();
                candidate_ids[i] = new int[size_ids[i]];
                for (int j = 0; j < size_ids[i]; j++) {
                    candidate_ids[i][j] = st.ids[j];
                }
                continue;
            }
            // First find a successor with largest built graph
            int target_sc = -1;
            for (auto ch : st.next) {
                if (largest_state[ch.second] != -1 && (target_sc == -1 || size_ids[largest_state[ch.second]] > size_ids[target_sc])) {
                    target_sc = largest_state[ch.second];
                }
            }
            inherit_states[i] = target_sc;
            if (target_sc == -1) {
                // No successor has built a graph, built the graph with all vector ids
                int M = 16, ef_construction = 200;
                hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), M, ef_construction);
                size_ids[i] = st.ids.size();
                candidate_ids[i] = new int[size_ids[i]];
                for (int j = 0; j < size_ids[i]; j++) {
                    int id = st.ids[j];
                    hnsws[i]->addPoint(vecs[id], id);    
                    candidate_ids[i][j] = id;
                }
                largest_state[i] = i;
            }
            else {
                // Found the largest successor, inherit from this successor
                inherit_states[i] = target_sc;
                largest_state[i] = target_sc;
                // Find remaining vertices
                int l = 0, r = 0, cnt = 0;
                size_ids[i] = st.ids.size() - size_ids[target_sc];
                candidate_ids[i] = new int[size_ids[i]];
                while (l < st.ids.size() || r < size_ids[target_sc]) {
                    if (r == size_ids[target_sc]) {
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
                if (size_ids[i] >= min_build_threshold) {
                    int M = 16, ef_construction = 200;
                    hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, size_ids[i], M, ef_construction);
                    for (int j = 0; j < size_ids[i]; j++) {
                        int id = candidate_ids[i][j];
                        hnsws[i]->addPoint(vecs[id], id);
                    }
                    if (size_ids[i] > size_ids[target_sc]) {
                        // Update largest state
                        largest_state[i] = i;
                    }
                }
            }
        }
    #else
        // TODO: implement
    #endif

    delete [] largest_state;
    clear_gsa();
}

void VectorMaton::build_full() {
    build_gsa();
    size_ids = new int[gsa.st.size()];
    candidate_ids = new int*[gsa.st.size()];
    for (int i = 0; i < gsa.st.size(); i++) {
        size_ids[i] = 0, candidate_ids[i] = nullptr;
    }

    // Build graph index
    #if USE_HNSW
        hnsws = new hnswlib::HierarchicalNSW<float>*[gsa.st.size()];
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
            size_ids[i] = st.ids.size();
            candidate_ids[i] = new int[size_ids[i]];
            for (int j = 0; j < size_ids[i]; j++) {
                candidate_ids[i][j] = st.ids[j];
            }
            if (st.ids.size() < min_build_threshold) {
                continue;
            }
            int M = 16, ef_construction = 200;
            hnsws[i] = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), M, ef_construction);
            for (auto id : st.ids) {
                hnsws[i]->addPoint(vecs[id], id);
            }
        }
    #else
        // TODO: implement
    #endif
    
    clear_gsa();
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
        #ifdef USE_HNSW
            if (!hnsws[i]) continue;
        #else
            if (!nsws[i]) continue;
        #endif
        total_vertices += size_ids[i];
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
    #if USE_HNSW
        std::vector<std::pair<float, hnswlib::labeltype>> local_res;
        if (!hnsws[i]) {
            // No graph built on this state, brute-force
            for (int j = 0; j < size_ids[i]; j++) {
                int id = candidate_ids[i][j];
                local_res.emplace_back(distance(vecs[id], vec, dim), id);
            }
            std::sort(local_res.begin(), local_res.end());
            if (local_res.size() > k) local_res.resize(k);
        }
        else {
            local_res = hnsws[i]->searchKnnCloserFirst(vec, k);
        }
        std::vector<std::pair<float, hnswlib::labeltype>> inherit_res;
        if (inherit_states && inherit_states[i] != -1) {
            if (!hnsws[inherit_states[i]]) {
                LOG_ERROR("HNSW for state ", i, "'s inherited state ", inherit_states[i], " should have been built but is not built!");
            }
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
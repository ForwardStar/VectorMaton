#include "vectormaton.h"

#if USE_HNSW
    hnswlib::HierarchicalNSW<float>* VectorMaton::deepCopyHNSW(const hnswlib::HierarchicalNSW<float>& orig, int num_elements) {
        using namespace hnswlib;
        // Allocate new index with the same max_elements
        auto* copy = new HierarchicalNSW<float>(space, num_elements);

        // Copy basic parameters
        copy->cur_element_count = orig.cur_element_count.load();
        copy->maxlevel_ = orig.maxlevel_;
        copy->enterpoint_node_ = orig.enterpoint_node_;
        copy->maxM_ = orig.maxM_;
        copy->maxM0_ = orig.maxM0_;
        copy->M_ = orig.M_;
        copy->mult_ = orig.mult_;
        copy->ef_construction_ = orig.ef_construction_;
        copy->size_data_per_element_ = orig.size_data_per_element_;
        copy->size_links_per_element_ = orig.size_links_per_element_;
        copy->revSize_ = orig.revSize_;
        copy->ef_ = orig.ef_;
        copy->offsetLevel0_ = orig.offsetLevel0_;
        copy->offsetData_ = orig.offsetData_;
        copy->label_offset_ = orig.label_offset_;
        copy->data_size_ = orig.data_size_;
        copy->fstdistfunc_ = orig.fstdistfunc_;
        copy->dist_func_param_ = orig.dist_func_param_;

        // Allocate and copy data_level0_memory_
        copy->data_level0_memory_ = (char*)malloc(copy->max_elements_ * copy->size_data_per_element_);
        if (!copy->data_level0_memory_)
            throw std::runtime_error("Not enough memory for deep copy of data_level0_memory_");
        std::memcpy(copy->data_level0_memory_, orig.data_level0_memory_,
                    copy->cur_element_count * copy->size_data_per_element_);

        // Copy element levels
        copy->element_levels_ = orig.element_levels_;

        // Initialize linkLists_
        copy->linkLists_ = (char**)malloc(sizeof(void*) * copy->max_elements_);
        if (!copy->linkLists_)
            throw std::runtime_error("Not enough memory for deep copy of linkLists_");

        // Copy each link list
        for (size_t i = 0; i < copy->cur_element_count; i++) {
            unsigned int linkListSize = copy->element_levels_[i] > 0
                                        ? copy->size_links_per_element_ * copy->element_levels_[i]
                                        : 0;
            if (linkListSize == 0) {
                copy->linkLists_[i] = nullptr;
            } else {
                copy->linkLists_[i] = (char*)malloc(linkListSize);
                if (!copy->linkLists_[i])
                    throw std::runtime_error("Not enough memory for link list deep copy");
                std::memcpy(copy->linkLists_[i], orig.linkLists_[i], linkListSize);
            }
        }

        // Deep copy visited_list_pool_
        copy->visited_list_pool_.reset(new VisitedListPool(1, copy->max_elements_));

        // Deep copy label lookup
        copy->label_lookup_ = orig.label_lookup_;

        // Reinitialize locks (mutex cannot be copied)
        copy->link_list_locks_ = std::vector<std::mutex>(copy->max_elements_);
        copy->label_op_locks_ = std::vector<std::mutex>(copy->MAX_LABEL_OPERATION_LOCKS);

        // Copy deleted elements set
        copy->deleted_elements = orig.deleted_elements;
        copy->num_deleted_ = orig.num_deleted_.load();
        copy->allow_replace_deleted_ = orig.allow_replace_deleted_;

        return copy;
    }
#endif

void VectorMaton::set_vectors(float** vectors, int dimension, int num_elems) {
    vecs = vectors;
    dim = dimension;
    num_elements = num_elems;
}

void VectorMaton::set_strings(std::string* strings) {
    strs = strings;
}

void VectorMaton::build_partial(double shrink_factor) {
    expand_rate = 1.0 / shrink_factor;
    // Build GSA
    LOG_DEBUG("Building Generalized Suffix Automaton (GSA)");
    unsigned long long start_time = currentTime();
    for (int i = 0; i < num_elements; i++) {
        gsa.add_string(i, strs[i]);
    }
    LOG_DEBUG("GSA built in ", timeFormatting(currentTime() - start_time).str());
    last_state_in_gsa = new int[gsa.st.size()];
    #if USE_HNSW
        if (!space) {
            space = new hnswlib::L2Space(dim);
        }
        int* build_threshold = new int[gsa.st.size()];
        for (int i = 0; i < gsa.st.size(); i++) {
            build_threshold[i] = 2147483647; // INT_MAX
        }
        int i = 0, ten_percent = gsa.st.size() / 10;
        for (int i = 0; i < gsa.st.size(); i++) {
            auto& st = gsa.st[i];
            st.hash_value = sethash::sha256_hash(st.ids);
            if (i % ten_percent == 0) {
                LOG_DEBUG("Building HNSW for state ", i, "/", gsa.st.size());
            }
            if (build_threshold[i] > gsa.st[i].ids.size()) {
                // Build HNSW for this state
                auto tmp = new hnswlib::HierarchicalNSW<float>(space, gsa.st[i].ids.size(), 16, 200);
                hnsws[gsa.st[i].hash_value] = tmp;
                for (auto id : gsa.st[i].ids) {
                    tmp->addPoint(vecs[id], id);
                }
                // Propagate build threshold to children
                int new_threshold = static_cast<int>(gsa.st[i].ids.size() * shrink_factor);
                for (auto& [key, value] : gsa.st[i].next) {
                    if (build_threshold[value] > new_threshold) {
                        build_threshold[value] = new_threshold;
                        last_state_in_gsa[value] = i;
                    }
                }
            }
            else {
                for (auto& [key, value] : gsa.st[i].next) {
                    if (build_threshold[value] > build_threshold[i]) {
                        build_threshold[value] = build_threshold[i];
                        last_state_in_gsa[value] = last_state_in_gsa[i];
                    }
                }
            }
        }
        delete [] build_threshold;
    #else
        int* build_threshold = new int[gsa.st.size()];
        for (int i = 0; i < gsa.st.size(); i++) {
            build_threshold[i] = 2147483647; // INT_MAX
        }
        int i = 0, ten_percent = gsa.st.size() / 10;
        for (int i = 0; i < gsa.st.size(); i++) {
            auto& st = gsa.st[i];
            st.hash_value = sethash::sha256_hash(st.ids);
            if (i % ten_percent == 0) {
                LOG_DEBUG("Building NSW for state ", i, "/", gsa.st.size());
            }
            if (build_threshold[i] > gsa.st[i].ids.size()) {
                // Build NSW for this state
                auto tmp = new NSW(vecs, dim);
                nsws[gsa.st[i].hash_value] = tmp;
                for (auto id : gsa.st[i].ids) {
                    tmp->insert(id);
                }
                // Propagate build threshold to children
                int new_threshold = static_cast<int>(gsa.st[i].ids.size() * shrink_factor);
                for (auto& [key, value] : gsa.st[i].next) {
                    if (build_threshold[value] > new_threshold) {
                        build_threshold[value] = new_threshold;
                        last_state_in_gsa[value] = i;
                    }
                }
            }
            else {
                for (auto& [key, value] : gsa.st[i].next) {
                    if (build_threshold[value] > build_threshold[i]) {
                        build_threshold[value] = build_threshold[i];
                        last_state_in_gsa[value] = last_state_in_gsa[i];
                    }
                }
            }
        }
        delete [] build_threshold;
    #endif
}

void VectorMaton::build() {
    // Build GSA
    LOG_DEBUG("Building Generalized Suffix Automaton (GSA)");
    unsigned long long start_time = currentTime();
    for (int i = 0; i < num_elements; i++) {
        gsa.add_string(i, strs[i]);
    }
    LOG_DEBUG("GSA built in ", timeFormatting(currentTime() - start_time).str());
    #if USE_HNSW
        if (!space) {
            space = new hnswlib::L2Space(dim);
        }
        int i = 0, ten_percent = gsa.st.size() / 10;
        for (int i = gsa.st.size() - 1; i >= 0; i--) {
            if (i % ten_percent == 0) {
                LOG_DEBUG("Building HNSW for state ", gsa.st.size() - i, "/", gsa.st.size());
            }
            auto& st = gsa.st[i];
            st.hash_value = sethash::sha256_hash(st.ids);
            if (hnsws.find(st.hash_value) == hnsws.end()) {
                // Reuse existing HNSW
                int largest_state = -1;
                for (auto& [key, value] : st.next) {
                    if (value > i && (largest_state == -1 || gsa.st[value].ids.size() > gsa.st[largest_state].ids.size())) {
                        largest_state = value;
                    }
                }
                #if ENABLE_DEEP_COPY_HNSW
                if (largest_state != -1) {
                    auto tmp = deepCopyHNSW(*hnsws[gsa.st[largest_state].hash_value], st.ids.size());
                    hnsws[st.hash_value] = tmp;
                    // Add vectors that are in st.ids but not in the copied HNSW
                    int l = 0, r = 0;
                    while (l < st.ids.size() || r < gsa.st[largest_state].ids.size()) {
                        if (r == gsa.st[largest_state].ids.size()) {
                            tmp->addPoint(vecs[st.ids[l]], st.ids[l]);
                            l++;
                        } else if (l == st.ids.size()) {
                            // Should not happen
                            LOG_WARN("Unexpected condition in VectorMaton::build(): child state has ids not in parent state");
                            break;
                        } else if (st.ids[l] < gsa.st[largest_state].ids[r]) {
                            tmp->addPoint(vecs[st.ids[l]], st.ids[l]);
                            l++;
                        } else if (st.ids[l] > gsa.st[largest_state].ids[r]) {
                            r++;
                        } else {
                            l++;
                            r++;
                        }
                    }
                } else 
                #endif
                {
                    int M = 16, ef_construction = 200;
                    auto tmp = new hnswlib::HierarchicalNSW<float>(space, st.ids.size(), M, ef_construction);
                    hnsws[st.hash_value] = tmp;
                    for (auto id : st.ids) {
                        tmp->addPoint(vecs[id], id);
                    }
                }
            }
        }
    #else
        int i = 0, ten_percent = gsa.st.size() / 10;
        for (int i = gsa.st.size() - 1; i >= 0; i--) {
            if (i % ten_percent == 0) {
                LOG_DEBUG("Building NSW for state ", gsa.st.size() - i, "/", gsa.st.size());
            }
            auto& st = gsa.st[i];
            st.hash_value = sethash::sha256_hash(st.ids);
            if (nsws.find(st.hash_value) == nsws.end()) {
                // Reuse existing NSW
                int largest_state = -1;
                for (auto& [key, value] : st.next) {
                    if (value > i && (largest_state == -1 || gsa.st[value].ids.size() > gsa.st[largest_state].ids.size())) {
                        largest_state = value;
                    }
                }
                if (largest_state != -1) {
                    auto tmp = new NSW(*nsws[gsa.st[largest_state].hash_value]);
                    nsws[st.hash_value] = tmp;
                    // Add vectors that are in st.ids but not in the copied NSW
                    int l = 0, r = 0;
                    while (l < st.ids.size() || r < gsa.st[largest_state].ids.size()) {
                        if (r == gsa.st[largest_state].ids.size()) {
                            tmp->insert(st.ids[l]);
                            l++;
                        } else if (l == st.ids.size()) {
                            // Should not happen
                            LOG_WARN("Unexpected condition in VectorMaton::build(): child state has ids not in parent state");
                            break;
                        } else if (st.ids[l] < gsa.st[largest_state].ids[r]) {
                            tmp->insert(st.ids[l]);
                            l++;
                        } else if (st.ids[l] > gsa.st[largest_state].ids[r]) {
                            r++;
                        } else {
                            l++;
                            r++;
                        }
                    }
                } else {
                    auto tmp = new NSW(vecs, dim);
                    nsws[st.hash_value] = tmp;
                    for (auto id : st.ids) {
                        tmp->insert(id);
                    }
                }
            }
        }
    #endif
}

size_t VectorMaton::size() {
    size_t total_size = 0;
    #if USE_HNSW
        for (auto hnsw : hnsws) {
            total_size += hnsw.second->max_elements_ * hnsw.second->size_data_per_element_; // size of data_level0_memory_
            total_size += sizeof(void*) * hnsw.second->max_elements_; // size of linkLists_
            // Ignore the other variables since they are relatively small
        }
    #else
        for (auto nsw : nsws) {
            for (auto& node : nsw.second->nodes) {
                total_size += sizeof(node.id);
                total_size += sizeof(int) * node.neighbors.capacity();
            }
        }
    #endif
    for (auto& s : gsa.st) {
        total_size += sizeof(uint32_t) * s.ids.size(); // size of ids vector
        total_size += sizeof(std::pair<char, int>) * s.next.size(); // size of adjacency map
        total_size += sizeof(s.len) + sizeof(s.link); // size of len and link
        total_size += s.hash_value.capacity(); // size of hash_value string
    }
    for (int i = 0; i < num_elements; i++) {
        total_size += sizeof(std::string) + strs[i].capacity(); // size of each string
        total_size += sizeof(float) * dim; // size of each vector
    }
    return total_size;
}

std::vector<int> VectorMaton::query(const float* vec, const std::string &s, int k) {
    int i = gsa.query(s);
    if (i == -1) return {};
    #if USE_HNSW
        if (hnsws.find(gsa.st[i].hash_value) == hnsws.end()) {
            double expand_factor = double(gsa.st[last_state_in_gsa[i]].ids.size()) / double(gsa.st[i].ids.size());
            expand_factor = std::max(expand_factor, expand_rate);
            auto results = hnsws[gsa.st[last_state_in_gsa[i]].hash_value]->searchKnnCloserFirst(vec, k * expand_factor);
            // Filter results to only those in gsa.st[i].ids
            std::vector<int> filtered_results;
            for (auto& pair : results) {
                if (std::binary_search(gsa.st[i].ids.begin(), gsa.st[i].ids.end(), pair.second)) {
                    filtered_results.push_back(pair.second);
                    if (filtered_results.size() >= static_cast<size_t>(k)) {
                        break;
                    }
                }
            }
            return filtered_results;
        }
        auto tmp = hnsws[gsa.st[i].hash_value]->searchKnnCloserFirst(vec, k);
        std::vector<int> results;
        for (auto& pair : tmp) {
            results.push_back(pair.second);
        }
        return results;
    #else
        if (nsws.find(gsa.st[i].hash_value) == nsws.end()) {
            double expand_factor = double(gsa.st[last_state_in_gsa[i]].ids.size()) / double(gsa.st[i].ids.size());
            expand_factor = std::max(expand_factor, expand_rate);
            auto results = nsws[gsa.st[last_state_in_gsa[i]].hash_value]->searchKNN(vec, k * expand_factor);
            // Filter results to only those in gsa.st[i].ids
            std::vector<int> filtered_results;
            for (auto id : results) {
                if (std::binary_search(gsa.st[i].ids.begin(), gsa.st[i].ids.end(), id)) {
                    filtered_results.push_back(id);
                    if (filtered_results.size() >= static_cast<size_t>(k)) {
                        break;
                    }
                }
            }
            return filtered_results;
        }
        return nsws[gsa.st[i].hash_value]->searchKNN(vec, k);
    #endif
}
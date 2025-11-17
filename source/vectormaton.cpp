#include "vectormaton.h"

#if USE_HNSW
    hnswlib::HierarchicalNSW<float>* deepCopyHNSW(const hnswlib::HierarchicalNSW<float>& orig) {
        using namespace hnswlib;
        auto space = new L2Space(orig.data_size_); // Create a new space with the same data size

        // Allocate new index with the same max_elements
        auto* copy = new HierarchicalNSW<float>(space, orig.max_elements_);

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
        copy->visited_list_pool_.reset(new VisitedListPool(1, orig.max_elements_));

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
                if (largest_state != -1) {
                    auto tmp = deepCopyHNSW(*hnsws[gsa.st[largest_state].hash_value]);
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
                } else {
                    int M = 16, ef_construction = 200;
                    auto tmp = new hnswlib::HierarchicalNSW<float>(space, num_elements, M, ef_construction);
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

std::vector<int> VectorMaton::query(const float* vec, const std::string &s, int k) {
    int i = gsa.query(s);
    if (i == -1) return {};
    #if USE_HNSW
        auto tmp = hnsws[gsa.st[i].hash_value]->searchKnnCloserFirst(vec, k);
        std::vector<int> results;
        for (auto& pair : tmp) {
            results.push_back(pair.second);
        }
        return results;
    #else
        return nsws[gsa.st[i].hash_value]->searchKNN(vec, k);
    #endif
}
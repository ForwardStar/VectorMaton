#include "baseline.h"

void Baseline::set_vectors(float** vectors, int dimension, int num_elems) {
    vecs = vectors;
    dim = dimension;
    num_elements = num_elems;
}

void Baseline::set_strings(std::string* strings) {
    strs = strings;
}

void Baseline::build() {
    #if USE_HNSW
        if (!hnsw) {
            space = new hnswlib::L2Space(dim);
            hnsw = new hnswlib::HierarchicalNSW<float>(space, num_elements, 16, 200);
            for (int i = 0; i < num_elements; i++) {
                hnsw->addPoint(vecs[i], i);
            }
        }
    #else
        if (!nsw) {
            nsw = new NSW(vecs, dim);
            for (int i = 0; i < num_elements; i++) {
                nsw->insert(i);
            }
        }
    #endif
}

size_t Baseline::size() {
    size_t total_size = 0;
    #if USE_HNSW
        total_size += hnsw->max_elements_ * hnsw->size_data_per_element_; // size of data_level0_memory_
        total_size += sizeof(void*) * hnsw->max_elements_; // size of linkLists_
        // Ignore the other variables since they are relatively small
    #else
        for (auto& node : nsw->nodes) {
            total_size += sizeof(node.id);
            total_size += sizeof(int) * node.neighbors.capacity();
        }
    #endif
    for (int i = 0; i < num_elements; i++) {
        total_size += sizeof(std::string) + strs[i].capacity(); // size of each string
        total_size += sizeof(float) * dim; // size of each vector
    }
    return total_size;
}

std::vector<int> Baseline::query(const float* vec, const std::string &s, int k, int threshold) {
    std::vector<int> results;
    int amplification = 2; // To improve recall, search for more candidates
    while (results.size() < k) {
        results.clear();
        #if USE_HNSW
            auto tmp = hnsw->searchKnnCloserFirst(vec, k * amplification);
            for (auto& pair : tmp) {
                int id = pair.second;
                if (strs[id].find(s) != std::string::npos) {
                    results.push_back(id);
                }
                if (results.size() >= static_cast<size_t>(k)) {
                    break;
                }
            }
        #else
            auto tmp = nsw->searchKNN(vec, k * amplification); // Get more candidates to improve recall
            for (uint32_t id : tmp) {
                if (strs[id].find(s) != std::string::npos) {
                    results.push_back(id);
                }
                if (results.size() >= static_cast<size_t>(k)) {
                    break;
                }
            }
        #endif
        amplification *= 2; // Double the amplification factor
        if (amplification > threshold) { // Avoid too large amplification
            break;
        }
    }
    return results;
}
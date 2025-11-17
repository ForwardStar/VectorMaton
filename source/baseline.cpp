#include "baseline.h"

int Baseline::insert(const std::vector<float>& vec, const std::string &s) {
    #if USE_HNSW
        if (!hnsw) {
            int dim = vec.size(), max_elements = 1e7, M = 16, ef_construction = 200;
            hnswlib::L2Space space(dim);
            hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        }
    #else
        if (!nsw) {
            nsw = new NSW(vecs);
        }
    #endif
    strs.emplace_back(s);
    vecs.emplace_back(vec);
    #if USE_HNSW
        float* data_ptr = new float[vec.size()];
        for (size_t i = 0; i < vec.size(); i++) {   
            data_ptr[i] = vec[i];
        }
        data_ptrs.emplace_back(data_ptr);
        hnsw->addPoint(data_ptr, strs.size() - 1);
    #else
        nsw->insert(strs.size() - 1);
    #endif
    return strs.size() - 1;
}

void Baseline::remove(int id) {
    // Work in progress
}

std::vector<int> Baseline::query(const std::vector<float>& vec, const std::string &s, int k, int threshold) {
    std::vector<int> results;
    int amplification = 2; // To improve recall, search for more candidates
    while (results.size() < k) {
        results.clear();
        #if USE_HNSW
            auto tmp = hnsw->searchKnnCloserFirst((void*)vec.data(), k * amplification);
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
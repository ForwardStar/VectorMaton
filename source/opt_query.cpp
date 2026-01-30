#include "opt_query.h"

void OptQuery::set_vectors(float* vectors, int dimension, int num_elems) {
    vecs = vectors;
    dim = dimension;
    num_elements = num_elems;
}

void OptQuery::set_strings(std::string* strings) {
    strs = strings;
}

void OptQuery::set_ef(int ef) {
    for (auto &pair : hnsw) {
        pair.second->setEf(ef);
    }
}

void OptQuery::build() {
    for (int i = 0; i < num_elements; i++) {
        for (int j = 0; j < strs[i].size(); j++) {
            for (int k = 1; k <= strs[i].size() - j; k++) {
                std::string substring = strs[i].substr(j, k);
                if (hnsw.find(substring) == hnsw.end()) {
                    // Create new HNSW index for this substring
                    space = new hnswlib::L2Space(dim);
                    hnsw[substring] = new hnswlib::HierarchicalNSW<float>(space, num_elements, vecs, 16, 200);
                }
                if (str_to_ids.find(substring) == str_to_ids.end()) {
                    str_to_ids[substring] = std::unordered_set<int>();
                }
                if (str_to_ids[substring].find(i) == str_to_ids[substring].end()) {
                    str_to_ids[substring].insert(i);
                    hnsw[substring]->addPoint(i);
                }
            }
        }
    }
    str_to_ids = std::unordered_map<std::string, std::unordered_set<int>>(); // free memory
}

void OptQuery::load_index(const char* input_folder) {
    // TODO
}

void OptQuery::save_index(const char* output_folder) {
    // TODO
}

size_t OptQuery::size() {
    size_t total_size = 0;
    // HNSW sizes
    for (auto &pair : hnsw) {
        total_size += pair.second->indexFileSize();
    }
    // unordered_map sizes
    for (auto &pair : hnsw) {
        total_size += sizeof(std::string) + pair.first.size() * sizeof(char); // key size
        total_size += sizeof(hnswlib::HierarchicalNSW<float>*); // value size
    }
    return total_size;
}

std::vector<int> OptQuery::query(const float* vec, const std::string &s, int k, int ef_search) {
    std::vector<int> results;
    if (hnsw.find(s) == hnsw.end()) {
        return results;
    }
    if (ef_search != 0) {
        hnsw[s]->setEf(ef_search);
    }
    auto tmp = hnsw[s]->searchKnnCloserFirst(vec, k);
    for (auto& pair : tmp) {
        results.push_back(pair.second);
    }
    return results;
}
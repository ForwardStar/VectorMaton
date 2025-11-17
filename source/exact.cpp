#include "exact.h"

void ExactSearch::set_vectors(float** vectors, int dimension, int max_elems) {
    vecs = vectors;
    dim = dimension;
    max_elements = max_elems;
}

void ExactSearch::set_strings(std::string* strings) {
    strs = strings;
}

std::vector<int> ExactSearch::query(const float* vec, const std::string &s, int k) {
    std::vector<int> results;
    for (int i = 0; i < max_elements; ++i) {
        if (strs[i].find(s) != std::string::npos) {
            results.push_back(i);
        }
    }
    std::sort(results.begin(), results.end(), [&](int a, int b) {
        return distance(vecs[a], vec, dim) < distance(vecs[b], vec, dim);
    });
    if (results.size() > k) {
        results.resize(k);
    }
    return results;
}
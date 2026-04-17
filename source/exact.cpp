#include "exact.h"

void ExactSearch::set_vectors(const std::vector<float>& vectors, int dimension) {
    vecs = vectors;
    dim = dimension;
    max_elements = dim == 0 ? 0 : static_cast<int>(vecs.size()) / dim;
}

void ExactSearch::set_strings(const std::vector<std::string>& strings) {
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
        return distance(vecs.data() + a * dim, vec, dim) < distance(vecs.data() + b * dim, vec, dim);
    });
    if (results.size() > k) {
        results.resize(k);
    }
    return results;
}

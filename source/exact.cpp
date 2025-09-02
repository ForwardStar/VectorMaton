#include "exact.h"

int ExactSearch::insert(const std::vector<float>& vec, const std::string &s) {
    strs.emplace_back(s);
    vecs.emplace_back(vec);
    return strs.size() - 1;
}

void ExactSearch::remove(int id) {
    // Work in progress
}

std::vector<int> ExactSearch::query(const std::vector<float>& vec, const std::string &s, int k) {
    std::vector<int> results;
    for (int i = 0; i < strs.size(); ++i) {
        if (strs[i].find(s) != std::string::npos) {
            results.push_back(i);
        }
    }
    std::sort(results.begin(), results.end(), [&](int a, int b) {
        return distance(vecs[a], vec) < distance(vecs[b], vec);
    });
    if (results.size() > k) {
        results.resize(k);
    }
    return results;
}
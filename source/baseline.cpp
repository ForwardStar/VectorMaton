#include "baseline.h"

int Baseline::insert(const std::vector<float>& vec, const std::string &s) {
    if (!nsw) {
        nsw = new NSW(vecs);
    }
    strs.emplace_back(s);
    vecs.emplace_back(vec);
    nsw->insert(strs.size() - 1);
    return strs.size() - 1;
}

void Baseline::remove(int id) {
    // Work in progress
}

std::vector<int> Baseline::query(const std::vector<float>& vec, const std::string &s, int k) {
    std::vector<int> results;
    auto tmp = nsw->searchKNN(vec, k * 5); // Get more candidates to improve recall
    for (int id : tmp) {
        if (strs[id].find(s) != std::string::npos) {
            results.push_back(id);
        }
        if (results.size() >= static_cast<size_t>(k)) {
            break;
        }
    }
    return results;
}
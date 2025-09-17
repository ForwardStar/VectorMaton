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

std::vector<int> Baseline::query(const std::vector<float>& vec, const std::string &s, int k, int threshold) {
    std::vector<int> results;
    int amplification = 2; // To improve recall, search for more candidates
    while (results.size() < k) {
        results.clear();
        auto tmp = nsw->searchKNN(vec, k * amplification); // Get more candidates to improve recall
        for (uint32_t id : tmp) {
            if (strs[id].find(s) != std::string::npos) {
                results.push_back(id);
            }
            if (results.size() >= static_cast<size_t>(k)) {
                break;
            }
        }
        amplification *= 2; // Double the amplification factor
        if (amplification > threshold) { // Avoid too large amplification
            break;
        }
    }
    return results;
}
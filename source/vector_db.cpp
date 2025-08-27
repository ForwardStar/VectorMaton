#include "vector_db.h"

int VectorDB::insert(const std::vector<float>& vec, const std::string &s) {
    gsa.add_string(strs.size(), s);
    strs.emplace_back(s);
    vecs.emplace_back(vec);
    for (auto i : gsa.affected_states) {
        if (i < nsws.size()) {
            nsws[i].insert(gsa.st[i].ids.back());
        }
    }
    for (int i = nsws.size(); i < gsa.st.size(); i++) {
        nsws.emplace_back(NSW(vecs));
        for (auto id : gsa.st[i].ids) {
            nsws.back().insert(id);
        }
    }
    return strs.size() - 1;
}

void VectorDB::remove(int id) {
    // Work in progress
}

std::vector<int> VectorDB::query(const std::vector<float>& vec, const std::string &s, int k) {
    int i = gsa.query(s);
    if (i == -1) return {};
    return nsws[i].searchKNN(vec, k);
}
#include "vector_db.h"

int VectorDB::insert(const std::vector<float>& vec, const std::string &s) {
    sa.add_string(strs.size(), s);
    strs.emplace_back(s);
    vecs.emplace_back(vec);
    for (auto i : sa.affected_states) {
        if (i < nsws.size()) {
            nsws[i].insert(sa.st[i].ids.back());
        }
    }
    for (int i = nsws.size(); i < sa.st.size(); i++) {
        nsws.emplace_back(NSW(vecs));
        for (auto id : sa.st[i].ids) {
            nsws.back().insert(id);
        }
    }
}

void VectorDB::remove(int id) {
    // Work in progress
}

std::vector<int> VectorDB::query(const std::vector<float>& vec, const std::string &s, int k) {
    int i = sa.query(s);
    if (i == -1) return {};
    return nsws[i].searchKNN(vec, k);
}
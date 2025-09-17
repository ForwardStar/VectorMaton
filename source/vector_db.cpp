#include "vector_db.h"

int VectorDB::insert(const std::vector<float>& vec, const std::string &s) {
    gsa.add_string(strs.size(), s);
    strs.emplace_back(s);
    vecs.emplace_back(vec);
    return strs.size() - 1;
}

void VectorDB::build() {
    nsws.clear();
    for (auto& st : gsa.st) {
        st.hash_value = sethash::sha256_hash(st.ids);
        if (nsws.find(st.hash_value) == nsws.end()) {
            auto tmp = new NSW(vecs);
            nsws[st.hash_value] = tmp;
            for (uint32_t id : st.ids) {
                tmp->insert(id);
            }
        }
    }
}

void VectorDB::remove(int id) {
    // Work in progress
}

std::vector<int> VectorDB::query(const std::vector<float>& vec, const std::string &s, int k) {
    int i = gsa.query(s);
    if (i == -1) return {};
    return nsws[gsa.st[i].hash_value]->searchKNN(vec, k);
}
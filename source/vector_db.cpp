#include "vector_db.h"

int VectorDB::insert(const std::vector<float>& vec, const std::string &s) {
    gsa.add_string(strs.size(), s);
    strs.emplace_back(s);
    vecs.emplace_back(vec);
    return strs.size() - 1;
}

void VectorDB::build() {
    nsws.clear();
    int i = 0, ten_percent = gsa.st.size() / 10;
    for (int i = gsa.st.size() - 1; i >= 0; i--) {
        if (i % ten_percent == 0) {
            LOG_DEBUG("Building NSW for state ", gsa.st.size() - i, "/", gsa.st.size());
        }
        auto& st = gsa.st[i];
        st.hash_value = sethash::sha256_hash(st.ids);
        if (nsws.find(st.hash_value) == nsws.end()) {
            // Reuse existing NSW
            int largest_state = -1;
            for (auto& [key, value] : st.next) {
                if (value > i && (largest_state == -1 || gsa.st[value].ids.size() > gsa.st[largest_state].ids.size())) {
                    largest_state = value;
                }
            }
            if (largest_state != -1) {
                auto tmp = new NSW(*nsws[gsa.st[largest_state].hash_value]);
                nsws[st.hash_value] = tmp;
                // Add vectors that are in st.ids but not in the copied NSW
                int l = 0, r = 0;
                while (l < st.ids.size() || r < gsa.st[largest_state].ids.size()) {
                    if (r == gsa.st[largest_state].ids.size()) {
                        tmp->insert(st.ids[l]);
                        l++;
                    } else if (l == st.ids.size()) {
                        // Should not happen
                        LOG_WARN("Unexpected condition in VectorDB::build(): child state has ids not in parent state");
                        break;
                    } else if (st.ids[l] < gsa.st[largest_state].ids[r]) {
                        tmp->insert(st.ids[l]);
                        l++;
                    } else if (st.ids[l] > gsa.st[largest_state].ids[r]) {
                        r++;
                    } else {
                        l++;
                        r++;
                    }
                }
            } else {
                auto tmp = new NSW(vecs);
                nsws[st.hash_value] = tmp;
                for (auto id : st.ids) {
                    tmp->insert(id);
                }
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
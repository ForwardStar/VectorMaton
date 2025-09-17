#ifndef VECTORDB_H
#define VECTORDB_H

#include "headers.h"
#include "nsw.h"
#include "sa.h"
#include "set_hash.h"

class VectorDB {
    public:
        std::vector<std::vector<float>> vecs;
        std::vector<std::string> strs;
        GeneralizedSuffixAutomaton gsa;
        std::unordered_map<std::string, NSW*> nsws;

        int insert(const std::vector<float>& vec, const std::string &s);
        void build(); // Build NSWs for all states in GSA
        void remove(int id);
        std::vector<int> query(const std::vector<float>& vec, const std::string &s, int k);

        VectorDB() {}
        ~VectorDB() {
            for (auto& pair : nsws) {
                delete pair.second;
            }
        }
};

#endif
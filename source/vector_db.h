#ifndef VECTORDB
#define VECTORDB

#include "headers.h"
#include "nsw.h"
#include "sa.h"

class VectorDB {
    private:
        std::vector<std::vector<float>> vecs;
        std::vector<std::string> strs;
        std::vector<NSW> nsws;
        GeneralizedSuffixAutomaton sa;

    public:
        int insert(const std::vector<float>& vec, const std::string &s);
        void remove(int id);
        std::vector<int> query(const std::vector<float>& vec, const std::string &s, int k);

        VectorDB() {};
        ~VectorDB() {};
};

#endif
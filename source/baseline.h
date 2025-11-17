#ifndef BASELINE_H
#define BASELINE_H

#include "headers.h"
#include "nsw.h"

class Baseline {
    public:
        std::vector<std::vector<float>> vecs;
        std::vector<float*> data_ptrs;
        std::vector<std::string> strs;
        #if USE_HNSW
            hnswlib::HierarchicalNSW<float>* hnsw = nullptr;
        #else
            NSW* nsw = nullptr;
        #endif

        int insert(const std::vector<float>& vec, const std::string &s);
        void remove(int id);
        std::vector<int> query(const std::vector<float>& vec, const std::string &s, int k, int threshold=2048);
        
        Baseline() {};
        ~Baseline() {
            #if USE_HNSW
                if (hnsw) delete hnsw;
            #else
                if (nsw) delete nsw;
            #endif
            for (auto ptr : data_ptrs) {
                delete [] ptr;
            }
        };
};

#endif
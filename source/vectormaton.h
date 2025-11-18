#ifndef VECTORMATON_H
#define VECTORMATON_H

#include "headers.h"
#include "nsw.h"
#include "sa.h"
#include "set_hash.h"
#define ENABLE_DEEP_COPY_HNSW 0

class VectorMaton {
    private:
        float** vecs; // It is user's responsibility to manage the memory of vecs
        std::string* strs; // It is user's responsibility to manage the memory of strs
        int dim = 0, num_elements = 0;

    public:
        GeneralizedSuffixAutomaton gsa;
        #if USE_HNSW
            hnswlib::L2Space* space = nullptr;
            std::unordered_map<std::string, hnswlib::HierarchicalNSW<float>*> hnsws;
            hnswlib::HierarchicalNSW<float>* deepCopyHNSW(const hnswlib::HierarchicalNSW<float>& orig, int num_elements);
        #else
            std::unordered_map<std::string, NSW*> nsws;
        #endif

        void set_vectors(float** vectors, int dimension, int num_elems);
        void set_strings(std::string* strings);
        void build();
        size_t size();
        std::vector<int> query(const float* vec, const std::string &s, int k);

        VectorMaton() {}
        ~VectorMaton() {
            #if USE_HNSW
                for (auto& pair : hnsws) {
                    delete pair.second;
                }
                delete space;
            #else
                for (auto& pair : nsws) {
                    delete pair.second;
                }
            #endif
        }
};

#endif
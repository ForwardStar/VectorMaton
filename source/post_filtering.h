#ifndef POSTFILTERING_H
#define POSTFILTERING_H

#include "headers.h"

class PostFiltering {
    private:
        float* vecs; // It is user's responsibility to manage the memory of vecs
        std::string* strs; // It is user's responsibility to manage the memory of strs
        int dim = 0, num_elements = 0;

    public:
        hnswlib::L2Space* space = nullptr;
        hnswlib::HierarchicalNSW<float>* hnsw = nullptr;

        void set_vectors(float* vectors, int dimension, int num_elems);
        void set_strings(std::string* strings);
        void set_ef(int ef);
        void build();
        void load_index(const char* input_folder);
        void save_index(const char* output_folder);
        size_t size();
        std::vector<int> query(const float* vec, const std::string &s, int k, int ef_search=0);
        
        PostFiltering() {};
        ~PostFiltering() {
            if (hnsw) {
                delete hnsw;
            }
            delete space;
        };
};

#endif
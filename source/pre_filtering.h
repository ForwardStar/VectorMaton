#ifndef PREFILTERING_H
#define PREFILTERING_H

#include "headers.h"
#include "sa.h"

class PreFiltering {
    private:
        float** vecs; // It is user's responsibility to manage the memory of vecs
        std::string* strs; // It is user's responsibility to manage the memory of strs
        int dim = 0, num_elements = 0;
        void build_gsa();
        void clear_gsa();

    public:
        GeneralizedSuffixAutomaton gsa;

        void set_vectors(float** vectors, int dimension, int num_elems);
        void set_strings(std::string* strings);
        void build();
        size_t size();
        std::vector<int> query(const float* vec, const std::string &s, int k, int threshold=2048);
        
        PreFiltering() {};
        ~PreFiltering() {};
};

#endif
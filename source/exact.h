#ifndef EXACT_H
#define EXACT_H

#include "headers.h"

class ExactSearch {
    private:
        float* vecs; // It is user's responsibility to manage the memory of vecs
        std::string* strs; // It is user's responsibility to manage the memory of strs
        int dim = 0, max_elements = 0;
        
    public:
        void set_vectors(float* vectors, int dimension, int max_elems);
        void set_strings(std::string* strings);
        void build();
        std::vector<int> query(const float* vec, const std::string &s, int k);

        ExactSearch() {};
        ~ExactSearch() {};
};

#endif
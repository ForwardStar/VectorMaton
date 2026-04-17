#ifndef EXACT_H
#define EXACT_H

#include "headers.h"

class ExactSearch {
    private:
        std::vector<float> vecs;
        std::vector<std::string> strs;
        int dim = 0, max_elements = 0;
        
    public:
        void set_vectors(const std::vector<float>& vectors, int dimension);
        void set_strings(const std::vector<std::string>& strings);
        std::vector<int> query(const float* vec, const std::string &s, int k);

        ExactSearch() {};
        ~ExactSearch() {};
};

#endif

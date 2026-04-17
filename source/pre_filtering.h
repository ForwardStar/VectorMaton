#ifndef PREFILTERING_H
#define PREFILTERING_H

#include "headers.h"
#include "sa.h"

class PreFiltering {
    private:
        std::vector<float> vecs;
        std::vector<std::string> strs;
        int dim = 0, num_elements = 0;
        void build_gsa();
        void clear_gsa();

    public:
        GeneralizedSuffixAutomaton gsa;

        void set_vectors(const std::vector<float>& vectors, int dimension);
        void set_strings(const std::vector<std::string>& strings);
        void build();
        void insert(const std::vector<float>& vec, const std::string& str);
        size_t size();
        std::vector<int> query(const float* vec, const std::string &s, int k, int threshold=2048);
        
        PreFiltering() {};
        ~PreFiltering() {};
};

#endif

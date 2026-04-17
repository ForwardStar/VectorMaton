#ifndef POSTFILTERING_H
#define POSTFILTERING_H

#include "headers.h"

class PostFiltering {
    private:
        std::vector<float> vecs;
        std::vector<std::string> strs;
        int dim = 0, num_elements = 0;

    public:
        hnswlib::L2Space* space = nullptr;
        hnswlib::HierarchicalNSW<float>* hnsw = nullptr;

        void set_vectors(const std::vector<float>& vectors, int dimension);
        void set_strings(const std::vector<std::string>& strings);
        void set_ef(int ef);
        void build();
        void insert(const std::vector<float>& vec, const std::string& str);
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

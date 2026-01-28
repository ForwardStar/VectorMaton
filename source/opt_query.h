#ifndef OPTQUERY_H
#define OPTQUERY_H

#include "headers.h"

class OptQuery {
    private:
        float* vecs; // It is user's responsibility to manage the memory of vecs
        std::string* strs; // It is user's responsibility to manage the memory of strs
        int dim = 0, num_elements = 0;

    public:
        hnswlib::L2Space* space = nullptr;
        std::unordered_map<std::string, hnswlib::HierarchicalNSW<float>*> hnsw;
        std::unordered_map<std::string, std::unordered_set<int>> str_to_ids;

        void set_vectors(float* vectors, int dimension, int num_elems);
        void set_strings(std::string* strings);
        void set_ef(int ef);
        void build();
        void load_index(const char* input_folder);
        void save_index(const char* output_folder);
        size_t size();
        std::vector<int> query(const float* vec, const std::string &s, int k, int ef_search=0);
        
        OptQuery() {};
        ~OptQuery() {
            for (auto &pair : hnsw) {
                delete pair.second;
            }
            delete space;
        };
};

#endif
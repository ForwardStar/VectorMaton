#ifndef VECTORMATON_H
#define VECTORMATON_H

#include "headers.h"
#include "sa.h"

class VectorMaton {
    private:
        float* vecs; // It is user's responsibility to manage the memory of vecs
        std::string* strs; // It is user's responsibility to manage the memory of strs
        int dim = 0, num_elements = 0;
        int min_build_threshold = 200; // minimum number of vectors to build HNSW/NSW
        void build_gsa();
        void clear_gsa();

    public:
        int* inherit_states = nullptr; // inherited state id
        int* size_ids = nullptr; // how many vectors are maintained in this state?
        int** candidate_ids = nullptr; // maintained vector ids in this state (others are inherited from inherit_states)
        GeneralizedSuffixAutomaton gsa;
        hnswlib::L2Space* space = nullptr;
        hnswlib::HierarchicalNSW<float>** hnsws = nullptr;

        void set_vectors(float* vectors, int dimension, int num_elems);
        void set_strings(std::string* strings);
        void build_smart();
        void build_full();
        void load_index(const char* input_folder);
        void save_index(const char* output_folder);
        size_t size();
        size_t vertex_num();
        void set_ef(int ef);
        void set_min_build_threshold(int threshold);
        std::vector<int> query(const float* vec, const std::string &s, int k);

        VectorMaton() {}
        ~VectorMaton();
};

#endif
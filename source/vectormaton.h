#ifndef VECTORMATON_H
#define VECTORMATON_H

#include "headers.h"
#include "sa.h"
#include "mpmc_queue.h"

class VectorMaton {
    private:
        std::vector<float> vecs;
        std::vector<std::string> strs;
        int dim = 0, num_elements = 0;
        int min_build_threshold = 200; // minimum number of vectors to build HNSW/NSW
        void build_gsa();
        void clear_gsa();

    public:
        std::vector<int> inherit_states = {}; // inherited state id
        std::vector<std::vector<int>> candidate_ids = {}; // maintained vector ids in this state (others are inherited from inherit_states)
        GeneralizedSuffixAutomaton gsa;
        hnswlib::L2Space* space = nullptr;
        std::vector<hnswlib::HierarchicalNSW<float>*> hnsws;

        void set_vectors(const std::vector<float>& vectors, int dimension);
        void set_strings(const std::vector<std::string>& strings);
        void build_parallel(int cores=8);
        void build_smart();
        void build_full();
        void insert(const std::vector<float>& vec, const std::string& str);
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

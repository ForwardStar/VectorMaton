#ifndef BM25_FILTERING_H
#define BM25_FILTERING_H

#include "headers.h"

class BM25Filtering {
    private:
        std::vector<float> vecs;
        std::vector<std::string> strs;
        int dim = 0;
        int num_elements = 0;

        std::unordered_map<std::string, int> term_to_id;
        std::vector<std::string> id_to_term;
        std::vector<std::vector<std::pair<uint32_t, int>>> postings;
        std::vector<int> doc_lengths;
        std::vector<double> idf;
        double avg_doc_len = 0.0;

        hnswlib::L2Space* space = nullptr;
        std::vector<hnswlib::HierarchicalNSW<float>*> hnsws;

        static std::vector<std::string> tokenize(const std::string& s);
        static int get_tf(const std::vector<std::pair<uint32_t, int>>& posting, int doc_id);
        double compute_idf(int df) const;
        void build_hnsw();

    public:
        long long peak_memory_usage = 0;

        void set_vectors(const std::vector<float>& vectors, int dimension);
        void set_strings(const std::vector<std::string>& strings);
        void set_ef(int ef);
        void build();
        void insert(const std::vector<float>& vec, const std::string& str);
        void load_index(const char* input_folder);
        void save_index(const char* output_folder);
        size_t size();
        std::vector<int> query(const float* vec, const std::string &s, int k, int ef_search=0);

        BM25Filtering() {};
        ~BM25Filtering();
};

#endif

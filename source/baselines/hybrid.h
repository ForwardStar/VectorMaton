#ifndef HYBRID_H
#define HYBRID_H

#include "headers.h"
#include "pre_filtering.h"
#include "post_filtering.h"

class Hybrid {
    private:
        PreFiltering pre_filtering;
        PostFiltering post_filtering;
        int num_elements = 0;

        void update_peak_memory_usage();
        double selectivity(const std::string& s) const;

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
        std::vector<int> query(const float* vec, const std::string& s, int k, int ef_search=0);

        Hybrid() {};
        ~Hybrid() {};
};

#endif

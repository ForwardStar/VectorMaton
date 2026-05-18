#include "hybrid.h"

void Hybrid::update_peak_memory_usage() {
    peak_memory_usage = std::max(peak_memory_usage, pre_filtering.peak_memory_usage);
    peak_memory_usage = std::max(peak_memory_usage, post_filtering.peak_memory_usage);
}

double Hybrid::selectivity(const std::string& s) const {
    if (num_elements <= 0) {
        return 0.0;
    }
    int state = pre_filtering.gsa.query(s);
    if (state == -1) {
        return 0.0;
    }
    return static_cast<double>(pre_filtering.gsa.st[state].ids.size()) / num_elements;
}

void Hybrid::set_vectors(const std::vector<float>& vectors, int dimension) {
    pre_filtering.set_vectors(vectors, dimension);
    post_filtering.set_vectors(vectors, dimension);
    num_elements = dimension == 0 ? 0 : static_cast<int>(vectors.size()) / dimension;
    update_peak_memory_usage();
}

void Hybrid::set_strings(const std::vector<std::string>& strings) {
    pre_filtering.set_strings(strings);
    post_filtering.set_strings(strings);
    update_peak_memory_usage();
}

void Hybrid::set_ef(int ef) {
    post_filtering.set_ef(ef);
}

void Hybrid::build() {
    pre_filtering.build();
    update_peak_memory_usage();
    post_filtering.build();
    update_peak_memory_usage();
}

void Hybrid::insert(const std::vector<float>& vec, const std::string& str) {
    pre_filtering.insert(vec, str);
    post_filtering.insert(vec, str);
    num_elements++;
    update_peak_memory_usage();
}

void Hybrid::load_index(const char* input_folder) {
    pre_filtering.build();
    update_peak_memory_usage();
    post_filtering.load_index(input_folder);
    update_peak_memory_usage();
}

void Hybrid::save_index(const char* output_folder) {
    post_filtering.save_index(output_folder);
}

size_t Hybrid::size() {
    return pre_filtering.size() + post_filtering.size();
}

std::vector<int> Hybrid::query(const float* vec, const std::string& s, int k, int ef_search) {
    int effective_ef = ef_search > 0 ? ef_search : k;
    double threshold = effective_ef > 0 ? 10.0 / effective_ef : std::numeric_limits<double>::infinity();
    if (selectivity(s) < threshold) {
        return pre_filtering.query(vec, s, k);
    }
    return post_filtering.query(vec, s, k, ef_search);
}

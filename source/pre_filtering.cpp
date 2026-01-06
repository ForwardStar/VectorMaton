#include "pre_filtering.h"

void PreFiltering::build_gsa() {
    LOG_DEBUG("Building Generalized Suffix Automaton (GSA)");
    unsigned long long start_time = currentTime();
    for (int i = 0; i < num_elements; i++) {
        gsa.add_string(i, strs[i]);
    }
    LOG_DEBUG("GSA built in ", timeFormatting(currentTime() - start_time).str());
    LOG_DEBUG("Total GSA states: ", std::to_string(gsa.size()), ", total string IDs in GSA: ", std::to_string(gsa.size_tot()));

    // Print GSA statistics
    auto stats = gsa.get_statistics();
    for (size_t depth = 0; depth < stats.size(); ++depth) {
        const auto &stat = stats[depth];
        LOG_DEBUG("Depth ", depth, ": num states = ", stat.sizes.size(),
                  ", median vector set size = ", stat.mid,
                  ", average vector set size = ", stat.avg);
    }
}

void PreFiltering::set_vectors(float* vectors, int dimension, int num_elems) {
    vecs = vectors;
    dim = dimension;
    num_elements = num_elems;
}

void PreFiltering::set_strings(std::string* strings) {
    strs = strings;
}

void PreFiltering::build() {
    build_gsa();
}

size_t PreFiltering::size() {
    size_t total_size = 0;
    for (int i = 0; i < num_elements; i++) {
        total_size += sizeof(std::string) + strs[i].capacity(); // size of each string
        total_size += sizeof(float) * dim; // size of each vector
        total_size += sizeof(uint32_t) * gsa.st[i].ids.capacity(); // size of each GSA state's id vector
    }
    return total_size;
}

std::vector<int> PreFiltering::query(const float* vec, const std::string &s, int k, int threshold) {
    int i = gsa.query(s);
    if (i == -1) return {};
    std::vector<int> results;
    for (const auto& id : gsa.st[i].ids) {
        results.push_back(id);
    }
    std::sort(results.begin(), results.end(), [&](int a, int b) {
        return distance(vecs + a * dim, vec, dim) < distance(vecs + b * dim, vec, dim);
    });
    results.resize(std::min(k, static_cast<int>(results.size())));
    return results;
}
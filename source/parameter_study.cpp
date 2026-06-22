#include "headers.h"
#include "exact.h"
#include "post_filtering.h"

namespace {

const std::vector<int> EF_SEARCH = {8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
const std::vector<float> SEARCH_K_RATIOS = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

std::vector<std::string> load_strings(const std::string& path) {
    std::vector<std::string> strings;
    std::ifstream input(path);
    std::string value;
    while (input >> value) {
        strings.emplace_back(value);
    }
    return strings;
}

std::vector<std::vector<float>> load_vectors(const std::string& path) {
    std::vector<std::vector<float>> vectors;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream iss(line);
        std::vector<float> vec;
        float value;
        while (iss >> value) {
            vec.push_back(value);
        }
        vectors.emplace_back(std::move(vec));
    }
    return vectors;
}

std::vector<int> load_k(const std::string& path) {
    std::vector<int> ks;
    std::ifstream input(path);
    int k;
    while (input >> k) {
        ks.emplace_back(k);
    }
    return ks;
}

std::vector<float> flatten_vectors(const std::vector<std::vector<float>>& vectors, int dim) {
    std::vector<float> flattened(vectors.size() * dim);
    for (size_t i = 0; i < vectors.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            flattened[i * dim + j] = vectors[i][j];
        }
    }
    return flattened;
}

bool validate_vectors(const std::vector<std::vector<float>>& vectors, int dim, const std::string& label) {
    for (size_t i = 0; i < vectors.size(); ++i) {
        if (static_cast<int>(vectors[i].size()) != dim) {
            LOG_ERROR(label, " vector dimension mismatch at index ", i, ": expected ", dim, ", got ", vectors[i].size());
            return false;
        }
    }
    return true;
}

void align_data(std::vector<std::string>& strings, std::vector<std::vector<float>>& vectors) {
    if (strings.size() == vectors.size()) {
        return;
    }
    LOG_WARN("Mismatched strings/vectors: aligning to smaller size");
    size_t n = std::min(strings.size(), vectors.size());
    strings.resize(n);
    vectors.resize(n);
}

void align_queries(std::vector<std::string>& strings, std::vector<std::vector<float>>& vectors, std::vector<int>& ks) {
    if (strings.size() == vectors.size() && strings.size() == ks.size()) {
        return;
    }
    LOG_WARN("Mismatched query strings/vectors/ks: aligning to smaller size");
    size_t n = std::min({strings.size(), vectors.size(), ks.size()});
    strings.resize(n);
    vectors.resize(n);
    ks.resize(n);
}

float compute_recall(
    const std::vector<std::vector<int>>& exact_results,
    const std::vector<std::vector<int>>& all_results
) {
    double total_recall = 0.0;
    int effective = 0;
    for (size_t i = 0; i < exact_results.size(); ++i) {
        if (exact_results[i].empty()) {
            continue;
        }

        std::unordered_set<int> exact_set(exact_results[i].begin(), exact_results[i].end());
        int correct = 0;
        for (int id : all_results[i]) {
            if (exact_set.find(id) != exact_set.end()) {
                correct++;
            }
        }
        effective++;
        total_recall += static_cast<double>(correct) / exact_results[i].size();
    }

    return effective ? static_cast<float>(total_recall / effective) : 0.0f;
}

bool run_postfiltering_study(
    const std::string& string_data_file,
    const std::string& vector_data_file,
    const std::string& string_query_file,
    const std::string& vector_query_file,
    const std::string& k_query_file,
    const std::string& statistics_file
) {
    auto strings = load_strings(string_data_file);
    auto vectors = load_vectors(vector_data_file);
    align_data(strings, vectors);
    if (vectors.empty()) {
        LOG_ERROR("No vectors loaded from ", vector_data_file);
        return false;
    }

    int dim = static_cast<int>(vectors[0].size());
    if (!validate_vectors(vectors, dim, "data")) {
        return false;
    }
    auto base_vectors = flatten_vectors(vectors, dim);
    vectors.clear();

    auto queried_strings = load_strings(string_query_file);
    auto queried_vectors = load_vectors(vector_query_file);
    auto queried_k = load_k(k_query_file);
    align_queries(queried_strings, queried_vectors, queried_k);
    if (!validate_vectors(queried_vectors, dim, "query")) {
        return false;
    }

    LOG_INFO("Data=", strings.size(), ", queries=", queried_strings.size(), ", dim=", dim);

    ExactSearch es;
    es.set_vectors(base_vectors, dim);
    es.set_strings(strings);
    unsigned long long start_time = currentTime();
    std::vector<std::vector<int>> exact_results;
    exact_results.reserve(queried_strings.size());
    double total_selectivity = 0.0;
    for (size_t i = 0; i < queried_strings.size(); ++i) {
        size_t match_count = 0;
        exact_results.emplace_back(es.query(queried_vectors[i].data(), queried_strings[i], queried_k[i], &match_count));
        if (strings.size()) {
            total_selectivity += static_cast<double>(match_count) / strings.size();
        }
    }
    auto exact_time = currentTime() - start_time;
    const double average_selectivity = queried_strings.empty() ? 0.0 : total_selectivity / queried_strings.size();
    float exact_time_per_query = queried_strings.empty() ? 0.0f : static_cast<float>(exact_time) / queried_strings.size();
    LOG_INFO("ExactSearch avg (us): ", exact_time_per_query);
    LOG_INFO("Average selectivity: ", average_selectivity);

    PostFiltering pf;
    pf.set_vectors(base_vectors, dim);
    pf.set_strings(strings);
    LOG_INFO("Building PostFiltering index");
    start_time = currentTime();
    pf.build();
    LOG_INFO("PostFiltering index built took ", timeFormatting(currentTime() - start_time).str());

    std::filesystem::path output_file = statistics_file.empty() ? "parameter_study.csv" : statistics_file;
    if (output_file.has_parent_path()) {
        std::filesystem::create_directories(output_file.parent_path());
    }
    std::ofstream stats(output_file);
    stats << "ef_search,search_k_ratio,search_k,time_us,recall,exact,average_selectivity\n";

    for (int ef : EF_SEARCH) {
        for (float ratio : SEARCH_K_RATIOS) {
            int search_k = std::max(1, static_cast<int>(std::round(ratio * ef)));
            start_time = currentTime();
            std::vector<std::vector<int>> all_results;
            all_results.reserve(queried_strings.size());
            for (size_t i = 0; i < queried_strings.size(); ++i) {
                all_results.emplace_back(
                    pf.query(queried_vectors[i].data(), queried_strings[i], queried_k[i], ef, search_k)
                );
            }
            float time_us = queried_strings.empty() ? 0.0f : static_cast<float>(currentTime() - start_time) / queried_strings.size();
            float recall = compute_recall(exact_results, all_results);

            stats << ef << "," << ratio << "," << search_k << "," << time_us << "," << recall << "," << exact_time_per_query << "," << average_selectivity << "\n";
            LOG_INFO("ef_search=", ef, ", search_k_ratio=", ratio, ", search_k=", search_k, ", time=", timeFormatting(time_us).str(), ", recall=", recall);
        }
    }

    LOG_INFO("Wrote ", output_file.string());
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 7 || std::strcmp(argv[6], "PostFiltering") != 0) {
        LOG_ERROR("Usage: ./parameter_study <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k_query_file> PostFiltering [--statistics-file=output.csv]");
        return 1;
    }

    std::string statistics_file = "";
    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--statistics-file=") == 0) {
            statistics_file = arg.substr(18);
        }
        else {
            LOG_ERROR("Unknown argument: ", arg);
            return 1;
        }
    }

    return run_postfiltering_study(argv[1], argv[2], argv[3], argv[4], argv[5], statistics_file) ? 0 : 1;
}

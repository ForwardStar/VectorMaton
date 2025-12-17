#include "headers.h"
#include "exact.h"
#include "baseline.h"
#include "vectormaton.h"

int main(int argc, char * argv[]) {
    if (argc < 8) {
        LOG_ERROR("Usage: ./main <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k_query_file> <output_file> <Exact|Baseline|VectorMaton>");
        return 1;
    }

    int data_size = 1000000000; // default: no limit
    // Parse optional arguments
    if (argc > 8) {
        for (int i = 0; i < argc; i++) {
            if (std::string(argv[i]) == "--debug") {
                Logger::instance().set_level(Logger::Level::DEBUG);
                LOG_DEBUG("Debug mode enabled");
                for (int j = i; j < argc - 1; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
                break;
            }
        }
        for (int i = 0; i < argc; i++) {
            if (std::string(argv[i]).find("--data-size=") == 0) {
                data_size = std::stoi(std::string(argv[i]).substr(12));
                LOG_DEBUG("Data size limit set to ", data_size);
                for (int j = i; j < argc - 1; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
                break;
            }
        }
    }

    // Read strings
    LOG_DEBUG("String data file: ", argv[1]);
    std::vector<std::string> strings;
    std::ifstream f_strings(argv[1]);
    std::string s;
    while (f_strings >> s) {
        strings.emplace_back(s);
        if (strings.size() >= data_size) {
            break;
        }
    }

    // Read vectors
    LOG_DEBUG("Vector data file: ", argv[2]);
    std::vector<std::vector<float>> vectors;
    std::ifstream f_vectors(argv[2]);
    std::string line;
    while (std::getline(f_vectors, line)) {
        std::istringstream iss(line);
        std::vector<float> vec;
        float value;
        while (iss >> value) {
            vec.push_back(value);
        }
        vectors.push_back(vec);
        if (vectors.size() >= data_size) {
            break;
        }
    }

    // Log the number of strings and vectors
    LOG_INFO("Number of strings: ", strings.size());
    LOG_INFO("Total length of strings: ", std::accumulate(strings.begin(), strings.end(), 0, [](int sum, const std::string& str) { return sum + str.size(); }));
    LOG_INFO("Number of vectors: ", vectors.size());
    if (strings.size() != vectors.size()) {
        LOG_WARN("Mismatched number of strings and vectors: aligning their sizes");
        size_t min_size = std::min(strings.size(), vectors.size());
        strings.resize(min_size);
        vectors.resize(min_size);
        LOG_DEBUG("Number of strings: ", strings.size());
        LOG_DEBUG("Number of vectors: ", vectors.size());
    }

    // Log vector dimensions
    LOG_DEBUG("Vector dimension: ", vectors[0].size());
    for (int i = 1; i < vectors.size(); i++) {
        if (vectors[i].size() != vectors[0].size()) {
            LOG_ERROR("Inconsistent vector dimensions at index ", i, "expected ", vectors[0].size(), "got ", vectors[i].size());
            return 1;
        }
    }
    
    // Read queries
    LOG_DEBUG("Query data file (string): ", argv[3]);
    std::vector<std::string> queried_strings;
    std::ifstream f_queried_strings(argv[3]);
    while (f_queried_strings >> s) {
        queried_strings.emplace_back(s);
    }
    LOG_DEBUG("Query data file (vector): ", argv[4]);
    std::vector<std::vector<float>> queried_vectors;
    std::ifstream f_queried_vectors(argv[4]);
    while (std::getline(f_queried_vectors, line)) {
        std::istringstream iss(line);
        std::vector<float> vec;
        float value;
        while (iss >> value) {
            vec.push_back(value);
        }
        queried_vectors.push_back(vec);
    }
    LOG_DEBUG("Query data file (k): ", argv[5]);
    std::vector<int> queried_k;
    std::ifstream f_queried_k(argv[5]);
    int k;
    while (f_queried_k >> k) {
        queried_k.emplace_back(k);
    }
    LOG_DEBUG("Number of query strings: ", queried_strings.size());
    LOG_DEBUG("Number of query vectors: ", queried_vectors.size());
    LOG_DEBUG("Number of query ks: ", queried_k.size());
    if (queried_strings.size() != queried_vectors.size() || queried_strings.size() != queried_k.size()) {
        LOG_WARN("Mismatched number of query strings, vectors, and ks: aligning their sizes");
        size_t min_size = std::min({queried_strings.size(), queried_vectors.size(), queried_k.size()});
        queried_strings.resize(min_size);
        queried_vectors.resize(min_size);
        queried_k.resize(min_size);
        LOG_DEBUG("Number of query strings: ", queried_strings.size());
        LOG_DEBUG("Number of query vectors: ", queried_vectors.size());
        LOG_DEBUG("Number of query ks: ", queried_k.size());
    }

    for (int i = 0; i < queried_vectors.size(); i++) {
        if (queried_vectors[i].size() != vectors[0].size()) {
            LOG_ERROR("Inconsistent query vector dimensions at index ", i, ": expected ", vectors[0].size(), ", got ", queried_vectors[i].size());
            return 1;
        }
    }

    // Turn vectors into float**
    float** vec_array = new float*[vectors.size()];
    for (size_t i = 0; i < vectors.size(); ++i) {
        vec_array[i] = vectors[i].data();
    }

    // Turn strings into std::string*
    std::string* str_array = new std::string[strings.size()];
    for (size_t i = 0; i < strings.size(); ++i) {
        str_array[i] = strings[i];
    }

    if (std::strcmp(argv[argc - 1], "Exact") == 0) {
        LOG_INFO("Using Exact search");
        ExactSearch es;
        es.set_vectors(vec_array, vectors[0].size(), vectors.size());
        es.set_strings(str_array);
        LOG_DEBUG("Processing queries");
        unsigned long long start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = es.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
            all_results.emplace_back(res);
        }
        LOG_INFO("ExactSearch query processing took ", timeFormatting(currentTime() - start_time).str());
        LOG_DEBUG("Writing results to ", argv[6]);
        std::ofstream f_results(argv[6]);
        for (const auto& res : all_results) {
            for (const auto& id : res) {
                f_results << id << " ";
            }
            f_results << "\n";
        }
    }

    if (std::strcmp(argv[argc - 1], "Baseline") == 0) {
        LOG_INFO("Using Baseline search");
        Baseline bs;
        bs.set_vectors(vec_array, vectors[0].size(), vectors.size());
        bs.set_strings(str_array);
        LOG_DEBUG("Building Baseline index");
        unsigned long long start_time = currentTime();
        bs.build();
        LOG_INFO("Baseline index built in ", timeFormatting(currentTime() - start_time).str());
        LOG_INFO("Total index size: ", bs.size(), " bytes");
        LOG_DEBUG("Processing queries");
        start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = bs.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
            all_results.emplace_back(res);
        }
        LOG_INFO("Baseline query processing took ", timeFormatting(currentTime() - start_time).str());
        LOG_DEBUG("Writing results to ", argv[6]);
        std::ofstream f_results(argv[6]);
        for (const auto& res : all_results) {
            for (const auto& id : res) {
                f_results << id << " ";
            }
            f_results << "\n";
        }
    }

    if (std::strcmp(argv[argc - 1], "VectorMaton-full") == 0) {
        LOG_INFO("Using VectorMaton-full");
        VectorMaton vdb;
        vdb.set_vectors(vec_array, vectors[0].size(), vectors.size());
        vdb.set_strings(str_array);
        LOG_DEBUG("Building VectorMaton-full index");
        unsigned long long start_time = currentTime();
        vdb.build_full();
        LOG_INFO("VectorMaton-full index built took ", timeFormatting(currentTime() - start_time).str());
        LOG_INFO("Total index size: ", vdb.size(), " bytes");
        LOG_INFO("Total states: ", std::to_string(vdb.gsa.size()), ", total string IDs in GSA: ", std::to_string(vdb.gsa.size_tot()));
        LOG_INFO("Total vertices in HNSW/NSW: ", std::to_string(vdb.vertex_num()));
        LOG_DEBUG("Processing queries");
        start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = vdb.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
            all_results.emplace_back(res);
        }
        LOG_INFO("VectorMaton-full query processing took ", timeFormatting(currentTime() - start_time).str());
        LOG_DEBUG("Writing results to ", argv[6]);
        std::ofstream f_results(argv[6]);
        for (const auto& res : all_results) {
            for (const auto& id : res) {
                f_results << id << " ";
            }
            f_results << "\n";
        }
    }

    if (std::strcmp(argv[argc - 1], "VectorMaton-smart") == 0) {
        LOG_INFO("Using VectorMaton-smart");
        VectorMaton vdb;
        vdb.set_vectors(vec_array, vectors[0].size(), vectors.size());
        vdb.set_strings(str_array);
        LOG_DEBUG("Building VectorMaton-smart index");
        unsigned long long start_time = currentTime();
        vdb.build_smart();
        LOG_INFO("VectorMaton-smart index built took ", timeFormatting(currentTime() - start_time).str());
        LOG_INFO("Total index size: ", vdb.size(), " bytes");
        LOG_INFO("Total states: ", std::to_string(vdb.gsa.size()), ", total string IDs in GSA: ", std::to_string(vdb.gsa.size_tot()));
        LOG_INFO("Total vertices in HNSW/NSW: ", std::to_string(vdb.vertex_num()));
        LOG_DEBUG("Processing queries");
        start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = vdb.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
            all_results.emplace_back(res);
        }
        LOG_INFO("VectorMaton-smart query processing took ", timeFormatting(currentTime() - start_time).str());
        LOG_DEBUG("Writing results to ", argv[6]);
        std::ofstream f_results(argv[6]);
        for (const auto& res : all_results) {
            for (const auto& id : res) {
                f_results << id << " ";
            }
            f_results << "\n";
        }
    }

    return 0;
}
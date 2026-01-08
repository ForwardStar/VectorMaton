#include "headers.h"
#include "exact.h"
#include "pre_filtering.h"
#include "post_filtering.h"
#include "vectormaton.h"

int main(int argc, char * argv[]) {
    if (argc < 7) {
        LOG_ERROR("Usage: ./main <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k_query_file> <PreFiltering/PostFiltering/VectorMaton-full/VectorMaton-smart> [--debug] [--data-size=N] [--statistics_file=output_statistics.csv] [--load-index=index_files_folder] [--save-index=index_files_folder]");
        return 1;
    }

    int data_size = 1000000000; // default: no limit
    std::string statistics_file = "";
    std::string index_in = "";
    std::string index_out = "";
    // Parse optional arguments
    if (argc > 7) {
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
                LOG_INFO("Data size limit set to ", data_size);
                for (int j = i; j < argc - 1; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
                break;
            }
        }
        for (int i = 0; i < argc; i++) {
            if (std::string(argv[i]).find("--statistics_file=") == 0) {
                statistics_file = std::string(argv[i]).substr(18);
                LOG_INFO("Statistics file set to ", statistics_file);
                for (int j = i; j < argc - 1; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
                break;
            }
        }
        for (int i = 0; i < argc; i++) {
            if (std::string(argv[i]).find("--load-index=") == 0) {
                index_in = std::string(argv[i]).substr(13);
                LOG_INFO("Index files folder (load) set to ", index_in);
                for (int j = i; j < argc - 1; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
                break;
            }
        }
        for (int i = 0; i < argc; i++) {
            if (std::string(argv[i]).find("--save-index=") == 0) {
                index_out = std::string(argv[i]).substr(13);
                LOG_INFO("Index files folder (save) set to ", index_out);
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
    LOG_INFO("Number of query strings: ", queried_strings.size());
    LOG_INFO("Number of query vectors: ", queried_vectors.size());
    LOG_INFO("Number of query ks: ", queried_k.size());
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

    // Turn vectors into float*
    int n = vectors.size(), dim = vectors[0].size();
    float* vec_array = new float[n * dim];
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            vec_array[i * dim + j] = vectors[i][j];
        }
    }
    vectors = std::vector<std::vector<float>>();

    // Turn strings into std::string*
    std::string* str_array = new std::string[n];
    for (size_t i = 0; i < n; ++i) {
        str_array[i] = strings[i];
    }
    strings = std::vector<std::string>();

    std::vector<std::vector<int>> exact_results;
    LOG_INFO("Doing ExactSearch for baseline comparison");
    ExactSearch es;
    es.set_vectors(vec_array, dim, n);
    es.set_strings(str_array);
    unsigned long long start_time = currentTime();
    std::vector<std::vector<int>> all_results;
    for (size_t i = 0; i < queried_strings.size(); ++i) {
        auto res = es.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
        exact_results.emplace_back(res);
    }
    auto exact_time = currentTime() - start_time;
    LOG_INFO("ExactSearch query processing took ", timeFormatting(exact_time).str(), ", avg (us): ", (static_cast<float>(exact_time) / queried_strings.size()));

    if (std::strcmp(argv[argc - 1], "PreFiltering") == 0) {
        LOG_INFO("Using PreFiltering");
        PreFiltering pf;
        pf.set_vectors(vec_array, dim, n);
        pf.set_strings(str_array);
        LOG_INFO("Building PreFiltering index");
        unsigned long long start_time = currentTime();
        pf.build();
        LOG_INFO("PreFiltering index built took ", timeFormatting(currentTime() - start_time).str());
        LOG_INFO("Total index size: ", pf.size(), " bytes");
        LOG_INFO("Processing queries");
        start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = pf.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
            all_results.emplace_back(res);
        }
        LOG_INFO("PreFiltering query processing took ", timeFormatting(currentTime() - start_time).str(), ", avg (us): ", (static_cast<float>(currentTime() - start_time) / queried_strings.size()));
        // Compute recall
        double total_recall = 0;
        int effective = 0;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            std::unordered_set<int> exact_set(exact_results[i].begin(), exact_results[i].end());
            int correct = 0;
            for (const auto& id : all_results[i]) {
                if (exact_set.find(id) != exact_set.end()) {
                    correct++;
                }
            }
            if (exact_results[i].size() != 0) effective++, total_recall += (double)correct / exact_results[i].size();
        }
        double recall = static_cast<double>(total_recall) / effective;
        LOG_INFO("PreFiltering recall: ", recall);
    }

    if (std::strcmp(argv[argc - 1], "PostFiltering") == 0) {
        LOG_INFO("Using PostFiltering");
        PostFiltering pf;
        pf.set_vectors(vec_array, dim, n);
        pf.set_strings(str_array);
        LOG_INFO("Building PostFiltering index");
        unsigned long long start_time = currentTime();
        pf.build();
        LOG_INFO("PostFiltering index built took ", timeFormatting(currentTime() - start_time).str());
        LOG_INFO("Total index size: ", pf.size(), " bytes");
        LOG_INFO("Processing queries");
        // TODO: implement
    }

    if (std::strcmp(argv[argc - 1], "VectorMaton-full") == 0) {
        LOG_INFO("Using VectorMaton-full");
        VectorMaton vdb;
        vdb.set_vectors(vec_array, dim, n);
        vdb.set_strings(str_array);
        if (index_in == "") {
            LOG_INFO("Building VectorMaton-full index");
            unsigned long long start_time = currentTime();
            vdb.build_full();
            LOG_INFO("VectorMaton-full index built took ", timeFormatting(currentTime() - start_time).str());
        }
        else {
            LOG_INFO("Loading index from: ", index_in);
            unsigned long long start_time = currentTime();
            vdb.load_index(index_in.c_str());
            LOG_INFO("VectorMaton-full index loaded in ", timeFormatting(currentTime() - start_time).str());
        }
        LOG_INFO("Total index size: ", vdb.size(), " bytes");
        LOG_DEBUG("Total vertices in HNSW: ", std::to_string(vdb.vertex_num()));
        if (index_out != "") {
            LOG_INFO("Saving index to: ", index_out);
            unsigned long long start_time = currentTime();
            vdb.save_index(index_out.c_str());
            LOG_INFO("VectorMaton-full index saved in ", timeFormatting(currentTime() - start_time).str());
        }
        LOG_INFO("Processing queries");
        std::vector<std::map<std::string, float>> statistics;
        for (int ef = 20; ef <= 200; ef += 20) {
            LOG_DEBUG("Set ef_search to ", ef);
            vdb.set_ef(ef);
            start_time = currentTime();
            std::vector<std::vector<int>> all_results;
            for (size_t i = 0; i < queried_strings.size(); ++i) {
                auto res = vdb.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
                all_results.emplace_back(res);
            }
            statistics.emplace_back();
            statistics.back()["ef_search"] = ef;
            statistics.back()["time_us"] = static_cast<float>(currentTime() - start_time) / queried_strings.size();
            // Calculate recall
            double total_recall = 0;
            int effective = 0;
            for (size_t i = 0; i < queried_strings.size(); ++i) {
                std::unordered_set<int> exact_set(exact_results[i].begin(), exact_results[i].end());
                int correct = 0;
                for (const auto& id : all_results[i]) {
                    if (exact_set.find(id) != exact_set.end()) {
                        correct++;
                    }
                }
                if (exact_results[i].size() != 0) effective++, total_recall += (double)correct / exact_results[i].size();
            }
            statistics.back()["recall"] = static_cast<float>(total_recall) / effective;
            LOG_INFO("ef_search=", ef, ", time=", timeFormatting(statistics.back()["time_us"]).str(), ", recall=", statistics.back()["recall"]);
        }
        if (statistics_file != "") {
            LOG_INFO("Writing statistics to ", statistics_file);
            std::ofstream f_stats(statistics_file);
            // Write header
            f_stats << "ef_search,time_us,recall,exact\n";
            // Compute exact search time per query
            float exact_time_per_query = static_cast<float>(exact_time) / queried_strings.size();
            // Write data
            for (const auto& stat : statistics) {
                f_stats << stat.at("ef_search") << "," << stat.at("time_us") << "," << stat.at("recall") << "," << exact_time_per_query << "\n";
            }
        }
    }

    if (std::strcmp(argv[argc - 1], "VectorMaton-smart") == 0) {
        LOG_INFO("Using VectorMaton-smart");
        VectorMaton vdb;
        vdb.set_vectors(vec_array, dim, n);
        vdb.set_strings(str_array);
        if (index_in == "") {
            LOG_INFO("Building VectorMaton-smart index");
            unsigned long long start_time = currentTime();
            vdb.build_smart();
            LOG_INFO("VectorMaton-smart index built took ", timeFormatting(currentTime() - start_time).str());
        }
        else {
            LOG_INFO("Loading index from: ", index_in);
            unsigned long long start_time = currentTime();
            vdb.load_index(index_in.c_str());
            LOG_INFO("VectorMaton-smart index loaded in ", timeFormatting(currentTime() - start_time).str());
        }
        LOG_INFO("Total index size: ", vdb.size(), " bytes");
        LOG_DEBUG("Total vertices in HNSW: ", std::to_string(vdb.vertex_num()));
        if (index_out != "") {
            LOG_INFO("Saving index to: ", index_out);
            unsigned long long start_time = currentTime();
            vdb.save_index(index_out.c_str());
            LOG_INFO("VectorMaton-smart index saved in ", timeFormatting(currentTime() - start_time).str());
        }
        LOG_INFO("Processing queries");
        std::vector<std::map<std::string, float>> statistics;
        for (int ef = 20; ef <= 200; ef += 20) {
            LOG_DEBUG("Set ef_search to ", ef);
            vdb.set_ef(ef);
            start_time = currentTime();
            std::vector<std::vector<int>> all_results;
            for (size_t i = 0; i < queried_strings.size(); ++i) {
                auto res = vdb.query(queried_vectors[i].data(), queried_strings[i], queried_k[i]);
                all_results.emplace_back(res);
            }
            statistics.emplace_back();
            statistics.back()["ef_search"] = ef;
            statistics.back()["time_us"] = static_cast<float>(currentTime() - start_time) / queried_strings.size();
            // Calculate recall
            double total_recall = 0;
            int effective = 0;
            for (size_t i = 0; i < queried_strings.size(); ++i) {
                std::unordered_set<int> exact_set(exact_results[i].begin(), exact_results[i].end());
                int correct = 0;
                for (const auto& id : all_results[i]) {
                    if (exact_set.find(id) != exact_set.end()) {
                        correct++;
                    }
                }
                if (exact_results[i].size() != 0) effective++, total_recall += (double)correct / exact_results[i].size();
            }
            statistics.back()["recall"] = static_cast<float>(total_recall) / effective;
            LOG_INFO("ef_search=", ef, ", time=", timeFormatting(statistics.back()["time_us"]).str(), ", recall=", statistics.back()["recall"]);
        }
        if (statistics_file != "") {
            LOG_INFO("Writing statistics to ", statistics_file);
            std::ofstream f_stats(statistics_file);
            // Write header
            f_stats << "ef_search,time_us,recall,exact\n";
            // Compute exact search time per query
            float exact_time_per_query = static_cast<float>(exact_time) / queried_strings.size();
            // Write data
            for (const auto& stat : statistics) {
                f_stats << stat.at("ef_search") << "," << stat.at("time_us") << "," << stat.at("recall") << "," << exact_time_per_query << "\n";
            }
        }
    }

    delete [] vec_array;
    delete [] str_array;

    return 0;
}
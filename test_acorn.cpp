#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <faiss/IndexACORN.h>
#include <faiss/impl/ACORN.h>
#include <faiss/index_io.h>

namespace {

struct Args {
    std::string string_data_file;
    std::string vector_data_file;
    std::string string_query_file;
    std::string vector_query_file;
    std::string k_file;
    std::string ground_truth_file;
    std::string output_file = "acorn_hnsw_stats.csv";
    int M = 32;
    int gamma = 12;
    int M_beta = 32;
};

long long current_memory_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long long value = 0;
            std::string unit;
            iss >> value >> unit;
            return value * 1024;
        }
    }
    return -1;
}

Args parse_args(int argc, char** argv) {
    if (argc < 7) {
        std::cerr
            << "Usage: " << argv[0]
            << " <string_data_file> <vector_data_file> <string_query_file>"
            << " <vector_query_file> <k_file> <ground_truth_file>"
            << " [--M=32] [--gamma=12] [--M-beta=32] [--output=acorn_hnsw_stats.csv]\n";
        std::exit(1);
    }

    Args args;
    args.string_data_file = argv[1];
    args.vector_data_file = argv[2];
    args.string_query_file = argv[3];
    args.vector_query_file = argv[4];
    args.k_file = argv[5];
    args.ground_truth_file = argv[6];

    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_after = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (arg.rfind("--M=", 0) == 0) {
            args.M = std::stoi(value_after("--M="));
        } else if (arg.rfind("--gamma=", 0) == 0) {
            args.gamma = std::stoi(value_after("--gamma="));
        } else if (arg.rfind("--M-beta=", 0) == 0) {
            args.M_beta = std::stoi(value_after("--M-beta="));
        } else if (arg.rfind("--output=", 0) == 0) {
            args.output_file = value_after("--output=");
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }
    return args;
}

std::vector<float> load_vectors(const std::string& path, int& dim) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open vector file: " + path);
    }

    std::vector<float> vectors;
    std::string line;
    dim = 0;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float value = 0.0f;
        while (iss >> value) {
            row.push_back(value);
        }
        if (row.empty()) {
            continue;
        }
        if (dim == 0) {
            dim = static_cast<int>(row.size());
        }
        if (static_cast<int>(row.size()) != dim) {
            throw std::runtime_error("Inconsistent vector dimension in: " + path);
        }
        vectors.insert(vectors.end(), row.begin(), row.end());
    }
    return vectors;
}

std::vector<std::string> load_strings(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open string file: " + path);
    }

    std::vector<std::string> strings;
    std::string line;
    while (std::getline(in, line)) {
        strings.push_back(line);
    }
    return strings;
}

std::vector<int> load_k(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open k file: " + path);
    }

    std::vector<int> ks;
    int value = 0;
    while (in >> value) {
        ks.push_back(value);
    }
    return ks;
}

std::vector<std::vector<int>> load_ground_truth(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open ground truth file: " + path);
    }

    std::vector<std::vector<int>> ground_truth;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<int> ids;
        int id = 0;
        while (iss >> id) {
            ids.push_back(id);
        }
        ground_truth.push_back(std::move(ids));
    }
    return ground_truth;
}

std::vector<char> make_filter_map(
        const std::vector<std::string>& data_strings,
        const std::vector<std::string>& query_strings) {
    const size_t nb = data_strings.size();
    const size_t nq = query_strings.size();
    std::vector<char> filter_map(nq * nb, 0);
    for (size_t qi = 0; qi < nq; ++qi) {
        for (size_t id = 0; id < nb; ++id) {
            filter_map[qi * nb + id] =
                    data_strings[id].find(query_strings[qi]) != std::string::npos;
        }
    }
    return filter_map;
}

size_t serialized_index_size(const faiss::Index& index) {
    std::filesystem::path path =
            std::filesystem::temp_directory_path() / "vectormaton_acorn_index.tmp";
    try {
        faiss::write_index(&index, path.string().c_str());
        size_t size = std::filesystem::file_size(path);
        std::filesystem::remove(path);
        return size;
    } catch (const std::exception& exc) {
        std::cerr << "Could not serialize ACORN index for size measurement: "
                  << exc.what() << "\n";
        std::filesystem::remove(path);
        return 0;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        std::cout << "Loading ACORN data...\n";
        int data_dim = 0;
        int query_dim = 0;
        std::vector<float> data_vectors = load_vectors(args.vector_data_file, data_dim);
        std::vector<std::string> data_strings = load_strings(args.string_data_file);
        std::vector<float> query_vectors = load_vectors(args.vector_query_file, query_dim);
        std::vector<std::string> query_strings = load_strings(args.string_query_file);
        std::vector<int> query_ks = load_k(args.k_file);
        std::vector<std::vector<int>> ground_truth = load_ground_truth(args.ground_truth_file);

        if (data_dim != query_dim) {
            throw std::runtime_error("Data/query vector dimensions differ.");
        }

        size_t nb = data_dim == 0 ? 0 : data_vectors.size() / data_dim;
        size_t nq = query_dim == 0 ? 0 : query_vectors.size() / query_dim;
        nb = std::min(nb, data_strings.size());
        nq = std::min({nq, query_strings.size(), query_ks.size(), ground_truth.size()});
        data_vectors.resize(nb * data_dim);
        data_strings.resize(nb);
        query_vectors.resize(nq * query_dim);
        query_strings.resize(nq);
        query_ks.resize(nq);
        ground_truth.resize(nq);

        std::cout << "Building ACORN index: nb=" << nb << ", dim=" << data_dim
                  << ", M=" << args.M << ", gamma=" << args.gamma
                  << ", M_beta=" << args.M_beta << "\n";

        long long baseline_memory = current_memory_bytes();
        std::vector<int> metadata(nb, 0);
        faiss::IndexACORNFlat index(data_dim, args.M, args.gamma, metadata, args.M_beta);
        index.add(static_cast<faiss::idx_t>(nb), data_vectors.data());
        long long build_memory = current_memory_bytes();
        long long build_memory_delta =
                baseline_memory >= 0 && build_memory >= 0 ? build_memory - baseline_memory : -1;
        size_t index_size = serialized_index_size(index);

        std::cout << "Building per-query substring filter map...\n";
        std::vector<char> filter_map = make_filter_map(data_strings, query_strings);

        std::vector<int> ef_search_values = {
                8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
        std::vector<double> times_us;
        std::vector<double> recalls;

        int max_k = 0;
        for (int k : query_ks) {
            max_k = std::max(max_k, k);
        }
        if (max_k <= 0) {
            throw std::runtime_error("All query k values are non-positive.");
        }

        std::vector<faiss::idx_t> labels(nq * max_k);
        std::vector<float> distances(nq * max_k);

        for (int ef : ef_search_values) {
            faiss::SearchParametersACORN params;
            params.efSearch = ef;

            auto start = std::chrono::steady_clock::now();
            index.search(
                    static_cast<faiss::idx_t>(nq),
                    query_vectors.data(),
                    max_k,
                    distances.data(),
                    labels.data(),
                    filter_map.data(),
                    &params);
            auto end = std::chrono::steady_clock::now();
            double elapsed_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            double total_recall = 0.0;
            int effective = 0;
            for (size_t qi = 0; qi < nq; ++qi) {
                const auto& truth = ground_truth[qi];
                if (truth.empty()) {
                    continue;
                }
                std::unordered_set<int> truth_set(truth.begin(), truth.end());
                int correct = 0;
                int k = std::min(query_ks[qi], max_k);
                for (int j = 0; j < k; ++j) {
                    faiss::idx_t label = labels[qi * max_k + j];
                    if (label >= 0 && truth_set.find(static_cast<int>(label)) != truth_set.end()) {
                        ++correct;
                    }
                }
                total_recall += static_cast<double>(correct) / truth.size();
                ++effective;
            }

            double avg_time_us = nq ? elapsed_us / nq : 0.0;
            double avg_recall = effective ? total_recall / effective : 0.0;
            times_us.push_back(avg_time_us);
            recalls.push_back(avg_recall);
            std::cout << "ef_search=" << ef << ", time=" << avg_time_us
                      << " us, recall=" << avg_recall << "\n";
        }

        std::ofstream out(args.output_file);
        out << "ef_search,M,gamma,M_beta,time_us,recall,build_peak_memory_bytes,index_size_bytes\n";
        for (size_t i = 0; i < ef_search_values.size(); ++i) {
            out << ef_search_values[i] << ","
                << args.M << ","
                << args.gamma << ","
                << args.M_beta << ","
                << times_us[i] << ","
                << recalls[i] << ","
                << build_memory_delta << ","
                << index_size << "\n";
        }
        std::cout << "Wrote ACORN statistics to " << args.output_file << "\n";
    } catch (const std::exception& exc) {
        std::cerr << "ACORN benchmark failed: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}

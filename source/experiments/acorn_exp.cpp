#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "sa.h"

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
    std::string query_output_file;
    std::string load_index_file;
    std::string save_index_file;
    int M = 32;
    int gamma = 12;
    int M_beta = 32;
    double insertion_percentage = 0.0;
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

class MemoryPeakTracker {
public:
    explicit MemoryPeakTracker(std::chrono::milliseconds interval = std::chrono::milliseconds(50))
            : interval_(interval) {}

    void start() {
        stop_.store(false);
        record();
        sampler_ = std::thread([this] {
            while (!stop_.load()) {
                record();
                std::this_thread::sleep_for(interval_);
            }
        });
    }

    long long stop() {
        record();
        stop_.store(true);
        if (sampler_.joinable()) {
            sampler_.join();
        }
        record();
        return peak_bytes_;
    }

private:
    void record() {
        long long current = current_memory_bytes();
        if (current >= 0 && current > peak_bytes_) {
            peak_bytes_ = current;
        }
    }

    std::chrono::milliseconds interval_;
    std::atomic<bool> stop_{false};
    std::thread sampler_;
    long long peak_bytes_ = -1;
};

Args parse_args(int argc, char** argv) {
    if (argc < 7) {
        std::cerr
            << "Usage: " << argv[0]
            << " <string_data_file> <vector_data_file> <string_query_file>"
            << " <vector_query_file> <k_file> <ground_truth_file>"
            << " [--M=32] [--gamma=12] [--M-beta=32]"
            << " [--insertion-percentage=0]"
            << " [--output=acorn_hnsw_stats.csv] [--write-output=output.txt]"
            << " [--load-index=path] [--save-index=path]\n";
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
        } else if (arg.rfind("--insertion-percentage=", 0) == 0) {
            args.insertion_percentage = std::stod(value_after("--insertion-percentage="));
        } else if (arg.rfind("--output=", 0) == 0) {
            args.output_file = value_after("--output=");
        } else if (arg.rfind("--write-output=", 0) == 0) {
            args.query_output_file = value_after("--write-output=");
        } else if (arg.rfind("--load-index=", 0) == 0) {
            args.load_index_file = value_after("--load-index=");
        } else if (arg.rfind("--save-index=", 0) == 0) {
            args.save_index_file = value_after("--save-index=");
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }
    if (args.insertion_percentage < 0.0 || args.insertion_percentage > 100.0) {
        throw std::runtime_error("--insertion-percentage must be in [0, 100].");
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

GeneralizedSuffixAutomaton build_gsa(const std::vector<std::string>& data_strings) {
    GeneralizedSuffixAutomaton gsa;
    for (size_t id = 0; id < data_strings.size(); ++id) {
        gsa.add_string(static_cast<uint32_t>(id), data_strings[id]);
    }
    gsa.shrink_to_fit();
    return gsa;
}

std::vector<char> make_filter_map(
        const GeneralizedSuffixAutomaton& gsa,
        size_t nb,
        const std::vector<std::string>& query_strings) {
    const size_t nq = query_strings.size();
    std::vector<char> filter_map(nq * nb, 0);
    for (size_t qi = 0; qi < nq; ++qi) {
        int state = gsa.query(query_strings[qi]);
        if (state == -1) {
            continue;
        }
        for (uint32_t id : gsa.st[state].ids) {
            if (id < nb) {
                filter_map[qi * nb + id] = 1;
            }
        }
    }
    return filter_map;
}

unsigned long long current_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
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

        size_t insertion_count =
                static_cast<size_t>(static_cast<double>(nb) * args.insertion_percentage / 100.0);
        size_t base_nb = nb - insertion_count;
        if (nb > 0 && base_nb == 0) {
            throw std::runtime_error(
                    "--insertion-percentage leaves no base vectors to build the index.");
        }
        if (insertion_count > 0) {
            std::cout << "Using " << base_nb << " base vectors and "
                      << insertion_count << " insertion vectors.\n";
        }

        if (args.load_index_file.empty()) {
            std::cout << "Building ACORN index: nb=" << base_nb << ", dim=" << data_dim
                      << ", M=" << args.M << ", gamma=" << args.gamma
                      << ", M_beta=" << args.M_beta << "\n";
        } else {
            std::cout << "Loading ACORN index from " << args.load_index_file << "\n";
        }

        long long baseline_memory = current_memory_bytes();
        MemoryPeakTracker build_peak_tracker;
        build_peak_tracker.start();
        unsigned long long index_start_time = current_time_us();
        unsigned long long index_elapsed_us = 0;
        std::unique_ptr<faiss::Index> loaded_index;
        std::unique_ptr<faiss::IndexACORN> owned_index;
        faiss::IndexACORN* index = nullptr;
        if (args.load_index_file.empty()) {
            std::vector<int> metadata(base_nb, 0);
            owned_index = std::make_unique<faiss::IndexACORNFlat>(
                    data_dim, args.M, args.gamma, metadata, args.M_beta);
            owned_index->add(static_cast<faiss::idx_t>(base_nb), data_vectors.data());
            index_elapsed_us = current_time_us() - index_start_time;
            std::cout << "ACORN index built took " << index_elapsed_us << "us ("
                      << index_elapsed_us / 1000000 << "s)\n";
            index = owned_index.get();
        } else {
            loaded_index.reset(faiss::read_index(args.load_index_file.c_str()));
            index = dynamic_cast<faiss::IndexACORN*>(loaded_index.get());
            if (!index) {
                throw std::runtime_error(
                        "Loaded index is not an ACORN index: " + args.load_index_file);
            }
            if (index->d != data_dim) {
                throw std::runtime_error("Loaded ACORN index dimension does not match vectors.");
            }
            if (static_cast<size_t>(index->ntotal) != base_nb) {
                throw std::runtime_error("Loaded ACORN index size does not match data strings.");
            }
            index_elapsed_us = current_time_us() - index_start_time;
            std::cout << "ACORN index loaded in " << index_elapsed_us << "us ("
                      << index_elapsed_us / 1000000 << "s)\n";
        }
        std::cout << "Building GeneralizedSuffixAutomaton for ACORN base filters...\n";
        GeneralizedSuffixAutomaton gsa;
        for (size_t id = 0; id < base_nb; ++id) {
            gsa.add_string(static_cast<uint32_t>(id), data_strings[id]);
        }
        gsa.shrink_to_fit();
        std::cout << "Base GSA states: " << gsa.size() << ", total ids: "
                  << gsa.size_tot() << "\n";

        double average_insertion_time_us = -1.0;
        if (insertion_count > 0) {
            std::cout << "Inserting additional " << insertion_count
                      << " vectors and strings into ACORN index...\n";
            unsigned long long insertion_start_time = current_time_us();
            index->add(
                    static_cast<faiss::idx_t>(insertion_count),
                    data_vectors.data() + base_nb * data_dim);
            for (size_t i = 0; i < insertion_count; ++i) {
                size_t id = base_nb + i;
                gsa.add_string(static_cast<uint32_t>(id), data_strings[id]);
            }
            unsigned long long insertion_elapsed_us = current_time_us() - insertion_start_time;
            average_insertion_time_us =
                    static_cast<double>(insertion_elapsed_us) / insertion_count;
            std::cout << "Insertion took " << insertion_elapsed_us << " us ("
                      << average_insertion_time_us << " us/vector+string).\n";
        }
        if (static_cast<size_t>(index->ntotal) != nb) {
            throw std::runtime_error("ACORN index size does not match final data size.");
        }
        std::cout << "GSA states: " << gsa.size() << ", total ids: " << gsa.size_tot() << "\n";
        long long build_peak_memory = build_peak_tracker.stop();
        long long build_memory_delta =
                baseline_memory >= 0 && build_peak_memory >= 0
                        ? build_peak_memory - baseline_memory
                        : -1;
        if (!args.save_index_file.empty()) {
            std::cout << "Saving ACORN index to " << args.save_index_file << "\n";
            faiss::write_index(index, args.save_index_file.c_str());
        }
        size_t index_size = serialized_index_size(*index);
        std::cout << "peak memory consumption: " << build_memory_delta << " bytes\n";
        std::cout << "index size: " << index_size << " bytes\n";

        std::cout << "Building per-query filter map from GSA...\n";
        auto filter_start = std::chrono::steady_clock::now();
        std::vector<char> filter_map = make_filter_map(gsa, nb, query_strings);
        auto filter_end = std::chrono::steady_clock::now();
        double filter_elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start)
                        .count();
        std::cout << "filter map build time: " << filter_elapsed_us << " us\n";

        std::vector<int> ef_search_values = {
                8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
        std::vector<double> times_us;
        std::vector<double> recalls;
        std::vector<std::vector<std::string>> query_output_rows(nq);

        auto record_query_output = [&](int ef, const std::vector<faiss::idx_t>& current_labels, int current_max_k) {
            for (size_t qi = 0; qi < nq; ++qi) {
                std::ostringstream row;
                row << "ef_search = " << ef << ", k = " << query_ks[qi] << ":";
                int k = std::min(query_ks[qi], current_max_k);
                for (int j = 0; j < k; ++j) {
                    faiss::idx_t label = current_labels[qi * current_max_k + j];
                    if (label >= 0) {
                        row << " " << static_cast<int>(label);
                    }
                }
                query_output_rows[qi].push_back(row.str());
            }
        };

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
            index->search(
                    static_cast<faiss::idx_t>(nq),
                    query_vectors.data(),
                    max_k,
                    distances.data(),
                    labels.data(),
                    filter_map.data(),
                    &params);
            auto end = std::chrono::steady_clock::now();
            double search_elapsed_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            double elapsed_us = filter_elapsed_us + search_elapsed_us;

            record_query_output(ef, labels, max_k);

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
                      << " us, recall=" << avg_recall
                      << " (includes filter map build time)\n";
        }

        std::ofstream out(args.output_file);
        out << "ef_search,M,gamma,M_beta,time_us,recall,average_insertion_time_us,"
               "build_peak_memory_bytes,index_size_bytes\n";
        for (size_t i = 0; i < ef_search_values.size(); ++i) {
            out << ef_search_values[i] << ","
                << args.M << ","
                << args.gamma << ","
                << args.M_beta << ","
                << times_us[i] << ","
                << recalls[i] << ",";
            if (average_insertion_time_us >= 0.0) {
                out << average_insertion_time_us;
            }
            out << ","
                << build_memory_delta << ","
                << index_size << "\n";
        }
        std::cout << "Wrote ACORN statistics to " << args.output_file << "\n";
        if (!args.query_output_file.empty()) {
            std::ofstream query_out(args.query_output_file);
            for (size_t qi = 0; qi < query_output_rows.size(); ++qi) {
                query_out << "Query " << (qi + 1) << ":\n";
                for (const auto& row : query_output_rows[qi]) {
                    query_out << row << "\n";
                }
            }
            std::cout << "Wrote ACORN query output to " << args.query_output_file << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "ACORN benchmark failed: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}

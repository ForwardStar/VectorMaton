#include "headers.h"
#include "exact.h"
#include "baseline.h"
#include "vector_db.h"

std::stringstream timeFormatting(unsigned long long microSeconds) {
    std::stringstream ret;
    ret << microSeconds << "μs" << " (";
    unsigned long long seconds = microSeconds / 1000000ull;
    if (seconds < 60) {
        ret << seconds << "s";
    }
    else if (seconds < 3600) {
        ret << seconds / 60ull << "min " << seconds % 60ull << "s";
    }
    else {
        ret << seconds / 3600ull << "h " << seconds % 3600ull / 60ull << "min " << seconds % 60ull << "s";
    }
    ret << ")";
    return ret;
}

unsigned long long currentTime() {
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::chrono::system_clock::duration duration = now.time_since_epoch();
    unsigned long long microSecondsOfDuration = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    return microSecondsOfDuration;
}

class Logger {
public:
    enum class Level { DEBUG = 0, INFO, WARNING, ERROR, NONE };

    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void set_level(Level lvl) { level_ = lvl; }

    template <typename... Args>
    void log(Level msg_level, const char* file, int line, Args&&... args) {
        if (msg_level < level_) return;

        std::ostringstream ss;
        (stream_arg(ss, std::forward<Args>(args)), ...);

        std::lock_guard<std::mutex> lk(mutex_);
        std::cout << make_header(msg_level) << ss.str();
        if (msg_level == Level::DEBUG || msg_level == Level::ERROR) {
            std::cout << " (" << file << ":" << line << ")";
        }
        std::cout << std::endl;
    }

    template <typename... Args> void debug(const char* f, int l, Args&&... a) { log(Level::DEBUG, f, l, std::forward<Args>(a)...); }
    template <typename... Args> void info(const char* f, int l, Args&&... a) { log(Level::INFO, f, l, std::forward<Args>(a)...); }
    template <typename... Args> void warn(const char* f, int l, Args&&... a) { log(Level::WARNING, f, l, std::forward<Args>(a)...); }
    template <typename... Args> void error(const char* f, int l, Args&&... a) { log(Level::ERROR, f, l, std::forward<Args>(a)...); }

private:
    Logger() : level_(Level::INFO) {}

    std::string make_header(Level lvl) {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto t = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        std::tm tm_snapshot;
    #if defined(_MSC_VER)
        localtime_s(&tm_snapshot, &t);
    #else
        localtime_r(&t, &tm_snapshot);
    #endif

        std::ostringstream os;
        os << "[" << std::put_time(&tm_snapshot, "%Y-%m-%d %H:%M:%S")
           << "." << std::setw(3) << std::setfill('0') << ms.count()
           << "]"
           << "[" << level_name(lvl) << "] ";
        return os.str();
    }

    static std::string level_name(Level lvl) {
        switch (lvl) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO: return "INFO";
            case Level::WARNING: return "WARN";
            case Level::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    template <typename T>
    static void stream_arg(std::ostringstream& ss, T&& v) { ss << std::forward<T>(v); }

    std::mutex mutex_;
    Level level_;
};

// Macros for convenience
#define LOG_DEBUG(...) Logger::instance().debug(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)  Logger::instance().info(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  Logger::instance().warn(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().error(__FILE__, __LINE__, __VA_ARGS__)

int main(int argc, char * argv[]) {
    if (argc < 8) {
        LOG_ERROR("Usage: ./main <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k_query_file> <output_file> <Exact|Baseline|VectorDB>");
        return 1;
    }

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
    }

    // Read strings
    LOG_DEBUG("String data file: ", argv[1]);
    std::vector<std::string> strings;
    std::ifstream f_strings(argv[1]);
    std::string s;
    while (f_strings >> s) {
        strings.emplace_back(s);
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
    }

    // Log the number of strings and vectors
    LOG_DEBUG("Number of strings: ", strings.size());
    LOG_DEBUG("Number of vectors: ", vectors.size());
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

    if (std::strcmp(argv[argc - 1], "Exact") == 0) {
        LOG_INFO("Using Exact search");
        ExactSearch es;
        LOG_DEBUG("Inserting strings and vectors into ExactSearch");
        unsigned long long start_time = currentTime();
        for (size_t i = 0; i < strings.size(); ++i) {
            es.insert(vectors[i], strings[i]);
        }
        LOG_INFO("ExactSearch insertion took ", timeFormatting(currentTime() - start_time).str());
        LOG_DEBUG("Processing queries");
        start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = es.query(queried_vectors[i], queried_strings[i], queried_k[i]);
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
        LOG_DEBUG("Inserting strings and vectors into Baseline");
        unsigned long long start_time = currentTime();
        for (size_t i = 0; i < strings.size(); ++i) {
            bs.insert(vectors[i], strings[i]);
        }
        LOG_INFO("Baseline insertion took ", timeFormatting(currentTime() - start_time).str());
        LOG_DEBUG("Processing queries");
        start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = bs.query(queried_vectors[i], queried_strings[i], queried_k[i]);
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

    if (std::strcmp(argv[argc - 1], "VectorDB") == 0) {
        LOG_INFO("Using VectorDB search");
        VectorDB vdb;
        LOG_DEBUG("Inserting strings and vectors into VectorDB");
        unsigned long long start_time = currentTime();
        for (size_t i = 0; i < strings.size(); ++i) {
            vdb.insert(vectors[i], strings[i]);
        }
        LOG_INFO("VectorDB insertion took ", timeFormatting(currentTime() - start_time).str());
        LOG_INFO("Building VectorDB indices");
        start_time = currentTime();
        vdb.build();
        LOG_INFO("VectorDB building took ", timeFormatting(currentTime() - start_time).str());
        LOG_DEBUG("Processing queries");
        start_time = currentTime();
        std::vector<std::vector<int>> all_results;
        for (size_t i = 0; i < queried_strings.size(); ++i) {
            auto res = vdb.query(queried_vectors[i], queried_strings[i], queried_k[i]);
            all_results.emplace_back(res);
        }
        LOG_INFO("VectorDB query processing took ", timeFormatting(currentTime() - start_time).str());
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
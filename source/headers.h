#ifndef HEADER_H
#define HEADER_H

#pragma once

#include <iostream>
#include <unordered_map>
#include <map>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <memory>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <functional>
#include <sstream>
#include <mutex>
#include <numeric>
#include <chrono>
#include <ctime>
#include <iomanip>

// Calculate Euclidean distance between two vectors
inline float distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

inline std::stringstream timeFormatting(unsigned long long microSeconds) {
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

inline unsigned long long currentTime() {
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

#endif
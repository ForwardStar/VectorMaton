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
#include <fstream>
#include <sstream>
#include <mutex>
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

#endif
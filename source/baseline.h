#ifndef BASELINE_H
#define BASELINE_H

#include "headers.h"
#include "nsw.h"

class Baseline {
    public:
        std::vector<std::vector<float>> vecs;
        std::vector<std::string> strs;
        NSW* nsw = nullptr;

        int insert(const std::vector<float>& vec, const std::string &s);
        void remove(int id);
        std::vector<int> query(const std::vector<float>& vec, const std::string &s, int k, int threshold=2048);
        
        Baseline() {};
        ~Baseline() {
            delete nsw;
        };
};

#endif
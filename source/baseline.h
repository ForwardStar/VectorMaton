#ifndef BASELINE_H
#define BASELINE_H

#include "headers.h"
#include "nsw.h"

class Baseline {
    private:
        std::vector<std::vector<float>> vecs;
        std::vector<std::string> strs;
        NSW nsw;

    public:
        int insert(const std::vector<float>& vec, const std::string &s);
        void remove(int id);
        std::vector<int> query(const std::vector<float>& vec, const std::string &s, int k);
        
        Baseline() {};
        ~Baseline() {};
};

#endif
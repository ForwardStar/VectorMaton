#ifndef EXACT_H
#define EXACT_H

#include "headers.h"

class ExactSearch {
    private:
        std::vector<std::vector<float>> vecs;
        std::vector<std::string> strs;
        
    public:
        int insert(const std::vector<float>& vec, const std::string &s);
        void remove(int id);
        std::vector<int> query(const std::vector<float>& vec, const std::string &s, int k);

        ExactSearch() {};
        ~ExactSearch() {};
};

#endif
#include "vector_db.h"
#include <iostream>

int main() {
    VectorDB db;
    std::cout << "Inserting {1.0, 2.0, 3.0} associated with 'banana'." << std::endl;
    db.insert({1.0, 2.0, 3.0}, "banana");
    std::cout << "Inserting {4.0, 5.0, 6.0} associated with 'anana'." << std::endl;
    db.insert({4.0, 5.0, 6.0}, "anana");
    std::cout << "Inserting {7.0, 8.0, 9.0} associated with 'nana'." << std::endl;
    db.insert({7.0, 8.0, 9.0}, "nana");
    std::cout << "Inserting {10.0, 11.0, 12.0} associated with 'ana'." << std::endl;
    db.insert({10.0, 11.0, 12.0}, "ana");
    std::cout << "Inserting {13.0, 14.0, 15.0} associated with 'na'." << std::endl;
    db.insert({13.0, 14.0, 15.0}, "na");

    auto print_res = [](const std::vector<int>& res){
        std::cout << "Result: {";
        for (size_t i=0;i<res.size();++i) {
            std::cout << res[i];
            if (i+1<res.size()) std::cout << ", ";
        }
        std::cout << "}\n";
    };
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'ana'." << std::endl;
    print_res(db.query({9.0, 10.0, 11.0}, "ana", 2)); // {2, 3}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'nana'." << std::endl;
    print_res(db.query({9.0, 10.0, 11.0}, "nana", 2)); // {1, 2}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'anana'." << std::endl;
    print_res(db.query({9.0, 10.0, 11.0}, "anana", 2)); // {0, 1}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'banana'." << std::endl;
    print_res(db.query({9.0, 10.0, 11.0}, "banana", 2)); // {0}

    return 0;
}

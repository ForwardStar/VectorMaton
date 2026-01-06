#include "vectormaton.h"
#include <iostream>

int main() {
    float* vecs;
    vecs = new float[15];
    for (int i = 0; i < 15; i++) {
        vecs[i] = i + 1;
    }
    
    std::string strings[5] = {
        "banana",
        "anana",
        "nana",
        "ana",
        "na"
    };

    VectorMaton db;
    db.set_min_build_threshold(0);
    db.set_vectors(vecs, 3, 5);
    db.set_strings(strings);
    db.build_full();

    auto print_res = [](const std::vector<int>& res){
        std::cout << "Result: {";
        for (size_t i=0;i<res.size();++i) {
            std::cout << res[i];
            if (i+1<res.size()) std::cout << ", ";
        }
        std::cout << "}\n";
    };

    float query_vec1[3] = {9.0, 10.0, 11.0};
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'ana'." << std::endl;
    print_res(db.query(query_vec1, "ana", 2)); // {2, 3}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'nana'." << std::endl;
    print_res(db.query(query_vec1, "nana", 2)); // {1, 2}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'anana'." << std::endl;
    print_res(db.query(query_vec1, "anana", 2)); // {0, 1}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'banana'." << std::endl;
    print_res(db.query(query_vec1, "banana", 2)); // {0}

    // Test build_smart
    VectorMaton pdb;
    pdb.set_vectors(vecs, 3, 5);
    pdb.set_strings(strings);
    pdb.build_smart();
    std::cout << "Smart VectorMaton tests:" << std::endl;
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'ana'." << std::endl;
    print_res(pdb.query(query_vec1, "ana", 2)); // {2, 3}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'nana'." << std::endl;
    print_res(pdb.query(query_vec1, "nana", 2)); // {1, 2}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'anana'." << std::endl;
    print_res(pdb.query(query_vec1, "anana", 2)); // {0, 1}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'banana'." << std::endl;
    print_res(pdb.query(query_vec1, "banana", 2)); // {0}
    
    return 0;
}

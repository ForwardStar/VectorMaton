#include "vectormaton.h"
#include <iostream>

int main() {
    std::vector<float> vecs(15);
    for (int i = 0; i < 15; i++) {
        vecs[i] = i + 1;
    }
    
    std::vector<std::string> strings = {
        "banana",
        "anana",
        "nana",
        "ana",
        "na"
    };

    VectorMaton db;
    db.set_min_build_threshold(0);
    db.set_vectors(vecs, 3);
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
    pdb.set_vectors(vecs, 3);
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

    // Test save/load index
    pdb.save_index("test_vectormaton");
    VectorMaton pdb1;
    pdb1.set_vectors(vecs, 3);
    pdb1.set_strings(strings);
    pdb1.load_index("test_vectormaton");
    std::cout << "Load VectorMaton tests:" << std::endl;
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'ana'." << std::endl;
    print_res(pdb1.query(query_vec1, "ana", 2)); // {2, 3}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'nana'." << std::endl;
    print_res(pdb1.query(query_vec1, "nana", 2)); // {1, 2}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'anana'." << std::endl;
    print_res(pdb1.query(query_vec1, "anana", 2)); // {0, 1}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'banana'." << std::endl;
    print_res(pdb1.query(query_vec1, "banana", 2)); // {0}

    // Test build parallel
    VectorMaton pdb2;
    pdb2.set_vectors(vecs, 3);
    pdb2.set_strings(strings);
    pdb2.build_parallel();
    std::cout << "Parallel VectorMaton tests:" << std::endl;
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'ana'." << std::endl;
    print_res(pdb2.query(query_vec1, "ana", 2)); // {2, 3}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'nana'." << std::endl;
    print_res(pdb2.query(query_vec1, "nana", 2)); // {1, 2}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'anana'." << std::endl;
    print_res(pdb2.query(query_vec1, "anana", 2)); // {0, 1}
    std::cout << "2-NNs of {9.0, 10.0, 11.0} associated with 'banana'." << std::endl;
    print_res(pdb2.query(query_vec1, "banana", 2)); // {0}
    
    return 0;
}

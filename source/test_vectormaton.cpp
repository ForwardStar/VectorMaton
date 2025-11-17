#include "vectormaton.h"
#include <iostream>

int main() {
    float** vecs;
    vecs = new float*[5];
    vecs[0] = new float[3]{1.0, 2.0, 3.0};
    vecs[1] = new float[3]{4.0, 5.0, 6.0};
    vecs[2] = new float[3]{7.0, 8.0, 9.0};
    vecs[3] = new float[3]{10.0, 11.0, 12.0};
    vecs[4] = new float[3]{13.0, 14.0, 15.0};
    
    std::string strings[5] = {
        "banana",
        "anana",
        "nana",
        "ana",
        "na"
    };

    VectorMaton db;
    db.set_vectors(vecs, 3, 5);
    db.set_strings(strings);
    db.build();

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
    
    return 0;
}

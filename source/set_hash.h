#ifndef SET_HASH_H
#define SET_HASH_H

#include "headers.h"

namespace sethash {

    // Method 1: XOR of all elements
    uint64_t xor_hash(const std::vector<uint32_t>& S);

    // Method 2: SUM of all elements (mod 2^64)
    uint64_t sum_hash(const std::vector<uint32_t>& S);

    // Method 3: Multiplicative Polynomial Hash
    uint64_t poly_hash(const std::vector<uint32_t>& S);

    // Method 4: Sort + SHA256 hash (returns hex string)
    std::string sha256_hash(const std::vector<uint32_t>& S);

} // namespace sethash

#endif // SET_HASH_H

#include "set_hash.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>  // For SHA256

namespace sethash {

    // -------------------------------
    // Method 1: XOR of elements
    // -------------------------------
    uint64_t xor_hash(const std::vector<uint32_t>& S) {
        uint64_t h = 0;
        for (uint32_t x : S) {
            h ^= static_cast<uint64_t>(x);
        }
        return h;
    }

    // -------------------------------
    // Method 2: SUM of elements
    // -------------------------------
    uint64_t sum_hash(const std::vector<uint32_t>& S) {
        uint64_t h = 0;
        for (uint32_t x : S) {
            h += static_cast<uint64_t>(x);
        }
        return h;
    }

    // -------------------------------
    // Method 3: Multiplicative Polynomial Hash
    // -------------------------------
    uint64_t poly_hash(const std::vector<uint32_t>& S) {
        const uint64_t mod = (1ULL << 61) - 1; // large prime
        const uint64_t p   = 1000003ULL;       // random odd constant
        __uint128_t h = 1;
        for (uint32_t x : S) {
            h = (h * (p + x)) % mod;
        }
        return static_cast<uint64_t>(h);
    }

    // -------------------------------
    // Method 4: Sort + SHA256
    // -------------------------------
    std::string sha256_hash(const std::vector<uint32_t>& S) {
        std::vector<uint32_t> sortedS = S;

        // In our GSA algorithm, the set is already sorted
        // std::sort(sortedS.begin(), sortedS.end());

        // Convert to string
        std::ostringstream oss;
        for (uint32_t x : sortedS) {
            oss << x << ",";
        }
        std::string data = oss.str();

        // Compute SHA256
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(data.c_str()), data.size(), hash);

        // Convert to hex string
        std::ostringstream hexstr;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            hexstr << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        return hexstr.str();
    }

} // namespace sethash

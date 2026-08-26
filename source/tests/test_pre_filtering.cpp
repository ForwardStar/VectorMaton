#include "pre_filtering.h"

#include <cassert>

int main() {
    PreFiltering pre_filtering;
    pre_filtering.set_vectors({0.0f}, 1);
    pre_filtering.set_strings({std::string(256, 'a')});
    pre_filtering.build();

    size_t gsa_id_storage = 0;
    for (const auto& state : pre_filtering.gsa.st) {
        gsa_id_storage += sizeof(uint32_t) * state.ids.capacity();
    }
    assert(pre_filtering.size() >= gsa_id_storage + sizeof(float));

    // More elements than GSA states previously caused an out-of-bounds read.
    PreFiltering repeated;
    repeated.set_vectors({0.0f, 1.0f, 2.0f}, 1);
    repeated.set_strings({"a", "a", "a"});
    repeated.build();
    assert(repeated.gsa.st.size() < 3);
    assert(repeated.size() > 0);

    return 0;
}

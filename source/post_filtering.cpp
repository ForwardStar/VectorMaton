#include "post_filtering.h"

void PostFiltering::set_vectors(float* vectors, int dimension, int num_elems) {
    vecs = vectors;
    dim = dimension;
    num_elements = num_elems;
}

void PostFiltering::set_strings(std::string* strings) {
    strs = strings;
}

void PostFiltering::set_ef(int ef) {
    hnsw->setEf(ef);
}

void PostFiltering::build() {
    if (!hnsw) {
        space = new hnswlib::L2Space(dim);
        hnsw = new hnswlib::HierarchicalNSW<float>(space, num_elements, vecs, 16, 200);
        for (int i = 0; i < num_elements; i++) {
            hnsw->addPoint(i);
        }
    }
}

void PostFiltering::load_index(const char* input_folder) {
    namespace fs = std::filesystem;
    fs::path in_path(input_folder);

    fs::path hnsw_file = in_path / "hnsw";

    LOG_DEBUG("Loading HNSW data");
    space = new hnswlib::L2Space(dim);
    if (fs::exists(hnsw_file)) {
        hnsw = new hnswlib::HierarchicalNSW<float>(space, hnsw_file.string(), num_elements);
    }
    else {
        hnsw = nullptr;
    }
}

void PostFiltering::save_index(const char* output_folder) {
    namespace fs = std::filesystem;
    fs::path out_path(output_folder);

    if (!fs::exists(out_path)) {
        fs::create_directories(out_path);  // safer than create_directory
    }

    LOG_DEBUG("Saving HNSW data");
    fs::path hnsw_file = out_path / "hnsw";
    std::string tmp = hnsw_file.string();
    hnsw->saveIndex(tmp);
}

size_t PostFiltering::size() {
    size_t total_size = 0;
    total_size += hnsw->indexFileSize();
    for (int i = 0; i < num_elements; i++) {
        total_size += sizeof(std::string) + strs[i].capacity(); // size of each string
        total_size += sizeof(float) * dim; // size of each vector
    }
    return total_size;
}

std::vector<int> PostFiltering::query(const float* vec, const std::string &s, int k, int ef_search) {
    std::vector<int> results;
    if (ef_search) set_ef(ef_search);
    auto tmp = hnsw->searchKnnCloserFirst(vec, (ef_search != 0 ? ef_search : k));
    for (auto& pair : tmp) {
        int id = pair.second;
        if (strs[id].find(s) != std::string::npos) {
            results.push_back(id);
        }
        if (results.size() >= static_cast<size_t>(k)) {
            break;
        }
    }
    return results;
}
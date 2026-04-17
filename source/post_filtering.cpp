#include "post_filtering.h"

void PostFiltering::set_vectors(const std::vector<float>& vectors, int dimension) {
    vecs = vectors;
    dim = dimension;
    num_elements = dim == 0 ? 0 : static_cast<int>(vecs.size()) / dim;
}

void PostFiltering::set_strings(const std::vector<std::string>& strings) {
    strs = strings;
}

void PostFiltering::set_ef(int ef) {
    hnsw->setEf(ef);
}

void PostFiltering::build() {
    if (!hnsw) {
        space = new hnswlib::L2Space(dim);
        hnsw = new hnswlib::HierarchicalNSW<float>(space, num_elements, vecs.data(), 16, 200);
        for (int i = 0; i < num_elements; i++) {
            hnsw->addPoint(i);
        }
    }
}

void PostFiltering::insert(const std::vector<float>& vec, const std::string& str) {
    if (static_cast<int>(vec.size()) != dim) return;
    vecs.insert(vecs.end(), vec.begin(), vec.end());
    strs.push_back(str);
    num_elements++;
    const int id = num_elements - 1;

    // HNSW stores a base pointer into the contiguous vector buffer.
    if (hnsw) {
        if (num_elements > static_cast<int>(hnsw->max_elements_)) {
            hnsw->resizeIndex(num_elements);
        }
        hnsw->external_data_ = reinterpret_cast<const char*>(vecs.data());
    }
    if (!hnsw) {
        space = new hnswlib::L2Space(dim);
        hnsw = new hnswlib::HierarchicalNSW<float>(space, num_elements, vecs.data(), 16, 200);
    }
    hnsw->addPoint(id);
}

void PostFiltering::load_index(const char* input_folder) {
    namespace fs = std::filesystem;
    fs::path in_path(input_folder);

    fs::path hnsw_file = in_path / "hnsw";

    LOG_DEBUG("Loading HNSW data");
    space = new hnswlib::L2Space(dim);
    if (fs::exists(hnsw_file)) {
        hnsw = new hnswlib::HierarchicalNSW<float>(space, hnsw_file.string(), vecs.data());
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
    // for (int i = 0; i < num_elements; i++) {
    //     total_size += sizeof(std::string) + strs[i].capacity(); // size of each string
    //     total_size += sizeof(float) * dim; // size of each vector
    // }
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

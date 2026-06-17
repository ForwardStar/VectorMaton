#include "bm25_filtering.h"

std::vector<std::string> BM25Filtering::tokenize(const std::string& s) {
    std::vector<std::string> tokens;
    const int gram_size = 2;
    if (s.size() < static_cast<size_t>(gram_size)) {
        if (!s.empty()) {
            tokens.emplace_back(s);
        }
        return tokens;
    }
    for (size_t i = 0; i <= s.size() - gram_size; ++i) {
        tokens.emplace_back(s.substr(i, gram_size));
    }
    return tokens;
}

int BM25Filtering::get_tf(const std::vector<std::pair<uint32_t, int>>& posting, int doc_id) {
    auto cmp = [](const std::pair<uint32_t, int>& a, uint32_t b) {
        return a.first < b;
    };
    auto it = std::lower_bound(posting.begin(), posting.end(), static_cast<uint32_t>(doc_id), cmp);
    if (it != posting.end() && it->first == static_cast<uint32_t>(doc_id)) {
        return it->second;
    }
    return 0;
}

void BM25Filtering::set_vectors(const std::vector<float>& vectors, int dimension) {
    vecs = vectors;
    dim = dimension;
    num_elements = dim == 0 ? 0 : static_cast<int>(vecs.size()) / dim;
    updatePeakMemoryUsage(peak_memory_usage);
}

void BM25Filtering::set_strings(const std::vector<std::string>& strings) {
    strs = strings;
    updatePeakMemoryUsage(peak_memory_usage);
}

void BM25Filtering::set_ef(int ef) {
    for (auto& hnsw : hnsws) {
        if (hnsw) {
            hnsw->setEf(ef);
        }
    }
}

void BM25Filtering::build_hnsw() {
    if (!space) {
        space = new hnswlib::L2Space(dim);
    }
    hnsws.assign(postings.size(), nullptr);
    for (int term_id = 0; term_id < postings.size(); term_id++) {
        const auto& posting = postings[term_id];
        if (posting.empty()) {
            continue;
        }
        hnsws[term_id] = new hnswlib::HierarchicalNSW<float>(space, posting.size(), vecs.data(), 16, 200);
        for (const auto& [doc_id, tf] : posting) {
            hnsws[term_id]->addPoint(doc_id);
        }
        updatePeakMemoryUsage(peak_memory_usage);
    }
}

void BM25Filtering::build() {
    term_to_id.clear();
    id_to_term.clear();
    postings.clear();
    doc_lengths.assign(num_elements, 0);
    avg_doc_len = 0.0;

    std::unordered_map<std::string, std::vector<std::pair<uint32_t, int>>> posting_map;
    posting_map.reserve(num_elements * 2);

    for (int i = 0; i < num_elements; i++) {
        auto tokens = tokenize(strs[i]);
        doc_lengths[i] = tokens.size();
        avg_doc_len += doc_lengths[i];

        std::unordered_map<std::string, int> tf;
        for (const auto& token : tokens) {
            tf[token]++;
        }

        for (const auto& [term, count] : tf) {
            posting_map[term].emplace_back(i, count);
        }
    }

    if (num_elements > 0) {
        avg_doc_len /= static_cast<double>(num_elements);
    }

    int term_id = 0;
    id_to_term.reserve(posting_map.size());
    postings.reserve(posting_map.size());
    idf.reserve(posting_map.size());

    for (auto& entry : posting_map) {
        const std::string& term = entry.first;
        term_to_id[term] = term_id;
        id_to_term.emplace_back(term);
        postings.emplace_back(std::move(entry.second));
        int df = postings.back().size();
        idf.emplace_back(compute_idf(df));
        term_id++;
    }

    build_hnsw();
    updatePeakMemoryUsage(peak_memory_usage);
}

void BM25Filtering::insert(const std::vector<float>& vec, const std::string& str) {
    if (static_cast<int>(vec.size()) != dim) return;
    vecs.insert(vecs.end(), vec.begin(), vec.end());
    strs.push_back(str);
    num_elements++;
    int doc_id = num_elements - 1;

    auto tokens = tokenize(str);
    int document_length = tokens.size();
    doc_lengths.push_back(document_length);
    avg_doc_len = ((avg_doc_len * (num_elements - 1)) + document_length) / num_elements;

    std::unordered_map<std::string, int> tf;
    for (const auto& token : tokens) {
        tf[token]++;
    }

    for (const auto& [term, count] : tf) {
        int term_id;
        auto it = term_to_id.find(term);
        if (it == term_to_id.end()) {
            term_id = id_to_term.size();
            term_to_id[term] = term_id;
            id_to_term.emplace_back(term);
            postings.emplace_back();
            idf.emplace_back(0.0);
            hnsws.emplace_back(nullptr);
        } else {
            term_id = it->second;
        }
        postings[term_id].emplace_back(doc_id, count);
        idf[term_id] = compute_idf(postings[term_id].size());

        if (hnsws[term_id]) {
            if (num_elements > static_cast<int>(hnsws[term_id]->max_elements_)) {
                hnsws[term_id]->resizeIndex(num_elements);
            }
            hnsws[term_id]->external_data_ = reinterpret_cast<const char*>(vecs.data());
            hnsws[term_id]->addPoint(doc_id);
        } else {
            if (!space) {
                space = new hnswlib::L2Space(dim);
            }
            hnsws[term_id] = new hnswlib::HierarchicalNSW<float>(space, postings[term_id].size(), vecs.data(), 16, 200);
            for (const auto& [id, tfcount] : postings[term_id]) {
                hnsws[term_id]->addPoint(id);
            }
        }
    }

    updatePeakMemoryUsage(peak_memory_usage);
}

void BM25Filtering::load_index(const char* input_folder) {
    namespace fs = std::filesystem;
    fs::path in_path(input_folder);
    fs::path meta_file = in_path / "bm25.in";

    if (!fs::exists(meta_file)) {
        LOG_WARN("BM25 index metadata not found: ", meta_file.string());
        return;
    }

    std::ifstream f(meta_file.string());
    int num_terms = 0;
    f >> num_elements >> dim >> avg_doc_len;
    int doc_count = 0;
    f >> doc_count;
    doc_lengths.assign(doc_count, 0);
    for (int i = 0; i < doc_count; i++) {
        f >> doc_lengths[i];
    }

    f >> num_terms;
    id_to_term.assign(num_terms, std::string());
    postings.assign(num_terms, std::vector<std::pair<uint32_t, int>>());
    idf.assign(num_terms, 0.0);
    term_to_id.clear();

    for (int term_id = 0; term_id < num_terms; term_id++) {
        int term_size = 0;
        f >> term_size;
        std::string term;
        term.reserve(term_size);
        f >> term;
        id_to_term[term_id] = term;
        term_to_id[term] = term_id;
        int posting_size = 0;
        f >> posting_size;
        postings[term_id].resize(posting_size);
        for (int j = 0; j < posting_size; j++) {
            uint32_t doc_id;
            int tf;
            f >> doc_id >> tf;
            postings[term_id][j].first = doc_id;
            postings[term_id][j].second = tf;
        }
        idf[term_id] = compute_idf(postings[term_id].size());
    }
    f.close();

    build_hnsw();
    updatePeakMemoryUsage(peak_memory_usage);
}

void BM25Filtering::save_index(const char* output_folder) {
    namespace fs = std::filesystem;
    fs::path out_path(output_folder);
    if (!fs::exists(out_path)) {
        fs::create_directories(out_path);
    }

    fs::path meta_file = out_path / "bm25.in";
    std::ofstream f(meta_file.string());
    f << num_elements << " " << dim << " " << avg_doc_len << "\n";
    f << doc_lengths.size();
    for (int len : doc_lengths) {
        f << " " << len;
    }
    f << "\n";
    f << postings.size() << "\n";
    for (int term_id = 0; term_id < postings.size(); term_id++) {
        f << id_to_term[term_id].size() << " " << id_to_term[term_id] << " " << postings[term_id].size();
        for (const auto& [doc_id, tf] : postings[term_id]) {
            f << " " << doc_id << " " << tf;
        }
        f << "\n";
    }
    f.close();

    for (int term_id = 0; term_id < hnsws.size(); term_id++) {
        if (!hnsws[term_id]) continue;
        std::string filename = "bm25_hnsw_" + std::to_string(term_id);
        fs::path hnsw_file = out_path / filename;
        hnsws[term_id]->saveIndex(hnsw_file.string());
    }

    updatePeakMemoryUsage(peak_memory_usage);
}

size_t BM25Filtering::size() {
    size_t total_size = 0;
    size_t hnsw_size = 0;
    for (auto* hnsw : hnsws) {
        if (hnsw) {
            hnsw_size += hnsw->indexFileSize();
        }
    }
    total_size += hnsw_size;
    for (const auto& term : id_to_term) {
        total_size += sizeof(std::string) + term.capacity();
    }
    size_t posting_size = 0;
    for (const auto& posting : postings) {
        posting_size += posting.size() * (sizeof(uint32_t) + sizeof(int));
    }
    total_size += posting_size;
    total_size += doc_lengths.size() * sizeof(int);
    total_size += term_to_id.size() * (sizeof(std::string) + sizeof(int));
    return total_size;
}

std::vector<int> BM25Filtering::query(const float* vec, const std::string &s, int k, int ef_search) {
    std::vector<std::string> tokens = tokenize(s);
    if (tokens.empty() || postings.empty()) {
        return {};
    }

    std::unordered_map<int, int> query_tf;
    for (const auto& token : tokens) {
        auto it = term_to_id.find(token);
        if (it != term_to_id.end()) {
            query_tf[it->second]++;
        }
    }
    if (query_tf.empty()) {
        return {};
    }

    std::unordered_set<int> candidate_set;
    int per_term_limit = std::max(k * 4, 64);
    for (const auto& [term_id, qfreq] : query_tf) {
        if (term_id < 0 || term_id >= postings.size()) {
            continue;
        }
        const auto& posting = postings[term_id];
        if (posting.empty()) {
            continue;
        }
        if (hnsws[term_id]) {
            if (ef_search != 0) {
                hnsws[term_id]->setEf(ef_search);
            }
            int search_k = std::min(static_cast<int>(posting.size()), per_term_limit);
            hnsws[term_id]->external_data_ = reinterpret_cast<const char*>(vecs.data());
            auto tmp = hnsws[term_id]->searchKnnCloserFirst(vec, search_k);
            for (const auto& pair : tmp) {
                candidate_set.insert(pair.second);
            }
        } else {
            for (const auto& pair : posting) {
                candidate_set.insert(pair.first);
            }
        }
    }
    if (candidate_set.empty()) {
        return {};
    }

    std::vector<std::pair<double, int>> scored_candidates;
    scored_candidates.reserve(candidate_set.size());
    const double k1 = 1.5;
    const double b = 0.75;
    const double k3 = 1000.0;

    for (int doc_id : candidate_set) {
        double score = 0.0;
        double dl = doc_lengths[doc_id];
        for (const auto& [term_id, qfreq] : query_tf) {
            int tf = get_tf(postings[term_id], doc_id);
            if (tf == 0) continue;
            double q_weight = ((qfreq * (k3 + 1.0)) / (qfreq + k3));
            double numerator = tf * (k1 + 1.0);
            double denominator = tf + k1 * (1.0 - b + b * dl / avg_doc_len);
            score += idf[term_id] * (numerator / denominator) * q_weight;
        }
        scored_candidates.emplace_back(score, doc_id);
    }

    int candidate_limit = std::min(static_cast<int>(scored_candidates.size()), std::max(k * 4, k));
    if (scored_candidates.size() > candidate_limit) {
        std::nth_element(scored_candidates.begin(), scored_candidates.begin() + candidate_limit, scored_candidates.end(), std::greater<>());
        scored_candidates.resize(candidate_limit);
    }
    std::sort(scored_candidates.begin(), scored_candidates.end(), std::greater<>());

    std::vector<std::pair<float, int>> final_results;
    final_results.reserve(scored_candidates.size());
    for (const auto& [score, doc_id] : scored_candidates) {
        final_results.emplace_back(distance(vecs.data() + doc_id * dim, vec, dim), doc_id);
    }
    std::sort(final_results.begin(), final_results.end());
    if (final_results.size() > static_cast<size_t>(k)) {
        final_results.resize(k);
    }

    std::vector<int> results;
    results.reserve(final_results.size());
    for (const auto& pair : final_results) {
        results.push_back(pair.second);
    }
    return results;
}

double BM25Filtering::compute_idf(int df) const {
    if (df <= 0 || num_elements <= 0) {
        return 0.0;
    }
    return std::log((num_elements - df + 0.5) / (df + 0.5) + 1.0);
}

BM25Filtering::~BM25Filtering() {
    for (auto* hnsw : hnsws) {
        delete hnsw;
    }
    delete space;
}

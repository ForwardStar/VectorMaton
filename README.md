# VectorMaton

VectorMaton is a C++ index for hybrid approximate nearest-neighbor queries where each vector has an associated string and each query asks for vectors whose strings contain a query substring. A query contains a string, a vector, and an integer `k`; the result is up to `k` approximate nearest neighbors under the substring constraint. The current implementation uses Euclidean distance.

## Table of contents

- [Build the library](#build-the-library)
- [Link in your project](#link-in-your-project)
- [Minimal demo](#minimal-demo)
- [API](#api)
- [Build tests and experiments](#build-tests-and-experiments)
- [Experiment inputs](#experiment-inputs)
- [Run the main experiment](#run-the-main-experiment)
- [Run PostFiltering case study](#run-postfiltering-case-study)
- [External experiments in `source/experiments`](#external-experiments-in-sourceexperiments)
  - [ACORN](#acorn)
  - [Elasticsearch](#elasticsearch)
  - [PostgreSQL/pgvector](#postgresqlpgvector)
- [Reproduce experiment suites](#reproduce-experiment-suites)
- [Minimal native-only experiment example](#minimal-native-only-experiment-example)

## Build the library

Fetch the submodules first. `hnswlib` is required by VectorMaton.

```sh
git submodule update --init --recursive
```

Build the `vectormaton` CMake target:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vectormaton -j
```

This generates:

```text
build/libvectormaton.a
```

The project is developed with C++17, GCC 10.5.0, `-O3`, and OpenMP.

## Link in your project

The simplest CMake integration is to add this repository as a subdirectory and link the exported `vectormaton` target:

```cmake
cmake_minimum_required(VERSION 3.10)
project(my_app CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(/path/to/VectorMaton external/vectormaton)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE vectormaton)
```

The `vectormaton` target publishes the required include directories and OpenMP linkage. In your C++ file, include:

```cpp
#include "vectormaton.h"
```

If you prefer to link against a prebuilt library manually, include `source/` and `third_party/hnswlib`, link `build/libvectormaton.a`, and link OpenMP.

## Minimal demo

```cpp
#include "vectormaton.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    const int dim = 3;
    // Five 3-dimensional vectors flattened row by row.
    std::vector<float> vectors = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f,
    };
    std::vector<std::string> strings = {
        "banana",
        "anana",
        "nana",
        "ana",
        "na",
    };

    VectorMaton index;
    index.set_min_build_threshold(0);
    index.set_vectors(vectors, dim);
    index.set_strings(strings);
    index.build_smart();
    index.set_ef(64);

    float query_vector[3] = {9.0f, 10.0f, 11.0f};
    std::vector<int> result = index.query(query_vector, "ana", 2);

    for (int id : result) {
        std::cout << id << "\n";
    }

    index.save_index("vectormaton_index");

    VectorMaton loaded;
    loaded.set_vectors(vectors, dim);
    loaded.set_strings(strings);
    loaded.load_index("vectormaton_index");

    index.insert({16.0f, 17.0f, 18.0f}, "bandana");
}
```

A fuller executable demo lives in `source/tests/test_vectormaton.cpp` and is built as `vectormaton_test`.

## API

- `set_vectors(const std::vector<float>& vectors, int dimension)`: stores a flattened row-major vector matrix. If you have `n` vectors and each vector has `dimension` coordinates, `vectors` must contain `n * dimension` floats. The coordinate `j` of vector `i` is stored at `vectors[i * dimension + j]`, so the number of indexed vectors is `vectors.size() / dimension`.
- `set_strings(const std::vector<std::string>& strings)`: stores the string attached to each vector. The i-th string corresponds to the i-th vector.
- `set_min_build_threshold(int threshold)`: sets the minimum candidate-set size required before VectorMaton builds an HNSW sub-index. Smaller candidate sets are searched by brute force, while larger ones use the HNSW structure.
- `build_smart()`: builds the space-optimized VectorMaton index using inheritance between suffix-automaton states (index-reuse strategy).
- `build_full()`: builds the VectorMaton index without index-reuse strategy.
- `build_parallel(int cores)`: parallel version of the smart build.
- `set_ef(int ef)`: sets HNSW `ef_search` for built HNSWs.
- `query(const float* vec, const std::string& s, int k)`: returns up to `k` zero-based vector IDs whose strings contain `s` and whose vectors are approximate nearest neighbors of `vec` under the substring constraint.
- `insert(const std::vector<float>& vec, const std::string& str)`: appends one vector/string pair to an existing index.
- `save_index(const char* output_folder)`: writes VectorMaton index files to disk.
- `load_index(const char* input_folder)`: loads VectorMaton index files. Call `set_vectors` and `set_strings` before `load_index` so HNSW distance computations can reference the vector payload.
- `size()`: returns the estimated VectorMaton index size in bytes.
- `vertex_num()`: returns the total number of HNSW vertices stored across built sub-indexes.

## Build tests and experiments

Build all default targets:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Default C++ targets:

- `libvectormaton.a`: the static library target.
- `sa_test`, `queue_test`, `hnsw_test`, `vectormaton_test`: small tests and demos.
- `main_exp`: the main experiment executable from `source/experiments/main_exp.cpp`.
- `postfiltering_case_study`: the PostFiltering case-study executable from `source/experiments/postfiltering_case_study.cpp`.

Run the VectorMaton demo:

```sh
./build/vectormaton_test
```

## Experiment inputs

Each experiment expects aligned string and vector files:

- `strings.txt`: one string per line.
- `vectors.txt`: one whitespace-separated vector per line.
- Query `strings.txt`, query `vectors.txt`, and `k.txt`: one query string, query vector, and integer `k` per line.

Download and generate datasets:

```sh
pip install datasets==3.6.0 numpy==1.26.4 transformers==4.56.0 torch==2.8.0 sentence_transformers==5.6.0
python3 scripts/download_datasets.py
```

Generate queries:

```sh
python3 scripts/generate_queries.py
```

The query generator asks for the dataset, query substring length, number of queries, `k`, and optional data-size limit. Generated query files are written to the working directory.

Analyze query-pattern distributions:

```sh
python3 scripts/plot/pattern_distribution_figure_11.py --all-datasets --output-csv results/pattern_distribution.csv --plot-dir figures/pattern_distribution
```

## Run the main experiment

`main_exp` evaluates the native VectorMaton methods and baseline methods implemented in `source/baselines`.

```sh
./build/main_exp \
  <string_data_file> <vector_data_file> \
  <string_query_file> <vector_query_file> \
  <k_query_file> \
  <OptQuery|PreFiltering|PostFiltering|Hybrid|BM25Filtering|VectorMaton-full|VectorMaton-smart|VectorMaton-parallel>
```

Example:

```sh
./build/main_exp \
  datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt \
  strings.txt vectors.txt k.txt \
  VectorMaton-smart \
  --statistics-file=results/vectormaton_smart.csv
```

Optional flags:

- `--debug`: show debug messages.
- `--data-size=<n>`: use only the first `n` data vectors and strings.
- `--statistics-file=output_statistics.csv`: write recall/time statistics to CSV.
- `--load-index=index_files_folder`: load a previously saved index from disk.
- `--save-index=index_files_folder`: save the built index to disk.
- `--num-threads=<n>`: set the number of threads for `VectorMaton-parallel`.
- `--write-ground-truth=ground_truth.txt`: write exact ground-truth results to a file.
- `--set-min-build-threshold=<n>`: set the minimum candidate-set size required before VectorMaton builds an HNSW sub-index.
- `--insert-percentage=<p>`: reserve the last `p` percent of the dataset for insertion-performance evaluation.
- `--write-output=output.txt`: write query result IDs.

## Run PostFiltering case study

```sh
./build/postfiltering_case_study \
  <string_data_file> <vector_data_file> \
  <string_query_file> <vector_query_file> \
  <k_query_file> \
  PostFiltering \
  --statistics-file=output.csv
```

For each `ef_search`, this evaluates `search_k` ratios (how many candidates are kept before filtering) `0.2`, `0.4`, `0.6`, `0.8`, and `1.0`.

## External experiments in `source/experiments`

The external baseline drivers are:

- `source/experiments/acorn_exp.cpp`: optional ACORN wrapper, built as `acorn_exp`.
- `source/experiments/elasticsearch_exp.py`: Elasticsearch baseline.
- `source/experiments/pgvector_exp.py`: PostgreSQL/pgvector baseline.

### ACORN

ACORN is built from the `third_party/ACORN` submodule, a fork based on FAISS.

```sh
git submodule update --init --recursive third_party/ACORN

cmake -S third_party/ACORN \
      -B third_party/ACORN/build \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DFAISS_ENABLE_C_API=OFF \
      -DBUILD_TESTING=OFF \
      -DFAISS_ENABLE_INSTALL=OFF \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/ACORN/build -j --target faiss

cmake -S . -B build -DENABLE_ACORN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target acorn_exp -j
```

Run ACORN with the same dataset/query files used by other experiments:

```sh
./build/acorn_exp \
  <string_data_file> <vector_data_file> \
  <string_query_file> <vector_query_file> \
  <k_query_file> <ground_truth_file> \
  --M=32 --gamma=12 --M-beta=32 \
  --output=acorn_hnsw_stats.csv
```

The output CSV contains:

```text
ef_search,M,gamma,M_beta,time_us,recall,build_peak_memory_bytes,index_size_bytes
```

### Elasticsearch

Install and start Elasticsearch separately, then run:

```sh
python3 source/experiments/elasticsearch_exp.py \
  <string_data_file> <vector_data_file> \
  <string_query_file> <vector_query_file> \
  <k_query_file> <ground_truth_file> \
  --rebuild
```

The experiments were tested with Elasticsearch 9.3.0. A large heap and at least 150 GB of free disk space are recommended for the full experiment schedule.

### PostgreSQL/pgvector

Install and start PostgreSQL with pgvector separately, then run:

```sh
python3 source/experiments/pgvector_exp.py \
  <string_data_file> <vector_data_file> \
  <string_query_file> <vector_query_file> \
  <k_query_file> <ground_truth_file> \
  --rebuild
```

## Reproduce experiment suites

You should follow the build instructions above and install python dependencies before running the experiment scripts. The scripts assume the current working directory is the repository root.

Firstly download datasets:

```sh
pip install datasets==3.6.0 numpy==1.26.4 transformers==4.56.0 torch==2.8.0 sentence_transformers==5.6.0
python3 scripts/download_datasets.py
```

Then run the scripts:

```sh
sh scripts/run/run-queries.sh
sh scripts/run/run-scalability.sh
sh scripts/run/run-parallel.sh
sh scripts/run/run-sift.sh
sh scripts/run/run-threshold.sh
sh scripts/run/run-postfiltering.sh
sh scripts/run/run-insertion.sh
sh scripts/run/run-long-sequence.sh
```

Plot results:

```sh
python3 scripts/plot/recall_qps_figure_9.py
python3 scripts/plot/recall_qps_figure_10.py
python3 scripts/plot/pattern_distribution_figure_11.py
python3 scripts/plot/plot_index_figure_12.py
python3 scripts/plot/plot_scalability_figure_13.py
python3 scripts/plot/plot_insertion_figure_14.py
python3 scripts/plot/plot_postfiltering_figure_15.py
python3 scripts/plot/plot_wikipedia_figure_16.py
python3 scripts/plot/plot_sift_figure_2.py
python3 scripts/plot/plot_threshold.py
```

Figures are written under `figures/`; raw outputs are written under `results/`.

## Minimal native-only experiment example

The external baselines require separate services or libraries: ACORN/FAISS, Elasticsearch, and PostgreSQL/pgvector. To run a smaller experiment pass with only the in-repository C++ methods, build the default targets and blacklist the external baselines in scripts that support method blacklists. The example below also skips the large protein and code datasets (`swissprot` and `code_search_net`) so the run finishes sooner.

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

pip install datasets==3.6.0 numpy==1.26.4 transformers==4.56.0 torch==2.8.0 sentence_transformers==5.6.0
python3 scripts/download_datasets.py

sh scripts/run/run-queries.sh \
  --blacklist=ACORN-gamma,ACORN-1,pgvector,ElasticSearch \
  --blacklist-dataset=swissprot,code_search_net

sh scripts/run/run-scalability.sh
sh scripts/run/run-sift.sh
sh scripts/run/run-threshold.sh
sh scripts/run/run-postfiltering.sh \
  --blacklist-dataset=swissprot,code_search_net

sh scripts/run/run-insertion.sh \
  --methods "OptQuery BM25Filtering PreFiltering PostFiltering Hybrid VectorMaton-smart" \
  --datasets "spam words mtg arxiv-small"

sh scripts/run/run-long-sequence.sh \
  --blacklist=ACORN-gamma,ACORN-1,pgvector,ElasticSearch
```

This minimal pass intentionally omits `scripts/run/run-parallel.sh`, because that script is designed for the larger `swissprot` and `code_search_net` datasets.
# VectorMaton
An elegant index that supports hybrid queries of ANNs whose associated strings contain a queried substring. Each data in the vector database consists of a string and a vector. Each query contains a string, a vector, and an integer k to return approximated k-nearest neighbors. The query results will contain data that involves the queried string as a substring, and its vector is an approximated k-nearest neighbor of the queried vector under the substring constraint. In this project, we use Euclidean distance as the measure of closeness, but it can be simply extended to support other metrics.

# Compile and run
Parts of the project depend on ``openssl``. Install on Ubuntu:
```sh
sudo apt-get install libssl-dev
```

Fetch the ``hnswlib`` submodule:
```sh
git submodule update --init --recursive
```

We developed and tested this vector database under ``GCC 10.5.0`` with ``O3`` optimization and ``OpenMP``. To compile the codes, simply run:
```sh
mkdir build && cd build
cmake ..
make
```

This will generate executable files ``nsw_test``, ``hnsw_test``, ``sa_test``, ``vectormaton_test``, ``main`` and ``parameter_study``. In particular, ``vectormaton_test`` corresponds to ``source/test_vectormaton.cpp``, which provides a demo on how to use the index.

The ``main`` is our experimental program. Run with:
```sh
./main <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k_query_file> <OptQuery|PreFiltering|PostFiltering|Hybrid|BM25Filtering|VectorMaton-full|VectorMaton-smart|VectorMaton-parallel>
```

It will output recall and time consumption statistics of the corresponding method.

Arguments:
- ``string_data_file``: data strings, one string per line.
- ``vector_data_file``: data vectors, one whitespace-separated vector per line.
- ``string_query_file``: query strings, one string per line.
- ``vector_query_file``: query vectors, one whitespace-separated vector per line.
- ``k_query_file``: query ``k`` values, one integer per line.
- ``method``: one of ``OptQuery``, ``PreFiltering``, ``PostFiltering``, ``Hybrid``, ``VectorMaton-full``, ``VectorMaton-smart`` or ``VectorMaton-parallel``.

Optional flags:
- ``--debug``: show debug messages.
- ``--data-size=<n>``: use only the first ``n`` data vectors and strings.
- ``--statistics-file=output_statistics.csv``: write recall/time statistics to a CSV file. For ef-search based methods, the CSV includes ``ef_search,time_us,recall,exact,average_selectivity``; ``average_selectivity`` is the average fraction of data strings that satisfy the query substring predicate, computed during ExactSearch.
- ``--load-index=index_files_folder``: load a previously saved index from disk.
- ``--save-index=index_files_folder``: save the built index to disk.
- ``--num-threads=<n>``: set the number of threads for ``VectorMaton-parallel``.
- ``--write-ground-truth=ground_truth.txt``: write exact ground-truth results to a file.
- ``--set-min-build-threshold=<n>``: set the minimum candidate-set size required before VectorMaton builds an HNSW sub-index.
- ``--insert-percentage=<p>``: reserve the last ``p`` percent of the dataset for insertion-performance evaluation.

The ``parameter_study`` executable runs one PostFiltering parameter-study benchmark over already generated query files:
```sh
./build/parameter_study <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k_query_file> PostFiltering --statistics-file=output.csv
```
For each ``ef_search``, it evaluates ``search_k`` ratios ``0.2``, ``0.4``, ``0.6``, ``0.8`` and ``1.0``. The helper script ``scripts/run-postfiltering.sh`` repeats the dataset/query schedule used by ``scripts/run-queries.sh`` for this parameter study and writes results to ``results/PostFiltering/parameter_study/<dataset>/<p>.csv``.

# Link VectorMaton in your project

VectorMaton is currently provided as C++ source files rather than as an installed library. The simplest way to use it from another CMake project is to add the source files directly to your target and include this repository plus ``third_party/hnswlib``.

Example CMake configuration:
```cmake
set(VECTORMATON_ROOT /path/to/VectorMaton)

add_executable(my_app
    main.cpp
    ${VECTORMATON_ROOT}/source/sa.cpp
    ${VECTORMATON_ROOT}/source/vectormaton.cpp
)

target_include_directories(my_app PRIVATE
    ${VECTORMATON_ROOT}/source
    ${VECTORMATON_ROOT}/third_party/hnswlib
)

find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)
target_link_libraries(my_app PRIVATE
    OpenMP::OpenMP_CXX
    OpenSSL::SSL
    OpenSSL::Crypto
)

target_compile_features(my_app PRIVATE cxx_std_17)
```

Minimal API example:
```cpp
#include "vectormaton.h"

#include <string>
#include <vector>

int main() {
    const int dim = 3;
    std::vector<float> vectors = {
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
    };
    std::vector<std::string> strings = {
        "banana",
        "bandana",
        "ananas",
    };

    VectorMaton index;
    index.set_vectors(vectors, dim);
    index.set_strings(strings);
    index.set_min_build_threshold(200);
    index.build_smart();
    index.set_ef(64);

    std::vector<float> query_vector = {0.5f, 0.5f, 0.5f};
    std::vector<int> ids = index.query(query_vector.data(), "ana", 2);
}
```

Main APIs:
- ``set_vectors(const std::vector<float>& vectors, int dimension)``: stores row-major vectors. The number of elements is ``vectors.size() / dimension``.
- ``set_strings(const std::vector<std::string>& strings)``: stores the string attached to each vector. The i-th string corresponds to the i-th vector.
- ``set_min_build_threshold(int threshold)``: controls the minimum candidate-set size of a state to build and maintain an HNSW index; if the state has less candidates, queries on this state will do brute-force search rather than HNSW search.
- ``build_smart()``: builds the space-optimized VectorMaton index using inheritance between suffix-automaton states.
- ``build_full()``: builds an HNSW sub-index for every suffix-automaton state.
- ``build_parallel(int cores)``: parallel version of the smart build.
- ``set_ef(int ef)``: sets HNSW ``ef_search`` for all built sub-indexes.
- ``query(const float* vec, const std::string& s, int k)``: returns up to ``k`` zero-based vector IDs whose strings contain ``s`` and whose vectors are approximate nearest neighbors of ``vec`` under the substring constraint.
- ``insert(const std::vector<float>& vec, const std::string& str)``: appends one vector/string pair to an existing index.
- ``save_index(const char* output_folder)`` and ``load_index(const char* input_folder)``: persist and reload VectorMaton index files. Call ``set_vectors`` before ``load_index`` so HNSW distance computations can reference the vector payload.
- ``size()``: returns the estimated VectorMaton index-structure size in bytes.
- ``vertex_num()``: returns the total number of HNSW vertices stored across built sub-indexes.

# Experiments

## Datasets
Since there are no existing vector datasets associated with strings, we include synthetic datasets for experiments. For most datasets, the strings are original natural language texts and the vectors are their embeddings generated by pre-trained language models. For SIFT, the vectors are original SIFT vectors and the strings are synthetic. The datasets include:
- [Spam](https://spamassassin.apache.org/old/publiccorpus): a datasets of spam email and their embeddings; to be specific, each data consists of an email title as its string and the email content embedding as its vector;
- [Words](https://huggingface.co/datasets/efarrall/word_embeddings): a dataset of words and their embeddings; to be specific, each data consists of a word of letters as its string and the embedding of this word as its vector;
- [MTG](https://huggingface.co/datasets/TrevorJS/mtg-scryfall-cropped-art-embeddings-siglip-so400m-patch14-384): a dataset of images and their embeddings; to be specific, each data consists of an image description as its string and image embedding as its vector;
- [CodeSearchNet](https://huggingface.co/datasets/irds/codesearchnet): a dataset of code snippets and their embeddings; to be specific, each data consists of a function name as its string and a code embedding vector (generated by CodeBERT) as its vector;
- [SwissProt](https://huggingface.co/datasets/khairi/uniprot-swissprot): a manually curated section of the UniProt protein sequence database; to be specific, each data consists of a protein sequence and its structural embedding vector (generated by ProtBERT) as its vector;
- [ArXiv](https://huggingface.co/datasets/Qdrant/arxiv-titles-instructorxl-embeddings): a dataset of paper titles and their embeddings; to be specific, each data consists of a paper title as its string and a text embedding vector (generated by InstructorXL) as its vector;
- [ArXiv-Small](https://huggingface.co/datasets/malteos/aspect-paper-embeddings): a dataset of paper titles and their embeddings; to be specific, each data consists of a paper title as its string and a text embedding vector (generated by all-mpnet-base-v2) as its vector;
- [SIFT](http://corpus-texmex.irisa.fr/): a classic ANN benchmark dataset of image descriptors; to be specific, each data consists of a synthetic lowercase string (generated by this repo for substring filtering experiments) and a SIFT base vector as its vector.

We provide a Python script ``scripts/download_datasets.py`` to download and generate aforementioned datasets. The generated datasets are stored in the ``datasets/`` folder. Each dataset contains a file ``vectors.txt`` and a file ``strings.txt``, where the i-th line of ``vectors.txt`` and the i-th line of ``strings.txt`` correspond to the vector and string of the i-th data, respectively. To execute the script, please first install the required ``datasets``, ``numpy``, ``transformers`` and ``torch`` packages via:
```sh
pip install datasets==3.6.0 numpy==1.26.4 transformers==4.56.0 torch==2.8.0
```

Note that: at the time of our development, some datasets are not compatible with ``datasets`` library version later than 3.6.0, so please make sure to install the exact version above.

Then run:
```sh
python3 scripts/download_datasets.py
```

Note that the script may take hours to finish, as it needs to download the dataset and compute the embeddings. The script does not support checkpointing for single dataset, so if it is interrupted, you need to delete that dataset folder and restart it. The script will skip the dataset generation if the dataset already exists.

## Queries
Generate queries by:
```sh
python3 generate_queries.py
```
and input the selected datasets, queried string length, etc. The queried strings are randomly sampled from the substrings of the original dataset. The queried vectors are randomly sampled from the original dataset. Generated quries are written into ``strings.txt``, ``vectors.txt`` and ``k.txt``.

Analyze the generated query pattern distribution by:
```sh
python3 scripts/pattern_distribution.py --all-datasets --output-csv results/pattern_distribution.csv --plot-dir figures/pattern_distribution
```
This reports query repetition, dataset substring frequency skew, query selectivity, and overlap among the data strings matched by different query patterns. These statistics help explain when pre-filtering, post-filtering, and VectorMaton see easier or harder substring predicates.

## Running example
Following is a minimal running example.

Firstly, compile the project:
```sh
> git submodule update --init --recursive
> mkdir build && cd build
> cmake ..
> make
```

Then download datasets by:
```sh
> python3 scripts/download_datasets.py
```

Then generate query data:
```sh
> python3 generate_queries.py
Available datasets:
0: arxiv-small
1: swissprot
2: code_search_net
3: arxiv
Enter the index of the dataset to use: 0
Enter the desired string length for queries: 3
Enter the number of queries to generate: 1000
Enter value k for k-NN search: 10
Enter the number of elements you want to select from the dataset (-1 for all): -1
```

Finally, run PreFiltering on the query data:
```sh
> ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt PreFiltering
```
Or run BM25Filtering instead:
```sh
> ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt BM25Filtering
```
## Optional ACORN baseline

This repository can also build a standalone ACORN baseline wrapper, ``test_acorn``. ACORN is built from the fork at [ForwardStar/ACORN](https://github.com/ForwardStar/ACORN), which is based on FAISS.

First fetch the ACORN submodule:
```sh
git submodule update --init --recursive third_party/ACORN
```

Build ACORN/FAISS locally. The benchmark links against the local ``faiss`` library target; it does not require installing FAISS system-wide.
```sh
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
```

Then enable and build the VectorMaton ACORN wrapper:
```sh
cmake -S . -B build -DENABLE_ACORN=ON
cmake --build build --target test_acorn
```

Run ACORN with the same dataset/query files used by the other experiments:
```sh
./build/test_acorn \
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

During ACORN indexing, the wrapper also builds a ``GeneralizedSuffixAutomaton`` over the data strings. Query-time ACORN filter bitmaps are built from GSA match ID lists instead of scanning every data string. A data point is eligible iff its string contains the query string as a substring. The metadata vector passed to ACORN is currently a placeholder because this benchmark's predicates are substring predicates rather than categorical attributes.

## All external baselines
The scripts ``test_elasticsearch.py`` and ``test_pgvector.py`` evaluate external baseline systems. The optional ``test_acorn`` executable evaluates ACORN. Elasticsearch and PostgreSQL/pgvector must already be running locally. ACORN is linked as a local FAISS-based library as described above. We test it with ElasticSearch 9.3.0 and Postgresql 17.6.

For ElasticSearch, we download it from [the official website](https://www.elastic.co/downloads/elasticsearch) and unpack it. Configure the JVM heap memory to 128GB by writing:
```
-Xms128g
-Xmx128g
```

to ``/path/to/elasticsearch-9.3.0/config/jvm.options.d/heap.options``. After that, launch it simply by:
```sh
/path/to/elasticsearch-9.3.0/bin/elasticsearch
```

And reset the password to ``123456``:
```sh
/path/to/elasticsearch-9.3.0/bin/elasticsearch-reset-password -u elastic -i
```

Note that ElasticSearch requires **at least 150GB free disk space** during the whole experimental process. Failure the meet the condition will result in connection lost.

For PostgreSQL, we install it by Anaconda:
```sh
conda create -n pg_env -c conda-forge postgresql
conda activate pg_env
```

Initialize its data folder:
```sh
rm -rf ./pgdata
initdb -D ./pgdata
```

Then launch it by:
```sh
pg_ctl -D ./pgdata -l logfile start
```

## Reproduce the minimal experimental results
**TODO**

## Reproduce the full experimental results
**TODO: update the guide**

Firstly, fetch the submodules and compile VectorMaton (and resolve dependency issues if needed):
```sh
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j
cd ..
```

Also, follow the above instructions to prepare for external baselines ElasticSearch and PostgreSQL.

Then prepare dataset (may need hours to finish):
```sh
python3 scripts/download_datasets.py
```

Finally, run all experiments:
```sh
sh scripts/run-queries.sh
sh scripts/run-scalability.sh
sh scripts/run-parallel.sh
sh scripts/run-sift.sh
sh scripts/run-threshold.sh
sh scripts/run-postfiltering.sh
```

Plot by:
```sh
python3 scripts/recall_qps.py
python3 scripts/recall_qps_selected.py
python3 scripts/plot_memory_consumption.py
python3 scripts/memory_and_time.py
python3 scripts/plot_scalability.py
python3 scripts/plot_threshold.py
python3 scripts/plot_sift.py
python3 scripts/plot_postfiltering.py
```

You will see results in the ``figures`` folder.
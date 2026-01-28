if [ ! -d "results" ]; then
    mkdir results
fi
if [ ! -d "results/OptQuery" ]; then
    mkdir results/OptQuery
fi
if [ ! -d "results/PreFiltering" ]; then
    mkdir results/PreFiltering
fi
if [ ! -d "results/PostFiltering" ]; then
    mkdir results/PostFiltering
fi
if [ ! -d "results/pgvector" ]; then
    mkdir results/pgvector
fi
if [ ! -d "results/VectorMaton" ]; then
    mkdir results/VectorMaton
fi

# Run arxiv-small
for s in 2 3 4
do
    python3 generate_queries.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/arxiv-small" ]; then
        mkdir results/PreFiltering/arxiv-small
    fi
    ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/arxiv-small/$s
    # OptQuery
    # if [ ! -d "results/OptQuery/arxiv-small" ]; then
    #     mkdir results/OptQuery/arxiv-small
    # fi
    # ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt OptQuery > results/OptQuery/arxiv-small/$s
    # PostFiltering
    if [ ! -d "results/PostFiltering/arxiv-small" ]; then
        mkdir results/PostFiltering/arxiv-small
    fi
    ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/arxiv-small/$s --statistics-file=results/PostFiltering/arxiv-small/$s.csv
    # pgvector
    if [ ! -d "results/pgvector/arxiv-small" ]; then
        mkdir results/pgvector/arxiv-small
    fi
    python3 test_pgvector.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt
    mv pgvector_hnsw_stats.csv results/pgvector/arxiv-small/$s.csv
    # VectorMaton
    if [ ! -d "results/VectorMaton/arxiv-small" ]; then
        mkdir results/VectorMaton/arxiv-small
    fi
    ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/arxiv-small/$s --statistics-file=results/VectorMaton/arxiv-small/$s.csv
done

# Run swissprot
for s in 2 3 4
do
    python3 generate_queries.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/swissprot" ]; then
        mkdir results/PreFiltering/swissprot
    fi
    ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/swissprot/$s
    # OptQuery
    # if [ ! -d "results/OptQuery/swissprot" ]; then
    #     mkdir results/OptQuery/swissprot
    # fi
    # ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt OptQuery > results/OptQuery/swissprot/$s
    # PostFiltering
    if [ ! -d "results/PostFiltering/swissprot" ]; then
        mkdir results/PostFiltering/swissprot
    fi
    ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/swissprot/$s --statistics-file=results/PostFiltering/swissprot/$s.csv
    # pgvector
    if [ ! -d "results/pgvector/swissprot" ]; then
        mkdir results/pgvector/swissprot
    fi
    python3 test_pgvector.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt
    mv pgvector_hnsw_stats.csv results/pgvector/swissprot/$s.csv
    # VectorMaton
    if [ ! -d "results/VectorMaton/swissprot" ]; then
        mkdir results/VectorMaton/swissprot
    fi
    ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/swissprot/$s --statistics-file=results/VectorMaton/swissprot/$s.csv
done

# Run code_search_net
for s in 2 3 4
do
    python3 generate_queries.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/code_search_net" ]; then
        mkdir results/PreFiltering/code_search_net
    fi
    ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/code_search_net/$s
    # OptQuery
    # if [ ! -d "results/OptQuery/code_search_net" ]; then
    #     mkdir results/OptQuery/code_search_net
    # fi
    # ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt OptQuery > results/OptQuery/code_search_net/$s
    # PostFiltering
    if [ ! -d "results/PostFiltering/code_search_net" ]; then
        mkdir results/PostFiltering/code_search_net
    fi
    ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/code_search_net/$s --statistics-file=results/PostFiltering/code_search_net/$s.csv
    # pgvector
    if [ ! -d "results/pgvector/code_search_net" ]; then
        mkdir results/pgvector/code_search_net
    fi
    python3 test_pgvector.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt
    mv pgvector_hnsw_stats.csv results/pgvector/code_search_net/$s.csv
    # VectorMaton
    if [ ! -d "results/VectorMaton/code_search_net" ]; then
        mkdir results/VectorMaton/code_search_net
    fi
    ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/code_search_net/$s --statistics-file=results/VectorMaton/code_search_net/$s.csv
done
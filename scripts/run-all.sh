#!/bin/bash
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

# Run spam
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/spam/strings.txt datasets/spam/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/spam" ]; then
        mkdir results/PreFiltering/spam
    fi
    ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/spam/$s &
    # OptQuery
    if [ ! -d "results/OptQuery/spam" ]; then
        mkdir results/OptQuery/spam
    fi
    ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings.txt vectors.txt k.txt OptQuery --statistics-file=results/OptQuery/spam/$s.csv > results/OptQuery/spam/$s &
    # PostFiltering
    if [ ! -d "results/PostFiltering/spam" ]; then
        mkdir results/PostFiltering/spam
    fi
    ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/spam/$s --statistics-file=results/PostFiltering/spam/$s.csv &
    # VectorMaton
    if [ ! -d "results/VectorMaton/spam" ]; then
        mkdir results/VectorMaton/spam
    fi
    ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/spam/$s --statistics-file=results/VectorMaton/spam/$s.csv &
    wait
    # pgvector
    if [ ! -d "results/pgvector/spam" ]; then
        mkdir results/pgvector/spam
    fi
    python3 test_pgvector.py datasets/spam/strings.txt datasets/spam/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/spam/$s.csv
done

# # Run Words
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/words/strings.txt datasets/words/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/words" ]; then
        mkdir results/PreFiltering/words
    fi
    ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/words/$s &
    # OptQuery
    if [ ! -d "results/OptQuery/words" ]; then
        mkdir results/OptQuery/words
    fi
    ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings.txt vectors.txt k.txt OptQuery --statistics-file=results/OptQuery/words/$s.csv > results/OptQuery/words/$s &
    # PostFiltering
    if [ ! -d "results/PostFiltering/words" ]; then
        mkdir results/PostFiltering/words
    fi
    ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/words/$s --statistics-file=results/PostFiltering/words/$s.csv &
    # VectorMaton
    if [ ! -d "results/VectorMaton/words" ]; then
        mkdir results/VectorMaton/words
    fi
    ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/words/$s --statistics-file=results/VectorMaton/words/$s.csv &
    wait
    # pgvector
    if [ ! -d "results/pgvector/words" ]; then
        mkdir results/pgvector/words
    fi
    python3 test_pgvector.py datasets/words/strings.txt datasets/words/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/words/$s.csv
done

# # Run mtg
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/mtg/strings.txt datasets/mtg/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/mtg" ]; then
        mkdir results/PreFiltering/mtg
    fi
    ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/mtg/$s &
    # OptQuery
    # if [ ! -d "results/OptQuery/mtg" ]; then
    #     mkdir results/OptQuery/mtg
    # fi
    # ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings.txt vectors.txt k.txt OptQuery --statistics-file=results/OptQuery/mtg/$s.csv > results/OptQuery/mtg/$s &
    # PostFiltering
    if [ ! -d "results/PostFiltering/mtg" ]; then
        mkdir results/PostFiltering/mtg
    fi
    ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/mtg/$s --statistics-file=results/PostFiltering/mtg/$s.csv &
    # VectorMaton
    if [ ! -d "results/VectorMaton/mtg" ]; then
        mkdir results/VectorMaton/mtg
    fi
    ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/mtg/$s --statistics-file=results/VectorMaton/mtg/$s.csv &
    wait
    # pgvector
    if [ ! -d "results/pgvector/mtg" ]; then
        mkdir results/pgvector/mtg
    fi
    python3 test_pgvector.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/mtg/$s.csv
done

# # Run arxiv-small
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/arxiv-small" ]; then
        mkdir results/PreFiltering/arxiv-small
    fi
    ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/arxiv-small/$s &
    # OptQuery
    # if [ ! -d "results/OptQuery/arxiv-small" ]; then
    #     mkdir results/OptQuery/arxiv-small
    # fi
    # ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt OptQuery --statistics-file=results/OptQuery/arxiv-small/$s.csv > results/OptQuery/arxiv-small/$s &
    # PostFiltering
    if [ ! -d "results/PostFiltering/arxiv-small" ]; then
        mkdir results/PostFiltering/arxiv-small
    fi
    ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/arxiv-small/$s --statistics-file=results/PostFiltering/arxiv-small/$s.csv &
    # VectorMaton
    if [ ! -d "results/VectorMaton/arxiv-small" ]; then
        mkdir results/VectorMaton/arxiv-small
    fi
    ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/arxiv-small/$s --statistics-file=results/VectorMaton/arxiv-small/$s.csv &
    wait
    # pgvector
    if [ ! -d "results/pgvector/arxiv-small" ]; then
        mkdir results/pgvector/arxiv-small
    fi
    python3 test_pgvector.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/arxiv-small/$s.csv
done

# # Run swissprot
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/swissprot" ]; then
        mkdir results/PreFiltering/swissprot
    fi
    ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/swissprot/$s &
    # OptQuery
    # if [ ! -d "results/OptQuery/swissprot" ]; then
    #     mkdir results/OptQuery/swissprot
    # fi
    # ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt OptQuery --statistics-file=results/OptQuery/swissprot/$s.csv > results/OptQuery/swissprot/$s &
    # PostFiltering
    if [ ! -d "results/PostFiltering/swissprot" ]; then
        mkdir results/PostFiltering/swissprot
    fi
    ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/swissprot/$s --statistics-file=results/PostFiltering/swissprot/$s.csv &
    # VectorMaton
    if [ ! -d "results/VectorMaton/swissprot" ]; then
        mkdir results/VectorMaton/swissprot
    fi
    ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/swissprot/$s --statistics-file=results/VectorMaton/swissprot/$s.csv &
    wait
    # pgvector
    if [ ! -d "results/pgvector/swissprot" ]; then
        mkdir results/pgvector/swissprot
    fi
    python3 test_pgvector.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/swissprot/$s.csv
done

# Run code_search_net
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt $s 1000 10 -1
    # PreFiltering
    if [ ! -d "results/PreFiltering/code_search_net" ]; then
        mkdir results/PreFiltering/code_search_net
    fi
    ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/code_search_net/$s &
    # OptQuery
    # if [ ! -d "results/OptQuery/code_search_net" ]; then
    #     mkdir results/OptQuery/code_search_net
    # fi
    # ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt OptQuery --statistics-file=results/OptQuery/code_search_net/$s.csv > results/OptQuery/code_search_net/$s &
    # PostFiltering
    if [ ! -d "results/PostFiltering/code_search_net" ]; then
        mkdir results/PostFiltering/code_search_net
    fi
    ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt PostFiltering > results/PostFiltering/code_search_net/$s --statistics-file=results/PostFiltering/code_search_net/$s.csv &
    # VectorMaton
    if [ ! -d "results/VectorMaton/code_search_net" ]; then
        mkdir results/VectorMaton/code_search_net
    fi
    ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > results/VectorMaton/code_search_net/$s --statistics-file=results/VectorMaton/code_search_net/$s.csv &
    wait
    # pgvector
    if [ ! -d "results/pgvector/code_search_net" ]; then
        mkdir results/pgvector/code_search_net
    fi
    python3 test_pgvector.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings.txt vectors.txt k.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/code_search_net/$s.csv
done
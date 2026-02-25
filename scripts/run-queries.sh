#!/bin/bash
set -eu
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

BLACKLIST_RAW=""
while [ $# -gt 0 ]; do
    case "$1" in
        --blacklist)
            if [ $# -lt 2 ]; then
                echo "Missing value for --blacklist" >&2
                exit 1
            fi
            BLACKLIST_RAW="$2"
            shift 2
            ;;
        --blacklist=*)
            BLACKLIST_RAW="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

BLACKLIST_NORM="$(echo ",${BLACKLIST_RAW}," | tr '[:upper:]' '[:lower:]' | tr -d ' ')"

is_blacklisted() {
    local alg
    alg="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "$BLACKLIST_NORM" in
        *,"$alg",*) return 0 ;;
        *) return 1 ;;
    esac
}

should_run() {
    ! is_blacklisted "$1"
}

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
if [ ! -d "results/ElasticSearch" ]; then
    mkdir results/ElasticSearch
fi
if [ ! -d "results/VectorMaton" ]; then
    mkdir results/VectorMaton
fi

# Run spam
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/spam/strings.txt datasets/spam/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/spam" ]; then
            mkdir results/PreFiltering/spam
        fi
        ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/spam/$s &
    fi
    # OptQuery
    if should_run "OptQuery"; then
        if [ ! -d "results/OptQuery/spam" ]; then
            mkdir results/OptQuery/spam
        fi
        ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt OptQuery --statistics-file=results/OptQuery/spam/$s.csv > results/OptQuery/spam/$s &
    fi
    # PostFiltering
    if should_run "PostFiltering"; then
        if [ ! -d "results/PostFiltering/spam" ]; then
            mkdir results/PostFiltering/spam
        fi
        ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering > results/PostFiltering/spam/$s --statistics-file=results/PostFiltering/spam/$s.csv &
    fi
    # VectorMaton
    if should_run "VectorMaton"; then
        if [ ! -d "results/VectorMaton/spam" ]; then
            mkdir results/VectorMaton/spam
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart > results/VectorMaton/spam/$s --statistics-file=results/VectorMaton/spam/$s.csv --save-index=results/VectorMaton/spam/index &
        else
            ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart > results/VectorMaton/spam/$s --statistics-file=results/VectorMaton/spam/$s.csv --load-index=results/VectorMaton/spam/index &
        fi
    fi
    wait
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/spam" ]; then
            mkdir results/pgvector/spam
        fi
        python3 test_pgvector.py datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/spam/$s.csv
    fi
    # Elasticsearch
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/spam" ]; then
            mkdir results/ElasticSearch/spam
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/spam/$s.csv
        else
            python3 test_elasticsearch.py datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/spam/$s.csv
        fi
    fi
done

# # Run Words
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/words/strings.txt datasets/words/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/words" ]; then
            mkdir results/PreFiltering/words
        fi
        ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/words/$s &
    fi
    # OptQuery
    if should_run "OptQuery"; then
        if [ ! -d "results/OptQuery/words" ]; then
            mkdir results/OptQuery/words
        fi
        ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt OptQuery --statistics-file=results/OptQuery/words/$s.csv > results/OptQuery/words/$s &
    fi
    # PostFiltering
    if should_run "PostFiltering"; then
        if [ ! -d "results/PostFiltering/words" ]; then
            mkdir results/PostFiltering/words
        fi
        ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering > results/PostFiltering/words/$s --statistics-file=results/PostFiltering/words/$s.csv &
    fi
    # VectorMaton
    if should_run "VectorMaton"; then
        if [ ! -d "results/VectorMaton/words" ]; then
            mkdir results/VectorMaton/words
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --save-index=results/VectorMaton/words/index > results/VectorMaton/words/$s --statistics-file=results/VectorMaton/words/$s.csv &
        else
            ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --load-index=results/VectorMaton/words/index > results/VectorMaton/words/$s --statistics-file=results/VectorMaton/words/$s.csv &
        fi
    fi
    wait
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/words" ]; then
            mkdir results/pgvector/words
        fi
        python3 test_pgvector.py datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/words/$s.csv
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/words" ]; then
            mkdir results/ElasticSearch/words
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/words/$s.csv
        else
            python3 test_elasticsearch.py datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/words/$s.csv
        fi
    fi
done

# # Run mtg
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/mtg/strings.txt datasets/mtg/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/mtg" ]; then
            mkdir results/PreFiltering/mtg
        fi
        ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/mtg/$s &
    fi
    # OptQuery
    # if [ ! -d "results/OptQuery/mtg" ]; then
    #     mkdir results/OptQuery/mtg
    # fi
    # ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt OptQuery --statistics-file=results/OptQuery/mtg/$s.csv > results/OptQuery/mtg/$s &
    # PostFiltering
    if should_run "PostFiltering"; then
        if [ ! -d "results/PostFiltering/mtg" ]; then
            mkdir results/PostFiltering/mtg
        fi
        ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering > results/PostFiltering/mtg/$s --statistics-file=results/PostFiltering/mtg/$s.csv &
    fi
    # VectorMaton
    if should_run "VectorMaton"; then
        if [ ! -d "results/VectorMaton/mtg" ]; then
            mkdir results/VectorMaton/mtg
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --save-index=results/VectorMaton/mtg/index > results/VectorMaton/mtg/$s --statistics-file=results/VectorMaton/mtg/$s.csv &
        else
            ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --load-index=results/VectorMaton/mtg/index > results/VectorMaton/mtg/$s --statistics-file=results/VectorMaton/mtg/$s.csv &
        fi
    fi
    wait
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/mtg" ]; then
            mkdir results/pgvector/mtg
        fi
        python3 test_pgvector.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/mtg/$s.csv
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/mtg" ]; then
            mkdir results/ElasticSearch/mtg
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/mtg/$s.csv
        else
            python3 test_elasticsearch.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/mtg/$s.csv
        fi
    fi
done

# # Run arxiv-small
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/arxiv-small" ]; then
            mkdir results/PreFiltering/arxiv-small
        fi
        ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/arxiv-small/$s &
    fi
    # OptQuery
    # if [ ! -d "results/OptQuery/arxiv-small" ]; then
    #     mkdir results/OptQuery/arxiv-small
    # fi
    # ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt OptQuery --statistics-file=results/OptQuery/arxiv-small/$s.csv > results/OptQuery/arxiv-small/$s &
    # PostFiltering
    if should_run "PostFiltering"; then
        if [ ! -d "results/PostFiltering/arxiv-small" ]; then
            mkdir results/PostFiltering/arxiv-small
        fi
        ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering > results/PostFiltering/arxiv-small/$s --statistics-file=results/PostFiltering/arxiv-small/$s.csv &
    fi
    # VectorMaton
    if should_run "VectorMaton"; then
        if [ ! -d "results/VectorMaton/arxiv-small" ]; then
            mkdir results/VectorMaton/arxiv-small
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --save-index=results/VectorMaton/arxiv-small/index > results/VectorMaton/arxiv-small/$s --statistics-file=results/VectorMaton/arxiv-small/$s.csv &
        else
            ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --load-index=results/VectorMaton/arxiv-small/index > results/VectorMaton/arxiv-small/$s --statistics-file=results/VectorMaton/arxiv-small/$s.csv &
        fi
    fi
    wait
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/arxiv-small" ]; then
            mkdir results/pgvector/arxiv-small
        fi
        python3 test_pgvector.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/arxiv-small/$s.csv
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/arxiv-small" ]; then
            mkdir results/ElasticSearch/arxiv-small
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/arxiv-small/$s.csv
        else
            python3 test_elasticsearch.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/arxiv-small/$s.csv
        fi
    fi
done

# # Run swissprot
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/swissprot" ]; then
            mkdir results/PreFiltering/swissprot
        fi
        ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/swissprot/$s &
    fi
    # OptQuery
    # if [ ! -d "results/OptQuery/swissprot" ]; then
    #     mkdir results/OptQuery/swissprot
    # fi
    # ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt OptQuery --statistics-file=results/OptQuery/swissprot/$s.csv > results/OptQuery/swissprot/$s &
    # PostFiltering
    if should_run "PostFiltering"; then
        if [ ! -d "results/PostFiltering/swissprot" ]; then
            mkdir results/PostFiltering/swissprot
        fi
        ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering > results/PostFiltering/swissprot/$s --statistics-file=results/PostFiltering/swissprot/$s.csv &
    fi
    # VectorMaton
    if should_run "VectorMaton"; then
        if [ ! -d "results/VectorMaton/swissprot" ]; then
            mkdir results/VectorMaton/swissprot
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --save-index=results/VectorMaton/swissprot/index > results/VectorMaton/swissprot/$s --statistics-file=results/VectorMaton/swissprot/$s.csv &
        else
            ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --load-index=results/VectorMaton/swissprot/index > results/VectorMaton/swissprot/$s --statistics-file=results/VectorMaton/swissprot/$s.csv &
        fi
    fi
    wait
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/swissprot" ]; then
            mkdir results/pgvector/swissprot
        fi
        python3 test_pgvector.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/swissprot/$s.csv
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/swissprot" ]; then
            mkdir results/ElasticSearch/swissprot
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/swissprot/$s.csv
        else
            python3 test_elasticsearch.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/swissprot/$s.csv
        fi
    fi
done

# Run code_search_net
for s in 2 3 4
do
    python3 scripts/generate_queries.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/code_search_net" ]; then
            mkdir results/PreFiltering/code_search_net
        fi
        ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt > results/PreFiltering/code_search_net/$s &
    fi
    # OptQuery
    # if [ ! -d "results/OptQuery/code_search_net" ]; then
    #     mkdir results/OptQuery/code_search_net
    # fi
    # ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt OptQuery --statistics-file=results/OptQuery/code_search_net/$s.csv > results/OptQuery/code_search_net/$s &
    # PostFiltering
    if should_run "PostFiltering"; then
        if [ ! -d "results/PostFiltering/code_search_net" ]; then
            mkdir results/PostFiltering/code_search_net
        fi
        ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering > results/PostFiltering/code_search_net/$s --statistics-file=results/PostFiltering/code_search_net/$s.csv &
    fi
    # VectorMaton
    if should_run "VectorMaton"; then
        if [ ! -d "results/VectorMaton/code_search_net" ]; then
            mkdir results/VectorMaton/code_search_net
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --save-index=results/VectorMaton/code_search_net/index > results/VectorMaton/code_search_net/$s --statistics-file=results/VectorMaton/code_search_net/$s.csv &
        else
            ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt VectorMaton-smart --load-index=results/VectorMaton/code_search_net/index > results/VectorMaton/code_search_net/$s --statistics-file=results/VectorMaton/code_search_net/$s.csv &
        fi
    fi
    wait
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/code_search_net" ]; then
            mkdir results/pgvector/code_search_net
        fi
        python3 test_pgvector.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/code_search_net/$s.csv
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/code_search_net" ]; then
            mkdir results/ElasticSearch/code_search_net
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/code_search_net/$s.csv
        else
            python3 test_elasticsearch.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/code_search_net/$s.csv
        fi
    fi
done

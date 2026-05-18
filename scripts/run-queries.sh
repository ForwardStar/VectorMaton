#!/bin/bash
set -eu
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

BLACKLIST_RAW=""
BLACKLIST_DATASET_RAW=""
ELASTICSEARCH_PID=""
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
        --blacklist-dataset)
            if [ $# -lt 2 ]; then
                echo "Missing value for --blacklist-dataset" >&2
                exit 1
            fi
            BLACKLIST_DATASET_RAW="$2"
            shift 2
            ;;
        --blacklist-dataset=*)
            BLACKLIST_DATASET_RAW="${1#*=}"
            shift
            ;;
        --elasticsearch-pid)
            if [ $# -lt 2 ]; then
                echo "Missing value for --elasticsearch-pid" >&2
                exit 1
            fi
            ELASTICSEARCH_PID="$2"
            shift 2
            ;;
        --elasticsearch-pid=*)
            ELASTICSEARCH_PID="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

BLACKLIST_NORM="$(echo ",${BLACKLIST_RAW}," | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
BLACKLIST_DATASET_NORM="$(echo ",${BLACKLIST_DATASET_RAW}," | tr '[:upper:]' '[:lower:]' | tr -d ' ')"

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

is_dataset_blacklisted() {
    local dataset
    dataset="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "$BLACKLIST_DATASET_NORM" in
        *,"$dataset",*) return 0 ;;
        *) return 1 ;;
    esac
}

should_run_dataset() {
    ! is_dataset_blacklisted "$1"
}

ELASTICSEARCH_PID_ARG=""
if [ -z "$ELASTICSEARCH_PID" ] && should_run "ElasticSearch"; then
    ELASTICSEARCH_PIDS="$(pgrep -af org.elasticsearch.bootstrap.Elasticsearch | awk '{print $1}' || true)"
    ELASTICSEARCH_PID_COUNT="$(printf '%s\n' "$ELASTICSEARCH_PIDS" | sed '/^$/d' | wc -l)"
    if [ "$ELASTICSEARCH_PID_COUNT" -eq 1 ]; then
        ELASTICSEARCH_PID="$ELASTICSEARCH_PIDS"
        echo "Detected Elasticsearch PID: $ELASTICSEARCH_PID"
    elif [ "$ELASTICSEARCH_PID_COUNT" -gt 1 ]; then
        echo "Multiple Elasticsearch processes found; pass --elasticsearch-pid explicitly." >&2
        printf '%s\n' "$ELASTICSEARCH_PIDS" >&2
    else
        echo "No Elasticsearch process found with pgrep; process memory stats may be unavailable." >&2
    fi
fi

if [ -n "$ELASTICSEARCH_PID" ]; then
    ELASTICSEARCH_PID_ARG="--process-pid $ELASTICSEARCH_PID"
fi

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
if [ ! -d "results/Hybrid" ]; then
    mkdir results/Hybrid
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

if [ ! -d "results/ACORN-gamma" ]; then
    mkdir results/ACORN-gamma
fi
if [ ! -d "results/ACORN-1" ]; then
    mkdir results/ACORN-1
fi

run_acorn_variants() {
    local dataset="$1"
    local strings_file="$2"
    local vectors_file="$3"
    local pattern_length="$4"

    if should_run "ACORN-gamma"; then
        if [ ! -d "results/ACORN-gamma/${dataset}" ]; then
            mkdir "results/ACORN-gamma/${dataset}"
        fi
        OMP_NUM_THREADS=1 ./build/test_acorn \
            "${strings_file}" "${vectors_file}" \
            strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt \
            --M=32 --gamma=12 --M-beta=32 \
            --output="results/ACORN-gamma/${dataset}/${pattern_length}.csv" \
            > "results/ACORN-gamma/${dataset}/${pattern_length}"
    fi

    if should_run "ACORN-1"; then
        if [ ! -d "results/ACORN-1/${dataset}" ]; then
            mkdir "results/ACORN-1/${dataset}"
        fi
        OMP_NUM_THREADS=1 ./build/test_acorn \
            "${strings_file}" "${vectors_file}" \
            strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt \
            --M=32 --gamma=1 --M-beta=64 \
            --output="results/ACORN-1/${dataset}/${pattern_length}.csv" \
            > "results/ACORN-1/${dataset}/${pattern_length}"
    fi
}
# Run spam
if should_run_dataset "spam"; then
for s in 2 3 4 5 6 7
do
    python3 scripts/generate_queries.py datasets/spam/strings.txt datasets/spam/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/spam" ]; then
            mkdir results/PreFiltering/spam
        fi
        ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt --statistics-file=results/PreFiltering/spam/$s.csv > results/PreFiltering/spam/$s &
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
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --save-index=results/PostFiltering/spam/index --statistics-file=results/PostFiltering/spam/$s.csv > results/PostFiltering/spam/$s &
        else
            ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --load-index=results/PostFiltering/spam/index --statistics-file=results/PostFiltering/spam/$s.csv > results/PostFiltering/spam/$s &
        fi
    fi
    # Hybrid
    if should_run "Hybrid"; then
        if [ ! -d "results/Hybrid/spam" ]; then
            mkdir results/Hybrid/spam
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --save-index=results/Hybrid/spam/index --statistics-file=results/Hybrid/spam/$s.csv > results/Hybrid/spam/$s &
        else
            ./build/main datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --load-index=results/Hybrid/spam/index --statistics-file=results/Hybrid/spam/$s.csv > results/Hybrid/spam/$s &
        fi
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
    run_acorn_variants "spam" datasets/spam/strings.txt datasets/spam/vectors.txt "$s"
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/spam" ]; then
            mkdir results/pgvector/spam
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_pgvector.py datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv pgvector_hnsw_stats.csv results/pgvector/spam/$s.csv
        else
            python3 test_pgvector.py datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/spam/$s.csv
        fi
    fi
    # Elasticsearch
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/spam" ]; then
            mkdir results/ElasticSearch/spam
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/spam/$s.csv
        else
            python3 test_elasticsearch.py datasets/spam/strings.txt datasets/spam/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/spam/$s.csv
        fi
    fi
done
fi

# # Run Words
if should_run_dataset "words"; then
for s in 2 3 4 5 6 7
do
    python3 scripts/generate_queries.py datasets/words/strings.txt datasets/words/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/words" ]; then
            mkdir results/PreFiltering/words
        fi
        ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt --statistics-file=results/PreFiltering/words/$s.csv > results/PreFiltering/words/$s &
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
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --save-index=results/PostFiltering/words/index --statistics-file=results/PostFiltering/words/$s.csv > results/PostFiltering/words/$s &
        else
            ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --load-index=results/PostFiltering/words/index --statistics-file=results/PostFiltering/words/$s.csv > results/PostFiltering/words/$s &
        fi
    fi
    # Hybrid
    if should_run "Hybrid"; then
        if [ ! -d "results/Hybrid/words" ]; then
            mkdir results/Hybrid/words
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --save-index=results/Hybrid/words/index --statistics-file=results/Hybrid/words/$s.csv > results/Hybrid/words/$s &
        else
            ./build/main datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --load-index=results/Hybrid/words/index --statistics-file=results/Hybrid/words/$s.csv > results/Hybrid/words/$s &
        fi
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
    run_acorn_variants "words" datasets/words/strings.txt datasets/words/vectors.txt "$s"
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/words" ]; then
            mkdir results/pgvector/words
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_pgvector.py datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv pgvector_hnsw_stats.csv results/pgvector/words/$s.csv
        else
            python3 test_pgvector.py datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/words/$s.csv
        fi
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/words" ]; then
            mkdir results/ElasticSearch/words
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/words/$s.csv
        else
            python3 test_elasticsearch.py datasets/words/strings.txt datasets/words/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/words/$s.csv
        fi
    fi
done
fi

# # Run mtg
if should_run_dataset "mtg"; then
for s in 2 3 4 5 6 7
do
    python3 scripts/generate_queries.py datasets/mtg/strings.txt datasets/mtg/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/mtg" ]; then
            mkdir results/PreFiltering/mtg
        fi
        ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt --statistics-file=results/PreFiltering/mtg/$s.csv > results/PreFiltering/mtg/$s &
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
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --save-index=results/PostFiltering/mtg/index --statistics-file=results/PostFiltering/mtg/$s.csv > results/PostFiltering/mtg/$s &
        else
            ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --load-index=results/PostFiltering/mtg/index --statistics-file=results/PostFiltering/mtg/$s.csv > results/PostFiltering/mtg/$s &
        fi
    fi
    # Hybrid
    if should_run "Hybrid"; then
        if [ ! -d "results/Hybrid/mtg" ]; then
            mkdir results/Hybrid/mtg
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --save-index=results/Hybrid/mtg/index --statistics-file=results/Hybrid/mtg/$s.csv > results/Hybrid/mtg/$s &
        else
            ./build/main datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --load-index=results/Hybrid/mtg/index --statistics-file=results/Hybrid/mtg/$s.csv > results/Hybrid/mtg/$s &
        fi
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
    run_acorn_variants "mtg" datasets/mtg/strings.txt datasets/mtg/vectors.txt "$s"
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/mtg" ]; then
            mkdir results/pgvector/mtg
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_pgvector.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv pgvector_hnsw_stats.csv results/pgvector/mtg/$s.csv
        else
            python3 test_pgvector.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/mtg/$s.csv
        fi
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/mtg" ]; then
            mkdir results/ElasticSearch/mtg
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/mtg/$s.csv
        else
            python3 test_elasticsearch.py datasets/mtg/strings.txt datasets/mtg/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/mtg/$s.csv
        fi
    fi
done
fi

# # Run arxiv-small
if should_run_dataset "arxiv-small"; then
for s in 2 3 4 5 6 7
do
    python3 scripts/generate_queries.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/arxiv-small" ]; then
            mkdir results/PreFiltering/arxiv-small
        fi
        ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt --statistics-file=results/PreFiltering/arxiv-small/$s.csv > results/PreFiltering/arxiv-small/$s &
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
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --save-index=results/PostFiltering/arxiv-small/index --statistics-file=results/PostFiltering/arxiv-small/$s.csv > results/PostFiltering/arxiv-small/$s &
        else
            ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --load-index=results/PostFiltering/arxiv-small/index --statistics-file=results/PostFiltering/arxiv-small/$s.csv > results/PostFiltering/arxiv-small/$s &
        fi
    fi
    # Hybrid
    if should_run "Hybrid"; then
        if [ ! -d "results/Hybrid/arxiv-small" ]; then
            mkdir results/Hybrid/arxiv-small
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --save-index=results/Hybrid/arxiv-small/index --statistics-file=results/Hybrid/arxiv-small/$s.csv > results/Hybrid/arxiv-small/$s &
        else
            ./build/main datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --load-index=results/Hybrid/arxiv-small/index --statistics-file=results/Hybrid/arxiv-small/$s.csv > results/Hybrid/arxiv-small/$s &
        fi
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
    run_acorn_variants "arxiv-small" datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt "$s"
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/arxiv-small" ]; then
            mkdir results/pgvector/arxiv-small
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_pgvector.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv pgvector_hnsw_stats.csv results/pgvector/arxiv-small/$s.csv
        else
            python3 test_pgvector.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/arxiv-small/$s.csv
        fi
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/arxiv-small" ]; then
            mkdir results/ElasticSearch/arxiv-small
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/arxiv-small/$s.csv
        else
            python3 test_elasticsearch.py datasets/arxiv-small/strings.txt datasets/arxiv-small/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/arxiv-small/$s.csv
        fi
    fi
done
fi

# # Run swissprot
if should_run_dataset "swissprot"; then
for s in 2 3 4 5 6 7
do
    python3 scripts/generate_queries.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/swissprot" ]; then
            mkdir results/PreFiltering/swissprot
        fi
        ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt --statistics-file=results/PreFiltering/swissprot/$s.csv > results/PreFiltering/swissprot/$s &
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
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --save-index=results/PostFiltering/swissprot/index --statistics-file=results/PostFiltering/swissprot/$s.csv > results/PostFiltering/swissprot/$s &
        else
            ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --load-index=results/PostFiltering/swissprot/index --statistics-file=results/PostFiltering/swissprot/$s.csv > results/PostFiltering/swissprot/$s &
        fi
    fi
    # Hybrid
    if should_run "Hybrid"; then
        if [ ! -d "results/Hybrid/swissprot" ]; then
            mkdir results/Hybrid/swissprot
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --save-index=results/Hybrid/swissprot/index --statistics-file=results/Hybrid/swissprot/$s.csv > results/Hybrid/swissprot/$s &
        else
            ./build/main datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --load-index=results/Hybrid/swissprot/index --statistics-file=results/Hybrid/swissprot/$s.csv > results/Hybrid/swissprot/$s &
        fi
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
    run_acorn_variants "swissprot" datasets/swissprot/strings.txt datasets/swissprot/vectors.txt "$s"
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/swissprot" ]; then
            mkdir results/pgvector/swissprot
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_pgvector.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv pgvector_hnsw_stats.csv results/pgvector/swissprot/$s.csv
        else
            python3 test_pgvector.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/swissprot/$s.csv
        fi
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/swissprot" ]; then
            mkdir results/ElasticSearch/swissprot
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/swissprot/$s.csv
        else
            python3 test_elasticsearch.py datasets/swissprot/strings.txt datasets/swissprot/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/swissprot/$s.csv
        fi
    fi
done
fi

# Run code_search_net
if should_run_dataset "code_search_net"; then
for s in 2 3 4 5 6 7
do
    python3 scripts/generate_queries.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt $s 1000 10 -1 queries
    # PreFiltering
    if should_run "PreFiltering"; then
        if [ ! -d "results/PreFiltering/code_search_net" ]; then
            mkdir results/PreFiltering/code_search_net
        fi
        ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PreFiltering --write-ground-truth=ground_truth.txt --statistics-file=results/PreFiltering/code_search_net/$s.csv > results/PreFiltering/code_search_net/$s &
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
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --save-index=results/PostFiltering/code_search_net/index --statistics-file=results/PostFiltering/code_search_net/$s.csv > results/PostFiltering/code_search_net/$s &
        else
            ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt PostFiltering --load-index=results/PostFiltering/code_search_net/index --statistics-file=results/PostFiltering/code_search_net/$s.csv > results/PostFiltering/code_search_net/$s &
        fi
    fi
    # Hybrid
    if should_run "Hybrid"; then
        if [ ! -d "results/Hybrid/code_search_net" ]; then
            mkdir results/Hybrid/code_search_net
        fi
        if [ "$s" -eq 2 ]; then
            ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --save-index=results/Hybrid/code_search_net/index --statistics-file=results/Hybrid/code_search_net/$s.csv > results/Hybrid/code_search_net/$s &
        else
            ./build/main datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt Hybrid --load-index=results/Hybrid/code_search_net/index --statistics-file=results/Hybrid/code_search_net/$s.csv > results/Hybrid/code_search_net/$s &
        fi
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
    run_acorn_variants "code_search_net" datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt "$s"
    # pgvector
    if should_run "pgvector"; then
        if [ ! -d "results/pgvector/code_search_net" ]; then
            mkdir results/pgvector/code_search_net
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_pgvector.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild && mv pgvector_hnsw_stats.csv results/pgvector/code_search_net/$s.csv
        else
            python3 test_pgvector.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt && mv pgvector_hnsw_stats.csv results/pgvector/code_search_net/$s.csv
        fi
    fi
    if should_run "ElasticSearch"; then
        if [ ! -d "results/ElasticSearch/code_search_net" ]; then
            mkdir results/ElasticSearch/code_search_net
        fi
        if [ "$s" -eq 2 ]; then
            python3 test_elasticsearch.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt --rebuild $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/code_search_net/$s.csv
        else
            python3 test_elasticsearch.py datasets/code_search_net/strings.txt datasets/code_search_net/vectors.txt strings_queries.txt vectors_queries.txt k_queries.txt ground_truth.txt $ELASTICSEARCH_PID_ARG && mv elasticsearch_hnsw_stats.csv results/ElasticSearch/code_search_net/$s.csv
        fi
    fi
done
fi

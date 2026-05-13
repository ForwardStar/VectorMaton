#!/bin/bash
set -eu
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

BLACKLIST_DATASET_RAW=""
while [ $# -gt 0 ]; do
    case "$1" in
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
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

BLACKLIST_DATASET_NORM="$(echo ",${BLACKLIST_DATASET_RAW}," | tr '[:upper:]' '[:lower:]' | tr -d ' ')"

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

run_dataset() {
    local dataset="$1"
    shift

    if ! should_run_dataset "$dataset"; then
        return
    fi

    mkdir -p "results/PostFiltering/parameter_study/$dataset"
    for s in "$@"; do
        python3 scripts/generate_queries.py "datasets/$dataset/strings.txt" "datasets/$dataset/vectors.txt" "$s" 1000 10 -1 queries
        ./build/parameter_study \
            "datasets/$dataset/strings.txt" \
            "datasets/$dataset/vectors.txt" \
            strings_queries.txt \
            vectors_queries.txt \
            k_queries.txt \
            PostFiltering \
            --statistics-file="results/PostFiltering/parameter_study/$dataset/$s.csv"
    done
}

run_dataset "spam" 2 3 4 5 6 7 8 16 32
run_dataset "words" 2 3 4 5 6 7 8 16
run_dataset "mtg" 2 3 4 5 6 7 8 16 32
run_dataset "arxiv-small" 2 3 4 5 6 7 8 16 32
run_dataset "swissprot" 2 3 4 5 6 7 8 16 32
run_dataset "code_search_net" 2 3 4 5 6 7 8 16 32

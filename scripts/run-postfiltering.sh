#!/bin/bash
set -eu
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

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

    if ! should_run_dataset "$dataset"; then
        return
    fi

    local result_dir="results/PostFiltering/parameter_study/$dataset"
    local query_dir="${result_dir}/.tmp_queries/mixed_2_4"
    mkdir -p "${result_dir}"
    rm -rf "${query_dir}"
    mkdir -p "${query_dir}"

    (
        cd "${query_dir}"
        python3 "${REPO_ROOT}/scripts/generate_queries.py" \
            "${REPO_ROOT}/datasets/${dataset}/strings.txt" \
            "${REPO_ROOT}/datasets/${dataset}/vectors.txt" \
            2 \
            1000 \
            10 \
            -1 \
            queries \
            --mixed-length
    )

    ./build/parameter_study \
        "datasets/$dataset/strings.txt" \
        "datasets/$dataset/vectors.txt" \
        "${query_dir}/strings_queries.txt" \
        "${query_dir}/vectors_queries.txt" \
        "${query_dir}/k_queries.txt" \
        PostFiltering \
        --statistics-file="${result_dir}/mixed_2_4.csv"
}

run_dataset "spam"
run_dataset "words"
run_dataset "mtg"
run_dataset "arxiv-small"
run_dataset "swissprot"
run_dataset "code_search_net"

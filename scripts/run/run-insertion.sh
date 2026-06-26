#!/bin/bash
set -eu
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

RESULT_ROOT="${RESULT_ROOT:-results/insertion}"
QUERY_STRING_LEN="${QUERY_STRING_LEN:-2}"
NUM_QUERIES="${NUM_QUERIES:-10}"
K_VALUE="${K_VALUE:-10}"
INSERTION_PERCENTAGES="${INSERTION_PERCENTAGES:-20}"
METHODS="${METHODS:-OptQuery BM25Filtering PreFiltering PostFiltering Hybrid VectorMaton-smart pgvector ElasticSearch}"
DATASETS="${DATASETS:-}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --result-root DIR             Output root (default: ${RESULT_ROOT})
  --insert-percentages LIST     Space-separated percentages (default: ${INSERTION_PERCENTAGES})
  --methods LIST                Space-separated methods (default: non-ACORN methods)
  --datasets LIST               Space-separated datasets (default: auto-discover)
  --num-queries N               Number of generated queries (default: ${NUM_QUERIES})
  --query-string-len N          Query substring length (default: ${QUERY_STRING_LEN})
  --k VALUE                     k for generated queries (default: ${K_VALUE})

Environment variables with the same uppercase names are also supported.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --result-root)
            RESULT_ROOT="$2"
            shift 2
            ;;
        --result-root=*)
            RESULT_ROOT="${1#*=}"
            shift
            ;;
        --insert-percentages)
            INSERTION_PERCENTAGES="$2"
            shift 2
            ;;
        --insert-percentages=*)
            INSERTION_PERCENTAGES="${1#*=}"
            shift
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --methods=*)
            METHODS="${1#*=}"
            shift
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --datasets=*)
            DATASETS="${1#*=}"
            shift
            ;;
        --num-queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --num-queries=*)
            NUM_QUERIES="${1#*=}"
            shift
            ;;
        --query-string-len)
            QUERY_STRING_LEN="$2"
            shift 2
            ;;
        --query-string-len=*)
            QUERY_STRING_LEN="${1#*=}"
            shift
            ;;
        --k)
            K_VALUE="$2"
            shift 2
            ;;
        --k=*)
            K_VALUE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

method_enabled() {
    local needle="$1"
    for method in ${METHODS}; do
        if [ "${method}" = "${needle}" ]; then
            return 0
        fi
    done
    return 1
}

needs_ground_truth() {
    method_enabled "ACORN-gamma" ||
        method_enabled "ACORN-1" ||
        method_enabled "pgvector" ||
        method_enabled "ElasticSearch"
}

if [ ! -x "./build/main_exp" ]; then
    echo "Missing executable ./build/main_exp (build the project first)." >&2
    exit 1
fi

if { method_enabled "ACORN-gamma" || method_enabled "ACORN-1"; } && [ ! -x "./build/acorn_exp" ]; then
    echo "Missing executable ./build/acorn_exp (build the project first)." >&2
    exit 1
fi

if [ -z "${DATASETS}" ]; then
    DATASETS=$(
        find datasets -mindepth 1 -maxdepth 1 -type d | sort | while read -r dataset_dir; do
            if [ -f "${dataset_dir}/strings.txt" ] && [ -f "${dataset_dir}/vectors.txt" ]; then
                basename "${dataset_dir}"
            fi
        done
    )
fi

if [ -z "${DATASETS}" ]; then
    echo "No valid datasets found under datasets/ (expected strings.txt and vectors.txt)." >&2
    exit 1
fi

mkdir -p "${RESULT_ROOT}"

run_main_method() {
    local method="$1"
    local dataset="$2"
    local pct="$3"
    local strings_file="$4"
    local vectors_file="$5"
    local query_strings="$6"
    local query_vectors="$7"
    local query_k="$8"
    local ground_truth_file="$9"

    local method_dir="${RESULT_ROOT}/${method}/${dataset}"
    local tag="insert_${pct}"
    mkdir -p "${method_dir}"

    local extra_ground_truth_arg=""
    if [ "${method}" = "PreFiltering" ] && [ -n "${ground_truth_file}" ]; then
        extra_ground_truth_arg="--write-ground-truth=${ground_truth_file}"
    fi

    echo "==> ${dataset}: ${method}, insertion=${pct}%"
    ./build/main_exp \
        "${strings_file}" \
        "${vectors_file}" \
        "${query_strings}" \
        "${query_vectors}" \
        "${query_k}" \
        "${method}" \
        "--insert-percentage=${pct}" \
        "--statistics-file=${method_dir}/${tag}.csv" \
        ${extra_ground_truth_arg} \
        > "${method_dir}/${tag}.log" 2>&1
}

generate_ground_truth_only() {
    local dataset="$1"
    local pct="$2"
    local strings_file="$3"
    local vectors_file="$4"
    local query_strings="$5"
    local query_vectors="$6"
    local query_k="$7"
    local ground_truth_file="$8"
    local log_file="${RESULT_ROOT}/.tmp_queries/${dataset}/ground_truth_insert.log"

    echo "==> ${dataset}: generating ground truth, insertion=${pct}%"
    ./build/main_exp \
        "${strings_file}" \
        "${vectors_file}" \
        "${query_strings}" \
        "${query_vectors}" \
        "${query_k}" \
        PreFiltering \
        "--insert-percentage=${pct}" \
        "--write-ground-truth=${ground_truth_file}" \
        > "${log_file}" 2>&1
}

run_acorn() {
    local variant="$1"
    local dataset="$2"
    local pct="$3"
    local strings_file="$4"
    local vectors_file="$5"
    local query_strings="$6"
    local query_vectors="$7"
    local query_k="$8"
    local ground_truth_file="$9"

    local gamma="32"
    local m_beta="64"
    if [ "${variant}" = "ACORN-1" ]; then
        gamma="1"
        m_beta="32"
    fi

    local method_dir="${RESULT_ROOT}/${variant}/${dataset}"
    local tag="insert_${pct}"
    mkdir -p "${method_dir}"

    echo "==> ${dataset}: ${variant}, insertion=${pct}%"
    OMP_NUM_THREADS=1 ./build/acorn_exp \
        "${strings_file}" \
        "${vectors_file}" \
        "${query_strings}" \
        "${query_vectors}" \
        "${query_k}" \
        "${ground_truth_file}" \
        --M=32 \
        "--gamma=${gamma}" \
        "--M-beta=${m_beta}" \
        "--insertion-percentage=${pct}" \
        "--output=${method_dir}/${tag}.csv" \
        > "${method_dir}/${tag}.log" 2>&1
}

run_pgvector() {
    local dataset="$1"
    local pct="$2"
    local strings_file="$3"
    local vectors_file="$4"
    local query_strings="$5"
    local query_vectors="$6"
    local query_k="$7"
    local ground_truth_file="$8"

    local method_dir="${RESULT_ROOT}/pgvector/${dataset}"
    local tag="insert_${pct}"
    mkdir -p "${method_dir}"

    echo "==> ${dataset}: pgvector, insertion=${pct}%"
    python3 source/experiments/pgvector_exp.py \
        "${strings_file}" \
        "${vectors_file}" \
        "${query_strings}" \
        "${query_vectors}" \
        "${query_k}" \
        "${ground_truth_file}" \
        --rebuild \
        "--insertion-percentage=${pct}" \
        > "${method_dir}/${tag}.log" 2>&1
    mv pgvector_hnsw_stats.csv "${method_dir}/${tag}.csv"
}

run_elasticsearch() {
    local dataset="$1"
    local pct="$2"
    local strings_file="$3"
    local vectors_file="$4"
    local query_strings="$5"
    local query_vectors="$6"
    local query_k="$7"
    local ground_truth_file="$8"

    local method_dir="${RESULT_ROOT}/ElasticSearch/${dataset}"
    local tag="insert_${pct}"
    local index_pct
    index_pct=$(printf "%s" "${pct}" | tr '.+' '__')
    mkdir -p "${method_dir}"

    echo "==> ${dataset}: ElasticSearch, insertion=${pct}%"
    python3 source/experiments/elasticsearch_exp.py \
        "${strings_file}" \
        "${vectors_file}" \
        "${query_strings}" \
        "${query_vectors}" \
        "${query_k}" \
        "${ground_truth_file}" \
        --rebuild \
        "--index-name=vectormaton_insertion_${dataset}_${index_pct}" \
        "--insertion-percentage=${pct}" \
        > "${method_dir}/${tag}.log" 2>&1
    mv elasticsearch_hnsw_stats.csv "${method_dir}/${tag}.csv"
}

run_dataset() {
    local dataset="$1"
    local strings_file="datasets/${dataset}/strings.txt"
    local vectors_file="datasets/${dataset}/vectors.txt"
    local query_dir="${RESULT_ROOT}/.tmp_queries/${dataset}"

    if [ ! -f "${strings_file}" ] || [ ! -f "${vectors_file}" ]; then
        echo "Skipping ${dataset}: missing strings.txt or vectors.txt" >&2
        return 0
    fi

    rm -rf "${query_dir}"
    mkdir -p "${query_dir}"

    echo "==> Generating insertion queries for ${dataset}"
    (
        cd "${query_dir}"
        python3 "${REPO_ROOT}/scripts/generate_queries.py" \
            "${REPO_ROOT}/${strings_file}" \
            "${REPO_ROOT}/${vectors_file}" \
            "${QUERY_STRING_LEN}" \
            "${NUM_QUERIES}" \
            "${K_VALUE}" \
            -1 \
            insert
    )

    local query_strings="${query_dir}/strings_insert.txt"
    local query_vectors="${query_dir}/vectors_insert.txt"
    local query_k="${query_dir}/k_insert.txt"

    for pct in ${INSERTION_PERCENTAGES}; do
        local ground_truth_file="${query_dir}/ground_truth_insert.txt"
        local prefilter_ran=0

        if method_enabled "PreFiltering"; then
            run_main_method \
                PreFiltering \
                "${dataset}" \
                "${pct}" \
                "${strings_file}" \
                "${vectors_file}" \
                "${query_strings}" \
                "${query_vectors}" \
                "${query_k}" \
                "${ground_truth_file}"
            prefilter_ran=1
        fi

        if needs_ground_truth && [ "${prefilter_ran}" -eq 0 ]; then
            generate_ground_truth_only \
                "${dataset}" \
                "${pct}" \
                "${strings_file}" \
                "${vectors_file}" \
                "${query_strings}" \
                "${query_vectors}" \
                "${query_k}" \
                "${ground_truth_file}"
        fi

        for method in ${METHODS}; do
            if [ "${method}" = "OptQuery" ] && [ "${dataset}" != "spam" ] && [ "${dataset}" != "words" ]; then
                echo "==> ${dataset}: skipping OptQuery (only run on spam and words)"
                continue
            fi
            if [ "${method}" = "pgvector" ] && [ "${dataset}" = "words" ]; then
                echo "==> ${dataset}: skipping pgvector"
                continue
            fi

            case "${method}" in
                PreFiltering)
                    ;;
                OptQuery|BM25Filtering|PostFiltering|Hybrid|VectorMaton-full|VectorMaton-smart|VectorMaton-parallel)
                    run_main_method \
                        "${method}" \
                        "${dataset}" \
                        "${pct}" \
                        "${strings_file}" \
                        "${vectors_file}" \
                        "${query_strings}" \
                        "${query_vectors}" \
                        "${query_k}" \
                        ""
                    ;;
                ACORN-gamma|ACORN-1)
                    run_acorn \
                        "${method}" \
                        "${dataset}" \
                        "${pct}" \
                        "${strings_file}" \
                        "${vectors_file}" \
                        "${query_strings}" \
                        "${query_vectors}" \
                        "${query_k}" \
                        "${ground_truth_file}"
                    ;;
                pgvector)
                    run_pgvector \
                        "${dataset}" \
                        "${pct}" \
                        "${strings_file}" \
                        "${vectors_file}" \
                        "${query_strings}" \
                        "${query_vectors}" \
                        "${query_k}" \
                        "${ground_truth_file}"
                    ;;
                ElasticSearch)
                    run_elasticsearch \
                        "${dataset}" \
                        "${pct}" \
                        "${strings_file}" \
                        "${vectors_file}" \
                        "${query_strings}" \
                        "${query_vectors}" \
                        "${query_k}" \
                        "${ground_truth_file}"
                    ;;
                *)
                    echo "Unknown method in METHODS: ${method}" >&2
                    exit 1
                    ;;
            esac
        done
    done
}

for dataset in ${DATASETS}; do
    run_dataset "${dataset}"
done

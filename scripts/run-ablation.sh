#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

RESULT_ROOT="${RESULT_ROOT:-results/ablation}"
QUERY_STRING_LEN="${QUERY_STRING_LEN:-2}"
NUM_QUERIES="${NUM_QUERIES:-10}"
K_VALUE="${K_VALUE:-10}"
TRUNCATE_LEN="${TRUNCATE_LEN:--1}"
METHODS="VectorMaton-full VectorMaton-smart"

mkdir -p "${RESULT_ROOT}"

DATASETS=$(
    find datasets -mindepth 1 -maxdepth 1 -type d | sort | while read -r dataset_dir; do
        if [ -f "${dataset_dir}/strings.txt" ] && [ -f "${dataset_dir}/vectors.txt" ]; then
            dataset_name=$(basename "${dataset_dir}")
            if [ "${dataset_name}" != "arxiv" ]; then
                echo "${dataset_name}"
            fi
        fi
    done
)

if [ -z "${DATASETS}" ]; then
    echo "No valid datasets found under datasets/ (expected strings.txt and vectors.txt)." >&2
    exit 1
fi

run_dataset() {
    dataset="$1"
    strings_file="datasets/${dataset}/strings.txt"
    vectors_file="datasets/${dataset}/vectors.txt"
    query_dir="${RESULT_ROOT}/.tmp_queries/${dataset}"

    rm -rf "${query_dir}"
    mkdir -p "${query_dir}"
    echo "==> Generating queries for ${dataset}"
    (
        cd "${query_dir}"
        python3 "${REPO_ROOT}/scripts/generate_queries.py" \
            "${REPO_ROOT}/${strings_file}" \
            "${REPO_ROOT}/${vectors_file}" \
            "${QUERY_STRING_LEN}" \
            "${NUM_QUERIES}" \
            "${K_VALUE}" \
            "${TRUNCATE_LEN}" \
            ablation
    )

    method_failed=0
    method_pids=""

    for method in ${METHODS}; do
        output_dir="${RESULT_ROOT}/${method}/${dataset}"
        mkdir -p "${output_dir}"

        echo "==> Running ${method} on ${dataset}"
        ./build/main \
            "${strings_file}" \
            "${vectors_file}" \
            "${query_dir}/strings_ablation.txt" \
            "${query_dir}/vectors_ablation.txt" \
            "${query_dir}/k_ablation.txt" \
            "${method}" \
            --statistics-file="${output_dir}/stats.csv" \
            > "${output_dir}/run.log" &
        method_pids="${method_pids} $!"
    done

    for pid in ${method_pids}; do
        if ! wait "${pid}"; then
            method_failed=1
        fi
    done

    if [ "${method_failed}" -ne 0 ]; then
        rm -rf "${query_dir}"
        return 1
    fi

    rm -rf "${query_dir}"
}

for dataset in ${DATASETS}; do
    run_dataset "${dataset}"
done

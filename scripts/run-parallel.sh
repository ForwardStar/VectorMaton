#!/bin/sh
set -eu

RESULT_ROOT="results/VectorMaton-parallel"
THREADS="1 2 4 8 16 32"

mkdir -p "${RESULT_ROOT}"

for dataset in swissprot code_search_net; do
    strings_file="datasets/${dataset}/strings.txt"
    vectors_file="datasets/${dataset}/vectors.txt"
    output_dir="${RESULT_ROOT}/${dataset}"

    mkdir -p "${output_dir}"

    # The binary requires query files as positional arguments.
    python3 scripts/generate_queries.py "${strings_file}" "${vectors_file}" 2 10 10 -1

    for num_threads in ${THREADS}; do
        log_file="${output_dir}/${num_threads}.log"
        ./build/main \
            "${strings_file}" \
            "${vectors_file}" \
            strings.txt \
            vectors.txt \
            k.txt \
            VectorMaton-parallel \
            --num-threads="${num_threads}" \
            > "${log_file}"
    done
done

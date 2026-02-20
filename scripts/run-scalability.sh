#!/bin/sh
set -eu

mkdir -p results/VectorMaton

for dataset in spam words mtg arxiv-small swissprot code_search_net; do
    strings_file="datasets/${dataset}/strings.txt"
    vectors_file="datasets/${dataset}/vectors.txt"
    output_dir="results/VectorMaton/${dataset}"

    python3 scripts/generate_queries.py "${strings_file}" "${vectors_file}" 2 1000 10 -1
    mkdir -p "${output_dir}"

    total_strings=$(awk 'END { print NR }' "${strings_file}")

    for pct in 10 20 30 40 50 60 70 80 90; do
        data_size=$((total_strings * pct / 100))
        if [ "${data_size}" -lt 1 ]; then
            data_size=1
        fi

        ./build/main "${strings_file}" "${vectors_file}" strings.txt vectors.txt k.txt VectorMaton-parallel --data-size="${data_size}" --num-threads=32 > "${output_dir}/${pct}%"
    done

    ./build/main "${strings_file}" "${vectors_file}" strings.txt vectors.txt k.txt VectorMaton-parallel --num-threads=32 > "${output_dir}/100%"
done

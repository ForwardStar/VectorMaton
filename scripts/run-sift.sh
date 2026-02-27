#!/bin/sh
set -eu

DATASET="sift"
STRINGS_FILE="datasets/${DATASET}/strings.txt"
VECTORS_FILE="datasets/${DATASET}/vectors.txt"

if [ ! -f "${STRINGS_FILE}" ] || [ ! -f "${VECTORS_FILE}" ]; then
    echo "SIFT dataset files not found. Please run: python3 scripts/download_datasets.py" >&2
    exit 1
fi

mkdir -p results/PreFiltering/"${DATASET}"
mkdir -p results/PostFiltering/"${DATASET}"
mkdir -p results/VectorMaton-parallel/"${DATASET}"

for s in 2 3 4
do
    python3 scripts/generate_queries.py "${STRINGS_FILE}" "${VECTORS_FILE}" "$s" 1000 10 -1 sift

    ./build/main \
        "${STRINGS_FILE}" \
        "${VECTORS_FILE}" \
        strings_sift.txt \
        vectors_sift.txt \
        k_sift.txt \
        PreFiltering \
        --write-ground-truth=ground_truth.txt \
        > "results/PreFiltering/${DATASET}/${s}" &

    ./build/main \
        "${STRINGS_FILE}" \
        "${VECTORS_FILE}" \
        strings_sift.txt \
        vectors_sift.txt \
        k_sift.txt \
        PostFiltering \
        --statistics-file="results/PostFiltering/${DATASET}/${s}.csv" \
        > "results/PostFiltering/${DATASET}/${s}" &

    ./build/main \
        "${STRINGS_FILE}" \
        "${VECTORS_FILE}" \
        strings_sift.txt \
        vectors_sift.txt \
        k_sift.txt \
        VectorMaton-parallel \
        --num-threads=32 \
        --statistics-file="results/VectorMaton-parallel/${DATASET}/${s}.csv" \
        > "results/VectorMaton-parallel/${DATASET}/${s}" &

    wait
done

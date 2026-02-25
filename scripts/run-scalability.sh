#!/bin/sh
set -eu

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

BLACKLIST_NORM=$(printf ",%s," "$BLACKLIST_RAW" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

is_blacklisted() {
    dataset=$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')
    case "$BLACKLIST_NORM" in
        *,"$dataset",*) return 0 ;;
        *) return 1 ;;
    esac
}

should_run_dataset() {
    if is_blacklisted "$1"; then
        return 1
    fi
    return 0
}

mkdir -p results/VectorMaton
mkdir -p results/OptQuery

run_optquery=true

run_dataset() {
    dataset_name="$1"
    strings_file="datasets/${dataset_name}/strings.txt"
    vectors_file="datasets/${dataset_name}/vectors.txt"
    output_dir="results/VectorMaton/scalability/${dataset_name}"
    optquery_output_dir="results/OptQuery/scalability/${dataset_name}"

    echo "Running scalability test for dataset ${dataset_name}..."

    python3 scripts/generate_queries.py "${strings_file}" "${vectors_file}" 2 10 10 -1 scalability_queries
    mkdir -p "${output_dir}"
    mkdir -p "${optquery_output_dir}"

    total_strings=$(awk 'END { print NR }' "${strings_file}")

    for pct in 10 20 30 40 50 60 70 80 90; do
        data_size=$((total_strings * pct / 100))
        if [ "${data_size}" -lt 1 ]; then
            data_size=1
        fi

        ./build/main "${strings_file}" "${vectors_file}" strings_scalability_queries.txt vectors_scalability_queries.txt k_scalability_queries.txt VectorMaton-smart --data-size="${data_size}" > "${output_dir}/${pct}%"
        if [ "${run_optquery}" = "true" ]; then
            ./build/main "${strings_file}" "${vectors_file}" strings_scalability_queries.txt vectors_scalability_queries.txt k_scalability_queries.txt OptQuery --data-size="${data_size}" > "${optquery_output_dir}/${pct}%"
        fi
    done

    ./build/main "${strings_file}" "${vectors_file}" strings_scalability_queries.txt vectors_scalability_queries.txt k_scalability_queries.txt VectorMaton-smart > "${output_dir}/100%"
    if [ "${run_optquery}" = "true" ]; then
        ./build/main "${strings_file}" "${vectors_file}" strings_scalability_queries.txt vectors_scalability_queries.txt k_scalability_queries.txt OptQuery > "${optquery_output_dir}/100%"
    fi
}

if should_run_dataset "spam"; then
    run_dataset "spam"
fi
if should_run_dataset "words"; then
    run_dataset "words"
fi

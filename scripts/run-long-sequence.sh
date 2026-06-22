#!/bin/bash
set -eu
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

BLACKLIST_RAW=""
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

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

DATASET="wikipedia"
DATASET_DIR="${DATASET_DIR:-datasets/${DATASET}}"
STRINGS_FILE="${DATASET_DIR}/strings.txt"
VECTORS_FILE="${DATASET_DIR}/vectors.txt"
PATTERN_LENGTHS="${PATTERN_LENGTHS:-12}"
FIRST_PATTERN_LENGTH="$(printf '%s\n' ${PATTERN_LENGTHS} | head -n 1)"
NUM_QUERIES="${NUM_QUERIES:-1000}"
K_VALUE="${K_VALUE:-10}"
TRUNCATE_LEN="${TRUNCATE_LEN:--1}"
RESULT_ROOT="${RESULT_ROOT:-results}"
QUERY_SUFFIX="${QUERY_SUFFIX:-long_sequence}"
QUERY_STRINGS="${QUERY_STRINGS:-string_prompt.txt}"
QUERY_VECTORS="${QUERY_VECTORS:-vectors_prompt.txt}"
QUERY_K="${QUERY_K:-k_prompt.txt}"
GROUND_TRUTH="ground_truth_${QUERY_SUFFIX}.txt"

if [ ! -x "./build/main" ]; then
    echo "Missing executable ./build/main (build the project first)." >&2
    exit 1
fi

if [ ! -x "./build/test_acorn" ]; then
    echo "Missing executable ./build/test_acorn (build the project first)." >&2
    exit 1
fi

if [ ! -f "${STRINGS_FILE}" ] || [ ! -f "${VECTORS_FILE}" ]; then
    echo "Missing dataset files under ${DATASET_DIR} (expected strings.txt and vectors.txt)." >&2
    exit 1
fi

if [ ! -f "${QUERY_STRINGS}" ] || [ ! -f "${QUERY_VECTORS}" ]; then
    echo "Missing prompt query files (expected ${QUERY_STRINGS} and ${QUERY_VECTORS})." >&2
    echo "Run: python3 scripts/prompt_embedding.py" >&2
    exit 1
fi

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

mkdir -p \
    "${RESULT_ROOT}/PreFiltering/${DATASET}" \
    "${RESULT_ROOT}/PostFiltering/${DATASET}" \
    "${RESULT_ROOT}/Hybrid/${DATASET}" \
    "${RESULT_ROOT}/VectorMaton/${DATASET}" \
    "${RESULT_ROOT}/BM25Filtering/${DATASET}" \
    "${RESULT_ROOT}/ACORN-gamma/${DATASET}" \
    "${RESULT_ROOT}/ACORN-1/${DATASET}" \
    "${RESULT_ROOT}/pgvector/${DATASET}" \
    "${RESULT_ROOT}/ElasticSearch/${DATASET}"

index_flag() {
    local method="$1"
    local pattern_length="$2"
    local index_dir="${RESULT_ROOT}/${method}/${DATASET}/index"

    if [ "$pattern_length" -eq "$FIRST_PATTERN_LENGTH" ]; then
        printf '%s' "--save-index=${index_dir}"
    else
        printf '%s' "--load-index=${index_dir}"
    fi
}

run_acorn_variant() {
    local result_method="$1"
    local gamma="$2"
    local m_beta="$3"
    local pattern_length="$4"

    if ! should_run "$result_method"; then
        return
    fi

    local index_dir="${RESULT_ROOT}/${result_method}/${DATASET}/index"
    local acorn_index_flag
    if [ "$pattern_length" -eq "$FIRST_PATTERN_LENGTH" ]; then
        acorn_index_flag="--save-index=${index_dir}"
    else
        acorn_index_flag="--load-index=${index_dir}"
    fi

    OMP_NUM_THREADS=1 ./build/test_acorn \
        "${STRINGS_FILE}" "${VECTORS_FILE}" \
        "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" "${GROUND_TRUTH}" \
        --M=32 --gamma="${gamma}" --M-beta="${m_beta}" \
        "${acorn_index_flag}" \
        --write-output="${RESULT_ROOT}/${result_method}/${DATASET}/${pattern_length}.queries" \
        --output="${RESULT_ROOT}/${result_method}/${DATASET}/${pattern_length}.csv" \
        > "${RESULT_ROOT}/${result_method}/${DATASET}/${pattern_length}"
}

for s in ${PATTERN_LENGTHS}; do
    echo "==> Using prompt query files: ${QUERY_STRINGS}, ${QUERY_VECTORS}, k=${K_VALUE}"
    awk -v k="${K_VALUE}" "{ print k }" "${QUERY_STRINGS}" > "${QUERY_K}"

    echo "==> Running PreFiltering for ground truth"
    ./build/main \
        "${STRINGS_FILE}" "${VECTORS_FILE}" \
        "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" \
        PreFiltering \
        --write-ground-truth="${GROUND_TRUTH}" \
        --statistics-file="${RESULT_ROOT}/PreFiltering/${DATASET}/${s}.csv" \
        --write-output="${RESULT_ROOT}/PreFiltering/${DATASET}/${s}.queries" \
        > "${RESULT_ROOT}/PreFiltering/${DATASET}/${s}"

    if should_run "PostFiltering"; then
        ./build/main \
            "${STRINGS_FILE}" "${VECTORS_FILE}" \
            "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" \
            PostFiltering \
            "$(index_flag PostFiltering "$s")" \
            --statistics-file="${RESULT_ROOT}/PostFiltering/${DATASET}/${s}.csv" \
            --write-output="${RESULT_ROOT}/PostFiltering/${DATASET}/${s}.queries" \
            > "${RESULT_ROOT}/PostFiltering/${DATASET}/${s}" &
    fi

    if should_run "Hybrid"; then
        ./build/main \
            "${STRINGS_FILE}" "${VECTORS_FILE}" \
            "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" \
            Hybrid \
            "$(index_flag Hybrid "$s")" \
            --statistics-file="${RESULT_ROOT}/Hybrid/${DATASET}/${s}.csv" \
            --write-output="${RESULT_ROOT}/Hybrid/${DATASET}/${s}.queries" \
            > "${RESULT_ROOT}/Hybrid/${DATASET}/${s}" &
    fi

    if should_run "VectorMaton"; then
        ./build/main \
            "${STRINGS_FILE}" "${VECTORS_FILE}" \
            "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" \
            VectorMaton-smart \
            "$(index_flag VectorMaton "$s")" \
            --statistics-file="${RESULT_ROOT}/VectorMaton/${DATASET}/${s}.csv" \
            --write-output="${RESULT_ROOT}/VectorMaton/${DATASET}/${s}.queries" \
            > "${RESULT_ROOT}/VectorMaton/${DATASET}/${s}" &
    fi

    if should_run "BM25Filtering"; then
        ./build/main \
            "${STRINGS_FILE}" "${VECTORS_FILE}" \
            "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" \
            BM25Filtering \
            "$(index_flag BM25Filtering "$s")" \
            --statistics-file="${RESULT_ROOT}/BM25Filtering/${DATASET}/${s}.csv" \
            --write-output="${RESULT_ROOT}/BM25Filtering/${DATASET}/${s}.queries" \
            > "${RESULT_ROOT}/BM25Filtering/${DATASET}/${s}" &
    fi

    wait

    run_acorn_variant "ACORN-gamma" 32 64 "$s"
    run_acorn_variant "ACORN-1" 1 32 "$s"

    if should_run "pgvector"; then
        if [ "$s" -eq "$FIRST_PATTERN_LENGTH" ]; then
            python3 test_pgvector.py \
                "${STRINGS_FILE}" "${VECTORS_FILE}" \
                "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" "${GROUND_TRUTH}" \
                --rebuild \
                --write-output "${RESULT_ROOT}/pgvector/${DATASET}/${s}.queries" \
                > "${RESULT_ROOT}/pgvector/${DATASET}/${s}" 2>&1
        else
            python3 test_pgvector.py \
                "${STRINGS_FILE}" "${VECTORS_FILE}" \
                "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" "${GROUND_TRUTH}" \
                --write-output "${RESULT_ROOT}/pgvector/${DATASET}/${s}.queries" \
                > "${RESULT_ROOT}/pgvector/${DATASET}/${s}" 2>&1
        fi
        mv pgvector_hnsw_stats.csv "${RESULT_ROOT}/pgvector/${DATASET}/${s}.csv"
    fi

    if should_run "ElasticSearch"; then
        if [ "$s" -eq "$FIRST_PATTERN_LENGTH" ]; then
            python3 test_elasticsearch.py \
                "${STRINGS_FILE}" "${VECTORS_FILE}" \
                "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" "${GROUND_TRUTH}" \
                --rebuild ${ELASTICSEARCH_PID_ARG} \
                --write-output "${RESULT_ROOT}/ElasticSearch/${DATASET}/${s}.queries" \
                > "${RESULT_ROOT}/ElasticSearch/${DATASET}/${s}" 2>&1
        else
            python3 test_elasticsearch.py \
                "${STRINGS_FILE}" "${VECTORS_FILE}" \
                "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}" "${GROUND_TRUTH}" \
                ${ELASTICSEARCH_PID_ARG} \
                --write-output "${RESULT_ROOT}/ElasticSearch/${DATASET}/${s}.queries" \
                > "${RESULT_ROOT}/ElasticSearch/${DATASET}/${s}" 2>&1
        fi
        mv elasticsearch_hnsw_stats.csv "${RESULT_ROOT}/ElasticSearch/${DATASET}/${s}.csv"
    fi
done

echo "==> Finished long-sequence benchmark"
echo "Results saved under: ${RESULT_ROOT}/*/${DATASET}"

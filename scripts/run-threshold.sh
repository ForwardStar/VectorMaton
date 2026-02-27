#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

DATASET_DIR="${DATASET_DIR:-datasets/arxiv-small}"
RESULT_ROOT="${RESULT_ROOT:-results/threshold}"
QUERY_STRING_LEN="${QUERY_STRING_LEN:-2}"
NUM_QUERIES="${NUM_QUERIES:-1000}"
K_VALUE="${K_VALUE:-10}"
TRUNCATE_LEN="${TRUNCATE_LEN:--1}"
THRESHOLDS="${THRESHOLDS:-100 200 400 800 1600 3200}"
METHOD="VectorMaton-smart"

STRINGS_FILE="${DATASET_DIR}/strings.txt"
VECTORS_FILE="${DATASET_DIR}/vectors.txt"
DATASET_NAME=$(basename "${DATASET_DIR}")

if [ ! -x "./build/main" ]; then
    echo "Missing executable ./build/main (build the project first)." >&2
    exit 1
fi

if [ ! -f "${STRINGS_FILE}" ] || [ ! -f "${VECTORS_FILE}" ]; then
    echo "Missing dataset files under ${DATASET_DIR} (expected strings.txt and vectors.txt)." >&2
    exit 1
fi

mkdir -p "${RESULT_ROOT}"

RUN_ROOT="${RESULT_ROOT}/${METHOD}/${DATASET_NAME}/len2to4_k${K_VALUE}_q${NUM_QUERIES}"
QUERY_DIR="${RUN_ROOT}/queries"
mkdir -p "${QUERY_DIR}"

echo "==> Generating mixed-length queries (${NUM_QUERIES}, |p| in [2,4], k=${K_VALUE}) on ${DATASET_NAME}"
(
    cd "${QUERY_DIR}"
    python3 "${REPO_ROOT}/scripts/generate_queries.py" \
        "${REPO_ROOT}/${STRINGS_FILE}" \
        "${REPO_ROOT}/${VECTORS_FILE}" \
        "${QUERY_STRING_LEN}" \
        "${NUM_QUERIES}" \
        "${K_VALUE}" \
        "${TRUNCATE_LEN}" \
        threshold \
        --mixed-length
)

QUERY_STRINGS="${QUERY_DIR}/strings_threshold.txt"
QUERY_VECTORS="${QUERY_DIR}/vectors_threshold.txt"
QUERY_K="${QUERY_DIR}/k_threshold.txt"

for f in "${QUERY_STRINGS}" "${QUERY_VECTORS}" "${QUERY_K}"; do
    if [ ! -f "${f}" ]; then
        echo "Query generation failed: missing ${f}" >&2
        exit 1
    fi
done

SUMMARY_CSV="${RUN_ROOT}/summary.csv"
printf "threshold,exact_baseline_avg_us,exact_baseline_qps,max_recall,max_recall_qps,max_recall_time_us\n" > "${SUMMARY_CSV}"

run_threshold() {
    threshold="$1"
    OUT_DIR="${RUN_ROOT}/threshold_${threshold}"
    INDEX_DIR="${OUT_DIR}/index"
    LOG_FILE="${OUT_DIR}/run.log"
    STATS_FILE="${OUT_DIR}/stats.csv"
    RECALL_QPS_FILE="${OUT_DIR}/recall_qps.csv"
    SUMMARY_ROW_FILE="${OUT_DIR}/summary_row.csv"

    mkdir -p "${OUT_DIR}"

    echo "==> Running ${METHOD} with --set-min-build-threshold=${threshold}"
    ./build/main \
        "${STRINGS_FILE}" \
        "${VECTORS_FILE}" \
        "${QUERY_STRINGS}" \
        "${QUERY_VECTORS}" \
        "${QUERY_K}" \
        "${METHOD}" \
        "--set-min-build-threshold=${threshold}" \
        "--statistics-file=${STATS_FILE}" \
        "--save-index=${INDEX_DIR}" \
        > "${LOG_FILE}" 2>&1

    if [ ! -f "${STATS_FILE}" ]; then
        echo "Missing statistics output: ${STATS_FILE}" >&2
        exit 1
    fi

    awk -F, '
        BEGIN { OFS="," }
        NR==1 { print "ef_search,recall,qps,time_us"; next }
        NR>1 && $2+0 > 0 { printf "%s,%s,%.10g,%s\n", $1, $3, 1000000/($2+0), $2 }
    ' "${STATS_FILE}" > "${RECALL_QPS_FILE}"

    exact_avg_us=$(grep "ExactSearch query processing took" "${LOG_FILE}" | sed -n 's/.*avg (us): \([0-9.eE+-][0-9.eE+-]*\).*/\1/p' | tail -n 1 || true)
    if [ -n "${exact_avg_us}" ]; then
        exact_qps=$(awk -v t="${exact_avg_us}" 'BEGIN { if (t > 0) printf "%.10g", 1000000/t; else printf "" }')
    else
        exact_qps=""
    fi

    max_line=$(awk -F, '
        NR==2 { best_recall=$3+0; best_time=$2+0 }
        NR>2 {
            r=$3+0; t=$2+0
            if (r > best_recall || (r == best_recall && t < best_time)) {
                best_recall=r; best_time=t
            }
        }
        END {
            if (NR >= 2) {
                qps = (best_time > 0 ? 1000000/best_time : 0)
                printf "%.10g,%.10g,%.10g", best_recall, qps, best_time
            }
        }
    ' "${STATS_FILE}")

    max_recall=$(printf "%s" "${max_line}" | cut -d, -f1)
    max_recall_qps=$(printf "%s" "${max_line}" | cut -d, -f2)
    max_recall_time_us=$(printf "%s" "${max_line}" | cut -d, -f3)

    printf "%s,%s,%s,%s,%s,%s\n" \
        "${threshold}" \
        "${exact_avg_us}" \
        "${exact_qps}" \
        "${max_recall}" \
        "${max_recall_qps}" \
        "${max_recall_time_us}" \
        > "${SUMMARY_ROW_FILE}"
}

failed=0
pids=""

for threshold in ${THRESHOLDS}; do
    run_threshold "${threshold}" &
    pids="${pids} $!"
done

for pid in ${pids}; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if [ "${failed}" -ne 0 ]; then
    echo "One or more threshold runs failed." >&2
    exit 1
fi

for threshold in ${THRESHOLDS}; do
    row_file="${RUN_ROOT}/threshold_${threshold}/summary_row.csv"
    if [ ! -f "${row_file}" ]; then
        echo "Missing summary row: ${row_file}" >&2
        exit 1
    fi
    cat "${row_file}" >> "${SUMMARY_CSV}"
done

echo "==> Finished"
echo "Results saved under: ${RUN_ROOT}"
echo "Summary: ${SUMMARY_CSV}"

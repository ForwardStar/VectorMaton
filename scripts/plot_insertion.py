import argparse
import os
import re

import pandas as pd

from plot_memory_consumption import (
    DATASET_LABELS,
    DATASETS,
    METHOD_HATCHES,
    HandlerOverlayPatch,
    configure_metric_axis,
    draw_metric_bars,
    max_observed,
    method_colors,
    method_label,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHODS = [
    "OptQuery",
    "PreFiltering",
    "PostFiltering",
    "Hybrid",
    "pgvector",
    "ElasticSearch",
    "BM25Filtering",
    "VectorMaton",
]

LOG_INSERTION_AVG_PATTERN = re.compile(r"Insertion took\s*\d+\s*(?:μs|us).*?avg \(us\):\s*([0-9.eE+-]+)")
CSV_INSERTION_COLUMNS = [
    ("average_insertion_time_us", 1.0),
    ("insertion_time_us", 1.0),
    ("index_insertion_time_us", 1.0),
    ("average_insertion_time_s", 1_000_000.0),
    ("insertion_time_s", 1_000_000.0),
    ("index_insertion_time_s", 1_000_000.0),
]


def result_method_dirs(method):
    if method == "VectorMaton":
        return ["VectorMaton-smart", "VectorMaton"]
    return [method]


def load_log_average_insertion_time_us(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    matches = LOG_INSERTION_AVG_PATTERN.findall(content)
    if not matches:
        return None

    return float(matches[-1])


def load_csv_average_insertion_time_us(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    for column, multiplier in CSV_INSERTION_COLUMNS:
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue

        return float(values.iloc[0]) * multiplier

    return None


def load_average_insertion_time_us(method, dataset, insertion_percentage, result_root):
    tag = f"insert_{insertion_percentage}"
    for result_method in result_method_dirs(method):
        csv_path = os.path.join(result_root, result_method, dataset, f"{tag}.csv")
        insertion_time_us = load_csv_average_insertion_time_us(csv_path)
        if insertion_time_us is not None:
            return insertion_time_us

        log_path = os.path.join(result_root, result_method, dataset, f"{tag}.log")
        insertion_time_us = load_log_average_insertion_time_us(log_path)
        if insertion_time_us is not None:
            return insertion_time_us

    return None


def plot_insertion(insertion_percentage, result_root, output):
    colors = method_colors()
    insertion_results = {
        method: [
            load_average_insertion_time_us(method, dataset, insertion_percentage, result_root)
            for dataset in DATASETS
        ]
        for method in METHODS
    }

    group_gap = 1.2
    x = [i * group_gap for i in range(len(DATASETS))]
    bar_width = 0.10

    fig, ax = plt.subplots(1, 1, figsize=(18, 4.5))

    axis_top_us = max_observed(insertion_results, 1.0) * 1.8
    draw_metric_bars(
        ax,
        insertion_results,
        1.0,
        axis_top_us,
        bar_width,
        x,
        colors,
        METHODS,
        show_optquery_na=True,
        show_pgvector_words_na=True,
    )
    configure_metric_axis(ax, "Time (us)", axis_top_us, bar_width, x, METHODS)
    ax.set_xticklabels(DATASET_LABELS, fontsize=35)

    fig.legend(
        handles=[
            (
                Patch(facecolor="none", edgecolor="black", linewidth=2.4),
                Patch(facecolor="none", edgecolor=colors[method], linewidth=0.1, hatch=METHOD_HATCHES[method]),
            )
            for method in METHODS
        ],
        labels=[method_label(method) for method in METHODS],
        loc="upper center",
        ncol=4,
        fontsize=30,
        handler_map={tuple: HandlerOverlayPatch()},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.68])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output)
    print(f"Saved {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot average index insertion time for all non-ACORN methods and datasets.")
    parser.add_argument("--insert-percentage", default="20", help="Insertion percentage result tag to plot.")
    parser.add_argument("--result-root", default="results/insertion", help="Insertion result root.")
    parser.add_argument("--output", default="figures/insertion.pdf", help="Output figure path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_insertion(args.insert_percentage, args.result_root, args.output)

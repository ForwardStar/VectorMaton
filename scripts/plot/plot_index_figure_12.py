import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
import pandas as pd


plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams["hatch.linewidth"] = 2.8
for font in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
    if "Libertine_R" in font:
        font_prop = fm.FontProperties(fname=font)
        font_name = font_prop.get_name()
        plt.rcParams.update({"font.family": font_name})
        rcParams["font.family"] = font_name
        rcParams["mathtext.fontset"] = "custom"
        rcParams["mathtext.rm"] = font_name


DATASETS = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]
DATASET_LABELS = ["spam", "words", "mtg", "arxiv", "prot", "code"]
METHODS = ["OptQuery", "PreFiltering", "PostFiltering", "Hybrid", "ACORN-1", "ACORN-gamma", "pgvector", "ElasticSearch", "BM25Filtering", "VectorMaton"]
METHOD_LABELS = {"ACORN-gamma": "ACORN-γ"}
METHOD_HATCHES = {
    "OptQuery": "\\",
    "PreFiltering": "",
    "PostFiltering": "+",
    "Hybrid": "o",
    "ACORN-1": "*",
    "ACORN-gamma": "x",
    "pgvector": "-",
    "ElasticSearch": "/",
    "BM25Filtering": "|",
    "VectorMaton": ".",
}


def method_label(method):
    return METHOD_LABELS.get(method, method)

PEAK_MEMORY_PATTERN = re.compile(r"peak memory consumption:\s*(\d+)\s*bytes")
INDEX_BUILD_TIME_PATTERN = re.compile(r"index built took\s*(\d+)\s*(?:μs|us)")
CSV_BUILD_TIME_COLUMNS = [
    ("build_time_us", 1_000_000.0),
    ("index_build_time_us", 1_000_000.0),
    ("index_build_elapsed_us", 1_000_000.0),
    ("build_time_s", 1.0),
    ("build_time_seconds", 1.0),
    ("index_build_time_s", 1.0),
    ("index_build_time_seconds", 1.0),
]


class HandlerOverlayPatch(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        base_proto, hatch_proto = orig_handle
        artists = []
        for proto in (base_proto, hatch_proto):
            patch = Rectangle(
                (xdescent, ydescent),
                width,
                height,
                facecolor=proto.get_facecolor(),
                edgecolor=proto.get_edgecolor(),
                hatch=proto.get_hatch(),
                linewidth=proto.get_linewidth(),
            )
            patch.set_transform(trans)
            artists.append(patch)
        return artists


def method_colors():
    # Match recall_qps.py: methods there use tab10 by method index, with PreFiltering drawn black.
    cs = plt.colormaps["tab10"]
    return {
        "PreFiltering": "black",
        "OptQuery": cs(0),
        "PostFiltering": cs(1),
        "Hybrid": cs(2),
        "ACORN-1": cs(3),
        "ACORN-gamma": cs(4),
        "pgvector": cs(5),
        "ElasticSearch": cs(6),
        "BM25Filtering": cs(7),
        "VectorMaton": cs(8),
    }


def load_log_memory_bytes(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    matches = PEAK_MEMORY_PATTERN.findall(content)
    if not matches:
        return None

    return int(matches[-1])


def load_csv_memory_bytes(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "build_peak_memory_bytes" not in df.columns or df.empty:
        return None

    values = pd.to_numeric(df["build_peak_memory_bytes"], errors="coerce").dropna()
    if values.empty:
        return None

    return int(values.iloc[0])


def load_log_build_time_seconds(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    matches = INDEX_BUILD_TIME_PATTERN.findall(content)
    if not matches:
        return None

    return int(matches[-1]) / 1_000_000.0


def load_csv_build_time_seconds(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    for column, divisor in CSV_BUILD_TIME_COLUMNS:
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue

        return float(values.iloc[0]) / divisor

    return None


def load_memory_bytes(method, dataset, pattern_length):
    if method in {"ACORN-1", "ACORN-gamma", "pgvector", "ElasticSearch"}:
        path = os.path.join("results", method, dataset, f"{pattern_length}.csv")
        return load_csv_memory_bytes(path)

    path = os.path.join("results", method, dataset, str(pattern_length))
    return load_log_memory_bytes(path)


def load_build_time_seconds(method, dataset, pattern_length):
    log_path = os.path.join("results", method, dataset, str(pattern_length))
    build_time_seconds = load_log_build_time_seconds(log_path)
    if build_time_seconds is not None:
        return build_time_seconds

    csv_path = os.path.join("results", method, dataset, f"{pattern_length}.csv")
    return load_csv_build_time_seconds(csv_path)


def draw_bars(ax, xs, ys, bar_width, hatch, color, label):
    # Match memory_and_time.py: black outline plus colored hatch overlay.
    ax.bar(
        xs,
        ys,
        width=bar_width,
        label=label,
        facecolor="none",
        edgecolor="black",
        linewidth=2.4,
    )
    ax.bar(
        xs,
        ys,
        width=bar_width,
        facecolor="none",
        edgecolor=color,
        hatch=hatch,
        linewidth=0.1,
    )


def max_observed(results, scale):
    max_value = 0
    for vals in results.values():
        for v in vals:
            if v is not None:
                max_value = max(max_value, v / scale)
    return max_value if max_value > 0 else 1


def draw_metric_bars(
    ax,
    results,
    value_scale,
    axis_top,
    bar_width,
    x,
    colors,
    methods,
    show_oom=False,
    show_optquery_na=False,
    show_pgvector_words_na=False,
):
    offset_center = (len(methods) - 1) / 2
    oom_target_datasets = {"mtg", "arxiv-small", "swissprot", "code_search_net"}
    oom_positions = []
    na_positions = []

    for i, method in enumerate(methods):
        xs, ys = [], []
        for j, value in enumerate(results[method]):
            dataset = DATASETS[j]
            x_pos = x[j] + (i - offset_center) * bar_width

            if value is None:
                if show_oom and method == "OptQuery" and dataset in oom_target_datasets:
                    xs.append(x_pos)
                    ys.append(axis_top)
                    oom_positions.append(x_pos)
                elif show_optquery_na and method == "OptQuery" and dataset in oom_target_datasets:
                    na_positions.append(x_pos)
                elif show_pgvector_words_na and method == "pgvector" and dataset == "words":
                    na_positions.append(x_pos)
                else:
                    continue
            else:
                xs.append(x_pos)
                ys.append(value / value_scale)

        draw_bars(ax, xs, ys, bar_width, METHOD_HATCHES[method], colors[method], method)

    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    for ox in oom_positions:
        ax.text(ox, 0.98, "OOM", ha="center", va="top", fontsize=30, fontweight="bold", color="red", transform=text_transform)
    for nx in na_positions:
        ax.text(nx, 0.01, "N/A", ha="center", va="bottom", fontsize=22, fontweight="bold", color="red", rotation=270, transform=text_transform)


def configure_metric_axis(ax, ylabel, axis_top, bar_width, x, methods):
    offset_center = (len(methods) - 1) / 2
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, fontsize=25)
    ax.tick_params(axis="y", labelsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    # ax.set_xlabel("Dataset", fontsize=45, fontweight="bold")
    ax.set_yscale("log")
    x_margin = bar_width * 0.25
    ax.set_xlim(x[0] - bar_width * (offset_center + 0.5) - x_margin, x[-1] + bar_width * (offset_center + 0.5) + x_margin)
    ax.set_ylim(top=axis_top)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)


def plot_memory(pattern_length, output):
    colors = method_colors()
    memory_results = {
        method: [load_memory_bytes(method, dataset, pattern_length) for dataset in DATASETS]
        for method in METHODS
    }
    time_results = {
        method: [load_build_time_seconds(method, dataset, pattern_length) for dataset in DATASETS]
        for method in METHODS
    }

    group_gap = 1.2
    x = [i * group_gap for i in range(len(DATASETS))]
    bar_width = 0.10
    memory_methods = [method for method in METHODS if method != "ElasticSearch"]

    fig, axes = plt.subplots(1, 2, figsize=(36, 4))
    ax_memory, ax_time = axes

    axis_top_mb = max_observed(memory_results, 1024 * 1024) * 1.8
    axis_top_time_s = max_observed(time_results, 1.0) * 1.8

    draw_metric_bars(ax_memory, memory_results, 1024 * 1024, axis_top_mb, bar_width, x, colors, memory_methods, show_oom=True, show_pgvector_words_na=True)
    configure_metric_axis(ax_memory, "Memory (MB)", axis_top_mb, bar_width, x, memory_methods)

    draw_metric_bars(ax_time, time_results, 1.0, axis_top_time_s, bar_width, x, colors, METHODS, show_optquery_na=True, show_pgvector_words_na=True)
    configure_metric_axis(ax_time, "Time (s)", axis_top_time_s, bar_width, x, METHODS)

    fig.legend(
        handles=[
            (
                Patch(facecolor="none", edgecolor="black", linewidth=2.4),
                Patch(facecolor="none", edgecolor=colors[method], linewidth=0.1, hatch=METHOD_HATCHES[method]),
            )
            for i, method in enumerate(METHODS)
        ],
        labels=[method_label(method) for method in METHODS],
        loc="upper center",
        ncol=10,
        fontsize=30,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=2,
        handler_map={tuple: HandlerOverlayPatch()},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.8])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output)
    print(f"Saved {output}")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot build peak memory consumption for all methods and datasets.")
    parser.add_argument("--pattern-length", type=int, default=2, help="Query pattern length result to plot.")
    parser.add_argument("--output", default="figures/memory_consumption.pdf", help="Output figure path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_memory(args.pattern_length, args.output)

import argparse
import os
import re

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
METHODS = ["OptQuery", "PreFiltering", "PostFiltering", "Hybrid", "ACORN-1", "ACORN-gamma", "pgvector", "VectorMaton"]
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
    "VectorMaton": ".",
}


def method_label(method):
    return METHOD_LABELS.get(method, method)

PEAK_MEMORY_PATTERN = re.compile(r"peak memory consumption:\s*(\d+)\s*bytes")


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
        "VectorMaton": cs(7),
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


def load_memory_bytes(method, dataset, pattern_length):
    if method in {"ACORN-1", "ACORN-gamma", "pgvector", "ElasticSearch"}:
        path = os.path.join("results", method, dataset, f"{pattern_length}.csv")
        return load_csv_memory_bytes(path)

    path = os.path.join("results", method, dataset, str(pattern_length))
    return load_log_memory_bytes(path)


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


def plot_memory(pattern_length, output):
    colors = method_colors()
    results = {
        method: [load_memory_bytes(method, dataset, pattern_length) for dataset in DATASETS]
        for method in METHODS
    }

    group_gap = 1.35
    x = [i * group_gap for i in range(len(DATASETS))]
    bar_width = 0.12
    offset_center = (len(METHODS) - 1) / 2

    fig, ax = plt.subplots(1, 1, figsize=(24, 8))

    # Determine maximum observed memory (MB) to use for the y-axis.
    max_mb = 0
    for vals in results.values():
        for v in vals:
            if v is not None:
                max_mb = max(max_mb, v / (1024 * 1024))
    if max_mb <= 0:
        max_mb = 1
    axis_top_mb = max_mb * 1.8

    # datasets where OptQuery should be shown as OOM when missing
    oom_target_datasets = {"mtg", "arxiv-small", "swissprot", "code_search_net"}
    oom_positions = []

    for i, method in enumerate(METHODS):
        xs, ys = [], []
        for j, memory_bytes in enumerate(results[method]):
            dataset = DATASETS[j]
            if memory_bytes is None:
                # For OptQuery on certain datasets, mark OOM by plotting at max
                if method == "OptQuery" and dataset in oom_target_datasets:
                    xs.append(x[j] + (i - offset_center) * bar_width)
                    ys.append(axis_top_mb)
                    oom_positions.append(xs[-1])
                else:
                    continue
            else:
                xs.append(x[j] + (i - offset_center) * bar_width)
                ys.append(memory_bytes / (1024 * 1024))

        draw_bars(ax, xs, ys, bar_width, METHOD_HATCHES[method], colors[method], method)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, fontsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.set_ylabel("Peak build memory (MB)", fontsize=35)
    ax.set_xlabel("Dataset", fontsize=35, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(top=axis_top_mb)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

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
        ncol=4,
        fontsize=35,
        handler_map={tuple: HandlerOverlayPatch()},
    )

    # Annotate OOM bars
    oom_text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    for ox in oom_positions:
        ax.text(ox, 0.98, "OOM", ha="center", va="top", fontsize=24, fontweight="bold", color="red", transform=oom_text_transform)

    plt.tight_layout(rect=[0, 0, 1, 0.75])
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

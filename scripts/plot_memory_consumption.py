import argparse
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
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
METHODS = ["OptQuery", "PreFiltering", "PostFiltering", "pgvector", "ElasticSearch", "VectorMaton"]

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
        "pgvector": cs(2),
        "ElasticSearch": cs(3),
        "VectorMaton": cs(4),
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
    if method in {"pgvector", "ElasticSearch"}:
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
    hatches = ["\\", "", "+", "x", "-", "/"]
    results = {
        method: [load_memory_bytes(method, dataset, pattern_length) for dataset in DATASETS]
        for method in METHODS
    }

    x = list(range(len(DATASETS)))
    bar_width = 0.12
    offset_center = (len(METHODS) - 1) / 2

    fig, ax = plt.subplots(1, 1, figsize=(22, 7))

    for i, method in enumerate(METHODS):
        xs, ys = [], []
        for j, memory_bytes in enumerate(results[method]):
            if memory_bytes is None:
                continue
            xs.append(j + (i - offset_center) * bar_width)
            ys.append(memory_bytes / (1024 * 1024))

        draw_bars(ax, xs, ys, bar_width, hatches[i], colors[method], method)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, fontsize=25)
    ax.tick_params(axis="y", labelsize=25)
    ax.set_ylabel("Peak build memory (MB)", fontsize=30)
    ax.set_xlabel("Dataset", fontsize=30, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    fig.legend(
        handles=[
            (
                Patch(facecolor="none", edgecolor="black", linewidth=2.4),
                Patch(facecolor="none", edgecolor=colors[method], linewidth=0.1, hatch=hatches[i]),
            )
            for i, method in enumerate(METHODS)
        ],
        labels=METHODS,
        loc="upper center",
        ncol=6,
        fontsize=28,
        handler_map={tuple: HandlerOverlayPatch()},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.84])
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

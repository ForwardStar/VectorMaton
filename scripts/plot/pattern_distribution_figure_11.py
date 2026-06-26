#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import rcParams


plt.rcParams.update({"font.family": "Times New Roman"})
for font in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
    if "Libertine_R" in font:
        font_prop = fm.FontProperties(fname=font)
        font_name = font_prop.get_name()
        plt.rcParams.update({"font.family": font_name})
        rcParams["font.family"] = font_name
        rcParams["mathtext.fontset"] = "custom"
        rcParams["mathtext.rm"] = font_name


def read_lines(path):
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def query_selectivities(strings_path, queries_path, pattern_length):
    queries = read_lines(queries_path)
    query_patterns = set(queries)
    doc_counts = {pattern: 0 for pattern in query_patterns}
    num_strings = 0

    with strings_path.open("r", encoding="utf-8") as f:
        for line in f:
            num_strings += 1
            string = line.rstrip("\n")
            if len(string) < pattern_length:
                continue

            seen = set()
            for start in range(len(string) - pattern_length + 1):
                pattern = string[start:start + pattern_length]
                if pattern in query_patterns:
                    seen.add(pattern)

            for pattern in seen:
                doc_counts[pattern] += 1

    return [doc_counts[query] / num_strings for query in queries]


def save_pattern_distribution(dataset_dir, query_dir, output_path, pattern_lengths):
    dataset_dir = Path(dataset_dir)
    query_dir = Path(query_dir)
    output_path = Path(output_path)
    strings_path = dataset_dir / "strings.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(pattern_lengths), figsize=(3.2 * len(pattern_lengths), 2))
    if len(pattern_lengths) == 1:
        axes = [axes]

    for column, (ax, pattern_length) in enumerate(zip(axes, pattern_lengths)):
        queries_path = query_dir / str(pattern_length) / "strings.txt"
        selectivities = query_selectivities(strings_path, queries_path, pattern_length)
        ax.hist(selectivities, bins=30, color="#4C78A8", alpha=0.75)
        ax.set_title(f"|p| = {pattern_length}", fontsize=15)
        ax.set_xlabel("Selectivity", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)

        cdf_ax = ax.twinx()
        sorted_selectivities = sorted(selectivities)
        cdf_values = [(index + 1) / len(sorted_selectivities) for index in range(len(sorted_selectivities))]
        cdf_ax.plot(sorted_selectivities, cdf_values, color="#F58518", linewidth=2)
        cdf_ax.set_ylim(0, 1)
        cdf_ax.tick_params(axis="y", labelsize=12, colors="#F58518")
        cdf_ax.spines["right"].set_color("#F58518")
        if column == len(pattern_lengths) - 1:
            cdf_ax.set_ylabel("CDF", fontsize=15, color="#F58518")

    axes[0].set_ylabel("No. of queries", fontsize=15, color="#4C78A8")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot arxiv-small query selectivity distributions.")
    parser.add_argument("--dataset-dir", default="datasets/arxiv-small")
    parser.add_argument("--query-dir", default="queries/arxiv-small")
    parser.add_argument("--output", default="figures/pattern_distribution.pdf")
    parser.add_argument("--pattern-lengths", "-p", nargs="+", type=int, default=[2, 3, 4])
    return parser.parse_args()


def main():
    args = parse_args()
    save_pattern_distribution(args.dataset_dir, args.query_dir, args.output, args.pattern_lengths)


if __name__ == "__main__":
    main()

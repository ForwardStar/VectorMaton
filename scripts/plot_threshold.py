import argparse
import csv
import os
import re

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


THRESHOLDS = [100, 200, 400, 800, 1600, 3200]
INDEX_SIZE_RE = re.compile(r"Total index size:\s*([0-9]+)\s*bytes")
INDEX_BUILD_TIME_RE = re.compile(r"index built took\s*([0-9]+)\s*(?:μs|us)")
MIN_RECALL = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot threshold-study recall-QPS and index size."
    )
    parser.add_argument(
        "--run-root",
        default="results/threshold/VectorMaton-smart/arxiv-small/len2_k10_q1000",
        help="Root directory that contains threshold_* subdirectories.",
    )
    parser.add_argument(
        "--output",
        default="figures/threshold_study.pdf",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_recall_qps(csv_path):
    if not os.path.exists(csv_path):
        return None
    recalls = []
    qpss = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "recall" not in (reader.fieldnames or []) or "qps" not in (reader.fieldnames or []):
            return None
        for row in reader:
            try:
                recalls.append(float(row["recall"]))
                qpss.append(float(row["qps"]))
            except (TypeError, ValueError, KeyError):
                continue
    if not recalls:
        return None
    filtered = [(r, q) for r, q in zip(recalls, qpss) if r >= MIN_RECALL]
    if not filtered:
        return None
    filtered_recalls = [r for r, _ in filtered]
    filtered_qpss = [q for _, q in filtered]
    return filtered_recalls, filtered_qpss


def load_index_size_bytes(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = INDEX_SIZE_RE.search(content)
    if match is None:
        return None
    return int(match.group(1))


def load_index_build_time_s(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = INDEX_BUILD_TIME_RE.search(content)
    if match is None:
        return None
    return int(match.group(1)) / 1_000_000.0


def main():
    args = parse_args()
    cs = plt.colormaps["tab10"]
    markers = ["o", "s", "^", "d"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    ax_curve, ax_size, ax_time = axes

    plotted_any_curve = False
    bar_labels = []
    bar_values_gb = []
    bar_colors = []
    time_labels = []
    build_times_s = []
    time_colors = []

    for idx, t in enumerate(THRESHOLDS):
        threshold_dir = os.path.join(args.run_root, f"threshold_{t}")
        curve_path = os.path.join(threshold_dir, "recall_qps.csv")
        log_path = os.path.join(threshold_dir, "run.log")

        curve = load_recall_qps(curve_path)
        if curve is not None:
            recall, qps = curve
            ax_curve.plot(
                recall,
                qps,
                marker=markers[idx % len(markers)],
                markersize=7,
                markerfacecolor="none",
                linewidth=2,
                color=cs(idx),
                label=f"T={t}",
            )
            plotted_any_curve = True

        index_size_bytes = load_index_size_bytes(log_path)
        if index_size_bytes is not None:
            bar_labels.append(f"T={t}")
            bar_values_gb.append(index_size_bytes / (1024 ** 3))
            bar_colors.append(cs(idx))

        index_build_time_s = load_index_build_time_s(log_path)
        if index_build_time_s is not None:
            time_labels.append(f"T={t}")
            build_times_s.append(index_build_time_s)
            time_colors.append(cs(idx))

    if plotted_any_curve:
        ax_curve.set_xlabel("Recall @ 10", fontsize=25)
        ax_curve.set_ylabel("QPS", fontsize=25)
        ax_curve.set_yscale("log")
        ax_curve.grid(True, linestyle="--", alpha=0.7)
        ax_curve.tick_params(axis="both", labelsize=20)
        ax_curve.set_title("Recall-QPS Curves", fontsize=25, fontweight="bold")
    else:
        ax_curve.text(
            0.5,
            0.5,
            "No valid recall_qps.csv files found",
            ha="center",
            va="center",
            transform=ax_curve.transAxes,
            fontsize=14,
        )
        ax_curve.set_axis_off()

    if bar_labels:
        ax_size.bar(bar_labels, bar_values_gb, color=bar_colors, edgecolor="black", alpha=0.9)
    ax_size.set_xlabel("Threshold", fontsize=25)
    ax_size.set_ylabel("Index size (GB)", fontsize=25)
    ax_size.tick_params(axis="both", labelsize=20)
    ax_size.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax_size.set_title("Index size by T", fontsize=25, fontweight="bold")

    if time_labels:
        ax_time.bar(time_labels, build_times_s, color=time_colors, edgecolor="black", alpha=0.9)
    ax_time.set_xlabel("Threshold", fontsize=25)
    ax_time.set_ylabel("Index time (s)", fontsize=25)
    ax_time.tick_params(axis="both", labelsize=20)
    ax_time.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax_time.set_title("Index time by T", fontsize=25, fontweight="bold")

    handles, labels = ax_curve.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=6,
            fontsize=30,
            handlelength=1.2,
            markerscale=1.5
            # bbox_to_anchor=(0.5, 1.08),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.8])
    else:
        plt.tight_layout()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()

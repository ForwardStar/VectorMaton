import argparse
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
RATIO_LABELS = {
    0.2: "20%",
    0.4: "40%",
    0.6: "60%",
    0.8: "80%",
    1.0: "100%",
}
DATASETS = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]
DATASET_TITLES = ["spam", "words", "mtg", "arxiv", "prot", "code"]


def configure_fonts():
    plt.rcParams.update({"font.family": "Times New Roman"})
    for font in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        if "Libertine_R" in font:
            font_prop = fm.FontProperties(fname=font)
            font_name = font_prop.get_name()
            plt.rcParams.update({"font.family": font_name})
            rcParams["font.family"] = font_name
            rcParams["mathtext.fontset"] = "custom"
            rcParams["mathtext.rm"] = font_name
            break


def load_parameter_study_curve(csv_path, ratio):
    df = pd.read_csv(csv_path)
    required_columns = {"search_k_ratio", "time_us", "recall"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    ratio_df = df[np.isclose(df["search_k_ratio"], ratio)].copy()
    ratio_df = ratio_df[(ratio_df["time_us"] > 0) & (ratio_df["recall"] > 0)]
    if ratio_df.empty:
        return None, None

    recall = ratio_df["recall"].to_numpy()
    qps = 1_000_000 / ratio_df["time_us"].to_numpy()
    order = np.argsort(recall)
    return recall[order], qps[order]


def plot_dataset(ax, dataset, title, input_root, query_length):
    csv_path = os.path.join(input_root, dataset, f"{query_length}.csv")
    markers = ["o", "s", "^", "d", "v"]
    colors = [plt.colormaps["tab10"](i) for i in range(len(RATIOS))]
    plotted = False

    if not os.path.exists(csv_path):
        ax.text(0.5, 0.5, "missing data", ha="center", va="center", fontsize=18)
    else:
        for i, ratio in enumerate(RATIOS):
            recall, qps = load_parameter_study_curve(csv_path, ratio)
            if recall is None:
                continue

            ax.plot(
                recall,
                qps,
                marker=markers[i],
                color=colors[i],
                label=RATIO_LABELS[ratio],
                markersize=10,
                markerfacecolor="none",
                linewidth=2,
            )
            plotted = True

    ax.set_title(title, fontsize=24, fontweight="bold")
    ax.set_xlabel("Recall @ 10", fontsize=22)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=18)
    if not plotted:
        ax.set_xlim(0, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PostFiltering parameter-study recall/QPS curves for all datasets."
    )
    parser.add_argument(
        "--input-root",
        default=os.path.join("results", "PostFiltering", "parameter_study"),
        help="Directory containing <dataset>/<query_length>.csv parameter-study files.",
    )
    parser.add_argument(
        "--query-length",
        default="3",
        help="Query substring length file to plot, for example 2, 3, 4, 8, 16, or 32.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("figures", "parameter_study_recall_qps.pdf"),
        help="Output figure path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_fonts()

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(30, 4.5), sharey=False)
    for i, dataset in enumerate(DATASETS):
        plot_dataset(axes[i], dataset, DATASET_TITLES[i], args.input_root, args.query_length)
        if i == 0:
            axes[i].set_ylabel("QPS", fontsize=22)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=5,
            fontsize=24,
            handlelength=1.2,
            markerscale=1.4,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.83])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output)

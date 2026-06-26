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
SELECTED_DATASETS = ["arxiv-small", "swissprot", "code_search_net"]
SELECTED_DATASET_TITLES = ["arxiv", "prot", "code"]
MIN_RECALL_BY_DATASET = {
    "spam": 0.2,
    "words": 0.2,
    "mtg": 0.5,
    "arxiv-small": 0.5,
    "swissprot": 0.4,
    "code_search_net": 0.4,
}
RESULT_FILE = "mixed_2_4.csv"


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


def load_postfiltering_case_study_curve(csv_path, ratio, min_recall):
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None, None
    required_columns = {"search_k_ratio", "time_us", "recall"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    ratio_df = df[np.isclose(df["search_k_ratio"], ratio)].copy()
    ratio_df = ratio_df[(ratio_df["time_us"] > 0) & (ratio_df["recall"] >= min_recall)]
    if ratio_df.empty:
        return None, None

    recall = ratio_df["recall"].to_numpy()
    qps = 1_000_000 / ratio_df["time_us"].to_numpy()
    order = np.argsort(recall)
    return recall[order], qps[order]


def plot_dataset(ax, dataset, title, input_root):
    csv_path = os.path.join(input_root, dataset, RESULT_FILE)
    min_recall = MIN_RECALL_BY_DATASET[dataset]
    markers = ["o", "s", "^", "d", "v"]
    colors = [plt.colormaps["tab10"](i) for i in range(len(RATIOS))]
    plotted = False

    if os.path.exists(csv_path):
        for i, ratio in enumerate(RATIOS):
            recall, qps = load_postfiltering_case_study_curve(csv_path, ratio, min_recall)
            if recall is None:
                continue

            ax.plot(
                recall,
                qps,
                marker=markers[i],
                color=colors[i],
                label=RATIO_LABELS[ratio],
                markersize=12,
                markerfacecolor="none",
                linewidth=2,
            )
            plotted = True

    ax.set_title(title, fontsize=25, fontweight="bold")
    ax.set_xlabel("Recall @ 10", fontsize=25)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=25)
    ax.xaxis.get_offset_text().set_size(25)
    ax.yaxis.get_offset_text().set_size(25)
    if not plotted:
        ax.text(0.5, 0.5, "missing data", ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PostFiltering case-study recall/QPS curves for all datasets."
    )
    parser.add_argument(
        "--input-root",
        default=os.path.join("results", "PostFiltering", "postfiltering_case_study"),
        help="Directory containing <dataset>/mixed_2_4.csv PostFiltering case-study files.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("figures", "postfiltering_case_study_recall_qps.pdf"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--selected-output",
        default=os.path.join("figures", "postfiltering_case_study_recall_qps_arxiv_prot_code.pdf"),
        help="Output path for the one-line arxiv/prot/code figure.",
    )
    return parser.parse_args()


def plot_datasets(datasets, titles, input_root, output, figsize):
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        plot_dataset(axes[i], dataset, titles[i], input_root)
        if i == 0:
            axes[i].set_ylabel("QPS", fontsize=25)

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
            ncol=6,
            fontsize=25,
            handlelength=1.2,
            markerscale=1.5,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.84])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    configure_fonts()

    plot_datasets(DATASETS, DATASET_TITLES, args.input_root, args.output, figsize=(30, 5))
    plot_datasets(
        SELECTED_DATASETS,
        SELECTED_DATASET_TITLES,
        args.input_root,
        args.selected_output,
        figsize=(15, 4),
    )

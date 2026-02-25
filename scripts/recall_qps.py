import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np

plt.rcParams.update({'font.family': 'Times New Roman'})
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if "Libertine_R" in font:
        font_prop = fm.FontProperties(fname=font)
        font_name = font_prop.get_name()
        plt.rcParams.update({'font.family': font_name})
        rcParams['font.family'] = font_name
        rcParams['mathtext.fontset'] = 'custom'
        rcParams['mathtext.rm'] = font_name

def extract_avg_time_us_from_log(filepath):
    """
    Extract the average time (in microseconds) from a log line.

    Expected pattern:
    avg (us): <number>

    Example:
    avg (us): 5.2246e+06
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(
        r"avg\s*\(us\)\s*:\s*([0-9.+\-eE]+)",
        content
    )

    if not match:
        return 1e10
        raise ValueError(f"Could not find avg (us) in file: {filepath}")

    return float(match.group(1))

def load_curve(csv_path):
    """
    Load recall-time curve data from a CSV file.
    Expects columns: time_us, recall
    """
    try:
        df = pd.read_csv(csv_path)

        if "time_us" not in df.columns or "recall" not in df.columns:
            raise ValueError(f"CSV file {csv_path} must contain 'time_us' and 'recall' columns.")

        return df["time_us"].values, df["recall"].values
    except:
        return None, None

def plot_3panel_block(dataset, methods, ds_brief, axes_block, left_block=False):
    # axes_block: list/array of length 3

    for p_len in range(2, 5):
        ax = axes_block[p_len - 2]

        csvs = []
        for method in methods:
            csv_path = os.path.join("results", method, dataset, f"{p_len}.csv")
            csvs.append(csv_path)

        qpss, recalls = [], []
        for csv in csvs:
            time, recall = load_curve(csv)
            qps = None
            if time is not None:
                qps = 1_000_000 / time

            if recall is not None:
                i = 1
                while i < len(recall):
                    if qps[i] >= qps[i - 1]:
                        recall = np.delete(recall, i - 1)
                        qps = np.delete(qps, i - 1)
                    elif recall[i] <= recall[i - 1] + 1e-2:
                        recall = np.delete(recall, i)
                        qps = np.delete(qps, i)
                    else:
                        i += 1

            qpss.append(qps)
            recalls.append(recall)
            # Sort by recall
            if recall is not None and qps is not None:
                sorted_indices = np.argsort(recall)
                recalls[-1] = recall[sorted_indices]
                qpss[-1] = qps[sorted_indices]
        
        # For all methods except VectorMaton, filter out points with recall < (minimum recall of VectorMaton), keep at least half of the points
        min_recall_vectormaton = float('inf')
        for i, method in enumerate(methods):
            if method == "VectorMaton" and recalls[i] is not None:
                min_recall_vectormaton = min(min_recall_vectormaton, np.min(recalls[i]))

        for i, method in enumerate(methods):
            if method != "VectorMaton" and recalls[i] is not None:
                mask = recalls[i] >= min_recall_vectormaton
                if np.sum(mask) > 0:
                    # Keep at least half of the points
                    indices = np.where(mask)[0]
                    if len(indices) < len(recalls[i]) // 2:
                        indices = np.argsort(recalls[i])[-(len(recalls[i]) // 2):]
                    recalls[i] = recalls[i][indices]
                    qpss[i] = qpss[i][indices]

        filepath_prefiltering = os.path.join(
            "results", "PreFiltering", dataset, f"{p_len}"
        )
        time_prefiltering = extract_avg_time_us_from_log(filepath_prefiltering)

        markers = ['o', 's', '^', 'd', 'v', 'x', '*']
        cs = plt.colormaps['tab10']
        colors = [cs(i) for i in range(len(methods))]

        if time_prefiltering is not None:
            qps_pref = 1_000_000 / time_prefiltering
            qps_min = float('inf')
            for qps in qpss:
                if qps is not None and len(qps) > 0:
                    qps_min = min(qps_min, np.min(qps))
            if qps_pref * 10 >= qps_min:
                ax.plot(1.0, qps_pref, marker='*', color='black',
                        label="PreFiltering", markersize=15, markerfacecolor='none')

        for i in range(len(methods)):
            if qpss[i] is not None:
                ax.plot(
                    recalls[i], qpss[i],
                    marker=markers[i],
                    color=colors[i],
                    label=methods[i],
                    markersize=15,
                    markerfacecolor='none'
                )

        ax.set_title(f"{ds_brief}, |p| = {p_len}", fontsize=25, fontweight='bold')
        ax.set_xlabel("Recall @ 10", fontsize=25)
        if left_block and p_len == 2:
            ax.set_ylabel("QPS", fontsize=25)
        ax.set_yscale("log")
        min_qps = float('inf')
        max_qps = float('-inf')
        for qps in qpss:
            if qps is not None and len(qps) > 0:
                min_qps = min(min_qps, np.min(qps))
                max_qps = max(max_qps, np.max(qps))
        if time_prefiltering is not None:
            qps_pref = 1_000_000 / time_prefiltering
            if qps_pref * 10 >= qps_min:
                min_qps = min(min_qps, qps_pref)
                max_qps = max(max_qps, qps_pref)
        ax.set_ylim(bottom=min_qps * 0.5, top=max_qps * 2)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis='both', labelsize=20)

def add_block_caption(fig, axes_block, text, fontsize=26, pad=0.015):
    """
    Place caption centered below a group of 3 subplots.
    pad is the vertical gap in figure coordinates.
    """
    # Bounding box covering all three axes
    x0 = min(ax.get_position().x0 for ax in axes_block)
    x1 = max(ax.get_position().x1 for ax in axes_block)
    y0 = min(ax.get_position().y0 for ax in axes_block)

    x_center = (x0 + x1) / 2
    y = y0 - pad

    fig.text(
        x_center, y,
        text,
        ha="center",
        va="top",
        fontsize=fontsize
    )

if __name__ == "__main__":
    methods = ["OptQuery", "PostFiltering", "pgvector", "elasticsearch", "VectorMaton"]
    datasets = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]
    ds_briefs = ["spam", "words", "mtg", "arxiv", "prot", "code"]

    n = len(datasets)
    fig, axes = plt.subplots(
        n // 2, 6,
        figsize=(28, 2.5 * n),
        sharey=False
    )

    for i, dataset in enumerate(datasets):
        if i % 2 == 0:
            plot_3panel_block(dataset, methods, ds_briefs[i], axes[i // 2, 0:3], left_block=True)
        else:
            plot_3panel_block(dataset, methods, ds_briefs[i], axes[i // 2, 3:6])

        # add_block_caption(fig, left_block,  f"{dataset} (Index A)")
        # add_block_caption(fig, right_block, f"{dataset} (Index B)")


    # Global legend (only once)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=6,
        fontsize=35,
        handlelength=1.2,
        markerscale=1.5
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/recall_qps_all_datasets.pdf")

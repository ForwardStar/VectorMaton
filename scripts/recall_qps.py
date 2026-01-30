import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

plt.rcParams.update({'font.family': 'Times New Roman'})
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if "Libertine_R" in font:
        font_prop = fm.FontProperties(fname=font)
        font_name = font_prop.get_name()
        plt.rcParams.update({'font.family': font_name})
        rcParams['font.family'] = font_name
        rcParams['mathtext.fontset'] = 'custom'
        rcParams['mathtext.rm'] = font_name

def extract_time_us_from_log(filepath):
    """
    Extract the time in microseconds from a log file.
    Expected pattern: took <number>μs
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"took\s+(\d+)\s*μs", content)
    if not match:
        raise ValueError(f"Could not find time in microseconds in file: {filepath}")

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

def plot_dataset_row(dataset, methods, axes_row):
    # axes_row is length-3 array: [ax0, ax1, ax2]

    for p_len in range(2, 5):
        ax = axes_row[p_len - 2]

        csvs = []
        for method in methods:
            csv_path = os.path.join("results", method, dataset, f"{p_len}.csv")
            csvs.append(csv_path)

        qpss = []
        recalls = []
        for csv in csvs:
            time, recall = load_curve(csv)
            qps = None
            if time is not None:
                qps = 1000 / time * 1_000_000

            if recall is not None:
                change = True
                while change:
                    change = False
                    for i in range(1, len(recall)):
                        if recall[i] == recall[i - 1]:
                            new_recall = []
                            new_qps = []
                            for j in range(len(recall)):
                                if j != i:
                                    new_recall.append(recall[j])
                                    new_qps.append(qps[j])
                            recall = new_recall[:]
                            qps = new_qps[:]
                            change = True
                            break

            qpss.append(qps)
            recalls.append(recall)

        # PreFiltering point
        filepath_prefiltering = os.path.join(
            "results", "PreFiltering", dataset, f"{p_len}"
        )
        time_prefiltering = extract_time_us_from_log(filepath_prefiltering)

        markers = ['o', 's', '^', 'd', 'v', 'x', '*']
        cs = plt.colormaps['tab10']
        colors = [cs(i) for i in range(len(methods))]

        for i in range(len(methods)):
            if qpss[i] is not None:
                ax.plot(
                    recalls[i], qpss[i],
                    marker=markers[i],
                    color=colors[i],
                    label=methods[i],
                    markersize=6
                )

        if time_prefiltering is not None:
            least_qps = float('inf')
            for qps in qpss:
                if qps is not None and len(qps) > 0:
                    least_qps = min(least_qps, min(qps))

            qps_pref = 1000 / time_prefiltering * 1_000_000
            if least_qps == float('inf') or qps_pref >= least_qps / 100:
                ax.plot(
                    1.0, qps_pref,
                    marker=markers[-1],
                    color=colors[-1],
                    label="PreFiltering",
                    markersize=6
                )

        ax.set_title(f"Length of p = {p_len}", fontsize=26)
        ax.set_xlabel("Recall @ 10", fontsize=26)
        ax.set_yscale("log")
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.yaxis.get_offset_text().set_fontsize(18)

if __name__ == "__main__":
    methods = ["OptQuery", "PostFiltering", "pgvector", "VectorMaton"]
    datasets = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]

    n = len(datasets)
    fig, axes = plt.subplots(
        n, 3,
        figsize=(15, 5 * n),
        sharey=True
    )

    for row_idx, dataset in enumerate(datasets):
        plot_dataset_row(dataset, methods, axes[row_idx])

        # Dataset caption (row label)
        fig.text(
            0.02,                     # left margin
            1 - (row_idx + 0.5) / n,  # vertical center of the row
            dataset,
            va='center',
            ha='left',
            fontsize=30,
            rotation=90
        )

    # Global legend (only once)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=4,
        fontsize=28
    )

    plt.tight_layout(rect=[0.05, 0, 1, 0.93])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/recall_qps_all_datasets.pdf")
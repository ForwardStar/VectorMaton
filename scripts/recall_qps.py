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
    df = pd.read_csv(csv_path)

    if "time_us" not in df.columns or "recall" not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'time_us' and 'recall' columns.")

    return df["time_us"].values, df["recall"].values


def main(filepath1, filepath2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i in range(2, 5):
        ax = axes[i - 2]

        csv1 = os.path.join(filepath1, f"{i}.csv")
        csv2 = os.path.join(filepath2, f"{i}.csv")

        # Load curves
        time1, recall1 = load_curve(csv1)
        qps1 = 1000 / time1 * 1000000
        time2, recall2 = load_curve(csv2)
        qps2 = 1000 / time2 * 1000000

        # Plot curves
        ax.plot(recall1, qps1, marker="o", label="PostFiltering")
        ax.plot(recall2, qps2, marker="s", label="VectorMaton")

        ax.set_title(f"Length of p = {i}", fontsize=30)
        ax.set_xlabel("Recall @ 10", fontsize=30)
        if i == 2:
            ax.set_ylabel("QPS", fontsize=30)

        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.yaxis.get_offset_text().set_fontsize(20)  # <-- this is the 1e7
        ax.xaxis.get_offset_text().set_fontsize(20)  # for x-axis if needed

        # Only show legend on the first subplot to avoid clutter
        if i == 2:
            ax.legend(fontsize=20)

    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig("figures/recall_qps_curves.pdf")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python recall_qps.py <PostFiltering folder> <VectorMaton folder>")
        sys.exit(1)

    filepath1 = sys.argv[1]
    filepath2 = sys.argv[2]

    main(filepath1, filepath2)

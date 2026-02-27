import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def extract_avg_time_us_from_log(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"avg\s*\(us\)\s*:\s*([0-9.+\-eE]+)", content)
    if not match:
        return 1e10
    return float(match.group(1))


def load_curve(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if "time_us" not in df.columns or "recall" not in df.columns:
            raise ValueError(f"CSV file {csv_path} missing required columns")
        return df["time_us"].values, df["recall"].values
    except Exception:
        return None, None


def _clean_curve(qps, recall):
    if qps is None or recall is None:
        return None, None

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

    i = 0
    while i < len(recall):
        if recall[i] <= 0.1:
            recall = np.delete(recall, i)
            qps = np.delete(qps, i)
        else:
            i += 1

    if len(recall) == 0:
        return None, None

    order = np.argsort(recall)
    return qps[order], recall[order]


def plot_sift_panel_block(axes_block):
    dataset = "sift"
    methods = [
        ("PostFiltering", "PostFiltering"),
        ("VectorMaton-parallel", "VectorMaton"),
    ]
    markers = ["s", "d"]
    cs = plt.colormaps["tab10"]
    colors = [cs(i) for i in range(3)]
    p_lens = [2, 3]

    for idx, p_len in enumerate(p_lens):
        ax = axes_block[idx]

        qpss, recalls = [], []
        for method_path, _ in methods:
            csv_path = os.path.join("results", method_path, dataset, f"{p_len}.csv")
            time_us, recall = load_curve(csv_path)
            qps = 1_000_000 / time_us if time_us is not None else None
            qps, recall = _clean_curve(qps, recall)
            qpss.append(qps)
            recalls.append(recall)

        pre_log = os.path.join("results", "PreFiltering", dataset, f"{p_len}")
        time_prefiltering = extract_avg_time_us_from_log(pre_log) if os.path.exists(pre_log) else None

        qps_min = float("inf")
        qps_max = float("-inf")
        for i, (_, method_label) in enumerate(methods):
            if qpss[i] is None or recalls[i] is None:
                continue
            ax.plot(
                recalls[i],
                qpss[i],
                marker=markers[i],
                color=colors[i + 1],
                label=method_label,
                markersize=12,
                markerfacecolor="none",
            )
            qps_min = min(qps_min, np.min(qpss[i]))
            qps_max = max(qps_max, np.max(qpss[i]))

        if time_prefiltering is not None:
            qps_pref = 1_000_000 / time_prefiltering
            ax.plot(
                1.0,
                qps_pref,
                marker="*",
                color="black",
                label="PreFiltering",
                markersize=14,
                markerfacecolor="none",
            )
            qps_min = min(qps_min, qps_pref)
            qps_max = max(qps_max, qps_pref)

        ax.set_title(f"pattern length={p_len}", fontsize=30, fontweight="bold")
        ax.set_xlabel("Recall @ 10", fontsize=30)
        if p_len == 2:
            ax.set_ylabel("QPS", fontsize=30)
        ax.set_yscale("log")
        if qps_min < float("inf") and qps_max > float("-inf"):
            ax.set_ylim(bottom=qps_min * 0.5, top=qps_max * 2)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", labelsize=25)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    plot_sift_panel_block(axes)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=30,
        handlelength=1.2,
        markerscale=1.2,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.8])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/recall_qps_sift.pdf")

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


METHODS = ["OptQuery", "PostFiltering", "ACORN-1", "ACORN-gamma", "pgvector", "ElasticSearch", "VectorMaton"]
DATASETS = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]
DS_BRIEFS = ["spam", "words", "mtg", "arxiv", "prot", "code"]
P_LENGTHS = [5, 6, 7]
METHOD_LABELS = {"ACORN-gamma": "ACORN-γ"}


def method_label(method):
    return METHOD_LABELS.get(method, method)


def extract_avg_time_us_from_log(filepath):
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"avg\s*\(us\)\s*:\s*([0-9.+\-eE]+)", content)
    return float(match.group(1)) if match else None


def load_curve(csv_path):
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    if "time_us" not in df.columns or "recall" not in df.columns:
        return None, None

    df = df[["time_us", "recall"]].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["time_us"] > 0) & (df["recall"] >= 0)]
    if df.empty:
        return None, None

    qps = 1_000_000 / df["time_us"].to_numpy()
    recall = df["recall"].to_numpy()
    order = np.argsort(recall)
    return qps[order], recall[order]



def load_selectivity(dataset, p_len):
    for method in METHODS:
        csv_path = os.path.join("results", method, dataset, f"{p_len}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "average_selectivity" in df.columns and not df["average_selectivity"].dropna().empty:
            return float(df["average_selectivity"].dropna().iloc[0])
    return None

def format_selectivity(selectivity):
    return "N/A" if selectivity is None else f"{selectivity:.3g}"

def simplify_curve(qps, recall):
    if qps is None or recall is None or len(qps) == 0:
        return qps, recall

    keep = recall > 0.1
    qps = qps[keep]
    recall = recall[keep]
    if len(qps) == 0:
        return qps, recall

    filtered_qps = []
    filtered_recall = []
    best_qps = np.inf
    for r, q in zip(recall, qps):
        if q < best_qps:
            filtered_recall.append(r)
            filtered_qps.append(q)
            best_qps = q

    return np.array(filtered_qps), np.array(filtered_recall)


def plot_panel(ax, dataset, label, p_len, left_axis=False):
    markers = ["o", "s", "^", "d", "P", "X", "v"]
    colors = [plt.colormaps["tab10"](i) for i in range(len(METHODS))]
    plotted_any = False
    plotted_qps = []

    for i, method in enumerate(METHODS):
        csv_path = os.path.join("results", method, dataset, f"{p_len}.csv")
        qps, recall = load_curve(csv_path)
        qps, recall = simplify_curve(qps, recall)
        if qps is None or recall is None or len(qps) == 0:
            continue

        plotted_any = True
        plotted_qps.append(qps)
        ax.plot(
            recall,
            qps,
            marker=markers[i],
            color=colors[i],
            label=method_label(method),
            markersize=8,
            linewidth=1.8,
            markerfacecolor="none",
        )

    prefiltering_time = extract_avg_time_us_from_log(
        os.path.join("results", "PreFiltering", dataset, f"{p_len}")
    )
    if prefiltering_time and prefiltering_time > 0:
        qps_pref = 1_000_000 / prefiltering_time
        plotted_any = True
        plotted_qps.append(np.array([qps_pref]))
        ax.plot(
            1.0,
            qps_pref,
            marker="*",
            color="black",
            label="PreFiltering",
            markersize=11,
            markerfacecolor="none",
        )

    selectivity = load_selectivity(dataset, p_len)
    ax.set_title(f"{label}, |p| = {p_len} ({format_selectivity(selectivity)})", fontsize=18, fontweight="bold")
    ax.set_xlabel("Recall @ 10", fontsize=16)
    if left_axis:
        ax.set_ylabel("QPS", fontsize=16)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=13)

    if plotted_any:
        all_qps = np.concatenate(plotted_qps)
        ax.set_ylim(bottom=np.min(all_qps) * 0.5, top=np.max(all_qps) * 2)
    else:
        ax.text(
            0.5,
            0.5,
            "missing results",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
        )


def plot_3panel_block(dataset, label, axes_block, left_block=False):
    for i, p_len in enumerate(P_LENGTHS):
        plot_panel(
            axes_block[i],
            dataset,
            label,
            p_len,
            left_axis=(left_block and i == 0),
        )


def main():
    n = len(DATASETS)
    fig, axes = plt.subplots(
        n // 2,
        6,
        figsize=(28, 2.5 * n),
        sharey=False,
    )

    for i, dataset in enumerate(DATASETS):
        row = i // 2
        if i % 2 == 0:
            plot_3panel_block(dataset, DS_BRIEFS[i], axes[row, 0:3], left_block=True)
        else:
            plot_3panel_block(dataset, DS_BRIEFS[i], axes[row, 3:6])

    handles, labels = [], []
    for ax in axes.flatten():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=35,
        handlelength=1.2,
        markerscale=1.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/recall_qps_p5_p6_p7.pdf")


if __name__ == "__main__":
    main()

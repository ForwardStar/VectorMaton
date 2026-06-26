import os
import re

import matplotlib
matplotlib.use("Agg")
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


METHODS = ["OptQuery", "PostFiltering", "Hybrid", "ACORN-1", "ACORN-gamma", "pgvector", "ElasticSearch", "BM25Filtering", "VectorMaton"]
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

    keep = recall > 0.1
    return qps[keep], recall[keep]


def plot_panel(ax, dataset, label, p_len, left_axis=False):
    markers = ["o", "s", "^", "d", "P", "X", "v", "x", "*"]
    colors = [plt.colormaps["tab10"](i) for i in range(len(METHODS))]
    qpss = []
    recalls = []

    for method in METHODS:
        csv_path = os.path.join("results", method, dataset, f"{p_len}.csv")
        qps, recall = load_curve(csv_path)
        qps, recall = simplify_curve(qps, recall)
        qpss.append(qps)
        recalls.append(recall)

    min_recall_vectormaton = float("inf")
    vectormaton_index = METHODS.index("VectorMaton")
    if recalls[vectormaton_index] is not None and len(recalls[vectormaton_index]) > 0:
        min_recall_vectormaton = np.min(recalls[vectormaton_index])

    for i, method in enumerate(METHODS):
        if method == "VectorMaton" or recalls[i] is None or len(recalls[i]) == 0:
            continue
        mask = recalls[i] >= min_recall_vectormaton
        if np.any(mask):
            indices = np.where(mask)[0]
            if len(indices) < len(recalls[i]) // 2:
                indices = np.argsort(recalls[i])[-(len(recalls[i]) // 2):]
            recalls[i] = recalls[i][indices]
            qpss[i] = qpss[i][indices]

    plotted_qps = [qps for qps in qpss if qps is not None and len(qps) > 0]
    qps_min = min((np.min(qps) for qps in plotted_qps), default=float("inf"))

    prefiltering_time = extract_avg_time_us_from_log(
        os.path.join("results", "PreFiltering", dataset, f"{p_len}")
    )
    qps_pref = None
    if prefiltering_time and prefiltering_time > 0:
        candidate = 1_000_000 / prefiltering_time
        if candidate * 10 >= qps_min:
            qps_pref = candidate
            ax.plot(
                1.0,
                qps_pref,
                marker="*",
                color="black",
                label="PreFiltering",
                markersize=12,
                markerfacecolor="none",
            )

    for i, method in enumerate(METHODS):
        if qpss[i] is None or len(qpss[i]) == 0:
            continue
        ax.plot(
            recalls[i],
            qpss[i],
            marker=markers[i],
            color=colors[i],
            label=method_label(method),
            markersize=12,
            markerfacecolor="none",
        )

    selectivity = load_selectivity(dataset, p_len)
    ax.set_title(f"{label}, |p| = {p_len} ({format_selectivity(selectivity)})", fontsize=20, fontweight="bold")
    ax.set_xlabel("Recall @ 10", fontsize=20)
    if left_axis:
        ax.set_ylabel("QPS", fontsize=20)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=20)

    if plotted_qps:
        if qps_pref is not None:
            plotted_qps.append(np.array([qps_pref]))
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
            fontsize=20,
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
        figsize=(30, 1.6 * n),
        sharey=False,
    )

    for i, dataset in enumerate(DATASETS):
        row = i // 2
        if i % 2 == 0:
            plot_3panel_block(dataset, DS_BRIEFS[i], axes[row, 0:3], left_block=True)
        else:
            plot_3panel_block(dataset, DS_BRIEFS[i], axes[row, 3:6])

    handle_by_label = {}
    for ax in axes.flatten():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            handle_by_label.setdefault(label, handle)

    legend_labels = ["OptQuery", "PreFiltering"] + [
        method_label(method) for method in METHODS if method != "OptQuery"
    ]
    labels = [label for label in legend_labels if label in handle_by_label]
    handles = [handle_by_label[label] for label in labels]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=10,
        fontsize=25,
        handlelength=1.2,
        markerscale=1.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/recall_qps_p5_p6_p7.pdf")

    arxiv_fig, arxiv_axes = plt.subplots(
        1,
        3,
        figsize=(15, 3.2),
        sharey=False,
    )
    plot_3panel_block(
        "arxiv-small",
        "arxiv",
        arxiv_axes,
        left_block=True,
    )

    arxiv_fig.tight_layout()
    arxiv_fig.savefig("figures/recall_qps_arxiv_p5_p6_p7.pdf")



if __name__ == "__main__":
    main()

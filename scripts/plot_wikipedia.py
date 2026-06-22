import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker


METHOD_LABELS = {"ACORN-gamma": "ACORN-γ"}
ALL_METHODS = [
    "OptQuery",
    "PostFiltering",
    "Hybrid",
    "ACORN-1",
    "ACORN-gamma",
    "pgvector",
    "ElasticSearch",
    "BM25Filtering",
    "VectorMaton",
]
DATASET = "wikipedia"
PATTERN_LENGTH = 12
QUERY_OUTPUT = os.path.join("results", "VectorMaton", DATASET, "12.queries")
STRINGS_FILE = os.path.join("datasets", DATASET, "strings.txt")
QUERY_STRINGS_FILE = "string_prompt.txt"


def method_label(method):
    return METHOD_LABELS.get(method, method)


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
        return None
    return float(match.group(1))


def load_curve(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if "time_us" not in df.columns or "recall" not in df.columns:
            raise ValueError(
                f"CSV file {csv_path} must contain 'time_us' and 'recall' columns."
            )
        return df["time_us"].to_numpy(), df["recall"].to_numpy()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Skipping {csv_path}: {error}")
        return None, None


def clean_curve(time_us, recall):
    if time_us is None or recall is None:
        return None, None

    time_us = np.asarray(time_us, dtype=float)
    recall = np.asarray(recall, dtype=float)
    valid = np.isfinite(time_us) & np.isfinite(recall) & (time_us > 0)
    qps = 1_000_000 / time_us[valid]
    recall = recall[valid]

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
    recall = recall[keep]
    qps = qps[keep]
    if len(recall) == 0:
        return None, None

    order = np.argsort(recall)
    return qps[order], recall[order]


def load_selectivity():
    for method in ALL_METHODS:
        csv_path = os.path.join(
            "results", method, DATASET, f"{PATTERN_LENGTH}.csv"
        )
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            continue
        if "average_selectivity" in df.columns:
            values = df["average_selectivity"].dropna()
            if not values.empty:
                return float(values.iloc[0])
    return None




def load_top_chunk_ids(query_number=1, top_k=3):
    """Return top chunk IDs from the last result row for one query."""
    with open(QUERY_OUTPUT, "r", encoding="utf-8") as query_file:
        content = query_file.read()

    query_match = re.search(
        rf"^Query {query_number}:\s*$\n(.*?)(?=^Query \d+:\s*$|\Z)",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not query_match:
        raise ValueError(f"Query {query_number} not found in {QUERY_OUTPUT}")

    result_lines = re.findall(
        r"^ef_search\s*=\s*[^,]+,\s*k\s*=\s*\d+\s*:\s*(.*)$",
        query_match.group(1),
        flags=re.MULTILINE,
    )
    if not result_lines:
        raise ValueError(f"No result rows found for Query {query_number}")
    return [int(chunk_id) for chunk_id in result_lines[-1].split()[:top_k]]


def load_chunks(chunk_ids):
    wanted = set(chunk_ids)
    chunks = {}
    with open(STRINGS_FILE, "r", encoding="utf-8") as strings_file:
        for chunk_id, text in enumerate(strings_file):
            if chunk_id in wanted:
                chunks[chunk_id] = text.strip()
                if len(chunks) == len(wanted):
                    break
    missing = wanted - chunks.keys()
    if missing:
        raise ValueError(f"Chunk IDs not found in {STRINGS_FILE}: {sorted(missing)}")
    return chunks


def context_excerpt(text, pattern, context_chars=110):
    match_start = text.find(pattern)
    if match_start == -1:
        return text[: 2 * context_chars], -1, -1

    start = max(0, match_start - context_chars)
    end = min(len(text), match_start + len(pattern) + context_chars)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "..." + excerpt
        match_start = match_start - start + 3
    else:
        match_start -= start
    if end < len(text):
        excerpt += "..."
    return excerpt, match_start, match_start + len(pattern)


def highlighted_lines(text, highlight_start, highlight_end, width=64):
    lines = []
    for line_start in range(0, len(text), width):
        line = text[line_start : line_start + width]
        line_end = line_start + len(line)
        overlap_start = max(line_start, highlight_start)
        overlap_end = min(line_end, highlight_end)

        segments = []
        if overlap_start >= overlap_end:
            segments.append(
                TextArea(line, textprops={"family": "monospace", "size": 9})
            )
        else:
            before = line[: overlap_start - line_start]
            highlighted = line[
                overlap_start - line_start : overlap_end - line_start
            ]
            after = line[overlap_end - line_start :]
            if before:
                segments.append(
                    TextArea(before, textprops={"family": "monospace", "size": 9})
                )
            segments.append(
                TextArea(
                    highlighted,
                    textprops={
                        "family": "monospace",
                        "size": 9,
                        "weight": "bold",
                        "color": "darkred",
                        "backgroundcolor": "#ffe66d",
                    },
                )
            )
            if after:
                segments.append(
                    TextArea(after, textprops={"family": "monospace", "size": 9})
                )
        lines.append(HPacker(children=segments, align="baseline", pad=0, sep=0))
    return lines


def plot_top_chunks(ax, query_number=1, query_text=None):
    with open(QUERY_STRINGS_FILE, "r", encoding="utf-8") as query_file:
        patterns = [line.strip() for line in query_file if line.strip()]
    if query_number < 1 or query_number > len(patterns):
        raise ValueError(f"Query {query_number} not found in {QUERY_STRINGS_FILE}")

    pattern = patterns[query_number - 1]
    chunk_ids = load_top_chunk_ids(query_number=query_number)
    chunks = load_chunks(chunk_ids)

    chunk_boxes = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        excerpt, highlight_start, highlight_end = context_excerpt(
            chunks[chunk_id], pattern
        )
        title = TextArea(
            f"{rank}. Chunk {chunk_id}",
            textprops={"size": 12, "weight": "bold"},
        )
        body = VPacker(
            children=highlighted_lines(excerpt, highlight_start, highlight_end),
            align="left",
            pad=0,
            sep=1,
        )
        chunk_boxes.append(
            VPacker(children=[title, body], align="left", pad=0, sep=4)
        )

    panel = VPacker(children=chunk_boxes, align="left", pad=0, sep=14)
    anchored_panel = AnchoredOffsetbox(
        loc="upper left",
        child=panel,
        frameon=False,
        bbox_to_anchor=(0, 0.84),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax.add_artist(anchored_panel)
    title = (
        f"Top-3 chunks for \"{query_text}\""
        if query_text
        else f"Top-3 chunks for {pattern}"
    )
    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=0,
        loc="left",
        y=0.93,
    )
    ax.set_axis_off()


def plot_wikipedia(ax):
    markers = ["o", "s", "^", "d", "P", "X", "v", "x", "*"]
    colors = plt.colormaps["tab10"]
    curves = []

    for method in ALL_METHODS:
        csv_path = os.path.join(
            "results", method, DATASET, f"{PATTERN_LENGTH}.csv"
        )
        time_us, recall = load_curve(csv_path)
        qps, recall = clean_curve(time_us, recall)
        curves.append((method, qps, recall))

    vector_recall = next(
        (recall for method, _, recall in curves if method == "VectorMaton"), None
    )
    if vector_recall is not None:
        min_vector_recall = np.min(vector_recall)
        filtered_curves = []
        for method, qps, recall in curves:
            if method != "VectorMaton" and recall is not None:
                indices = np.flatnonzero(recall >= min_vector_recall)
                minimum_count = len(recall) // 2
                if len(indices) < minimum_count:
                    indices = np.argsort(recall)[-minimum_count:]
                if len(indices) > 0:
                    qps, recall = qps[indices], recall[indices]
            filtered_curves.append((method, qps, recall))
        curves = filtered_curves

    qps_values = []
    for i, (method, qps, recall) in enumerate(curves):
        if qps is None:
            continue
        ax.plot(
            recall,
            qps,
            marker=markers[i],
            color=colors(i),
            label=method_label(method),
            markersize=12,
            markerfacecolor="none",
        )
        qps_values.extend(qps)

    prefiltering_log = os.path.join(
        "results", "PreFiltering", DATASET, str(PATTERN_LENGTH)
    )
    if os.path.exists(prefiltering_log):
        time_us = extract_avg_time_us_from_log(prefiltering_log)
        if time_us is not None and time_us > 0:
            qps_prefiltering = 1_000_000 / time_us
            if not qps_values or qps_prefiltering * 10 >= min(qps_values):
                ax.plot(
                    1.0,
                    qps_prefiltering,
                    marker="*",
                    color="black",
                    label="PreFiltering",
                    markersize=14,
                    markerfacecolor="none",
                )
                qps_values.append(qps_prefiltering)

    selectivity = load_selectivity()
    selectivity_text = "N/A" if selectivity is None else f"{selectivity:.3g}"
    ax.set_title(
        f"wikipedia",
        fontsize=25,
        fontweight="bold",
    )
    ax.set_xlabel("Recall @ 10", fontsize=25)
    ax.set_ylabel("QPS", fontsize=25)
    ax.set_yscale("log")
    if qps_values:
        ax.set_ylim(bottom=min(qps_values) * 0.5, top=max(qps_values) * 2)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=20)


if __name__ == "__main__":
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    plot_wikipedia(axes[0])
    plot_top_chunks(
        axes[1],
        query_number=3,
        query_text="What careers and institutions are part of academia?",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_entries = dict(zip(labels, handles))
    labels = ["PreFiltering"] + [
        method_label(method) for method in ALL_METHODS
    ]
    labels = [label for label in labels if label in legend_entries]
    handles = [legend_entries[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=25,
        handlelength=1.2,
        markerscale=1.2,
        bbox_to_anchor=(0.5, 1.0),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.78])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/recall_qps_wikipedia.pdf", bbox_inches="tight")

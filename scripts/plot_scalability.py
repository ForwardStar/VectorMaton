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


PCTS = list(range(10, 101, 10))
THREADS = [1, 2, 4, 8, 16, 32]
SCALABILITY_METHODS = ["OptQuery", "VectorMaton"]
DATASETS = [
    ("spam", "spam"),
    ("words", "words"),
]
PARALLEL_DATASETS = [
    ("swissprot", "prot"),
    ("code_search_net", "code"),
]
PARALLEL_RESULT_ROOT = "results/VectorMaton-parallel"
METHOD_STYLES = {
    "OptQuery": {"marker": "o", "color_idx": 0},
    "VectorMaton": {"marker": "v", "color_idx": 4},
}


def parse_metrics(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    m_match = re.search(r"Total length of strings:\s*([0-9]+)", content)
    index_match = re.search(r"Total index size:\s*([0-9]+)\s*bytes", content)
    time_match = re.search(r"index built took\s*([0-9]+)\s*(?:μs|us)", content)
    if m_match is None or index_match is None:
        return None

    total_length = int(m_match.group(1))
    index_size_mb = int(index_match.group(1)) / 1_000_000.0
    index_time_s = int(time_match.group(1)) / 1_000_000.0 if time_match else None
    return total_length, index_size_mb, index_time_s


def collect_points(method, dataset_dir):
    xs = []
    ys_size = []
    ys_time = []
    base_dir = os.path.join("results", method, "scalability", dataset_dir)
    for pct in PCTS:
        path = os.path.join(base_dir, f"{pct}%")
        if not os.path.exists(path):
            continue
        point = parse_metrics(path)
        if point is None:
            continue
        x, y_size, y_time = point
        xs.append(x)
        ys_size.append(y_size)
        ys_time.append(y_time)
    return xs, ys_size, ys_time


def parse_parallel_build_time_seconds(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"VectorMaton-parallel index built took\s*([0-9]+)\S*", content)
    if match is None:
        return None
    return int(match.group(1)) / 1_000_000.0


def collect_parallel_points(dataset):
    xs = []
    ys = []
    for num_threads in THREADS:
        path = os.path.join(PARALLEL_RESULT_ROOT, dataset, f"{num_threads}.log")
        if not os.path.exists(path):
            continue
        build_time_s = parse_parallel_build_time_seconds(path)
        if build_time_s is None:
            continue
        xs.append(num_threads)
        ys.append(build_time_s)
    return xs, ys


def main():
    fig, axes = plt.subplots(1, 6, figsize=(30, 6), sharey=False)
    cs = plt.colormaps["tab10"]
    legend_handles = None
    legend_labels = None

    for col, (dataset_dir, dataset_title) in enumerate(DATASETS):
        ax_size = axes[col]
        ax_time = axes[col + 2]
        plotted_size = False
        plotted_time = False

        for method in SCALABILITY_METHODS:
            x, y_size, y_time = collect_points(method, dataset_dir)
            if not x:
                continue

            style = METHOD_STYLES[method]
            ax_size.plot(
                x,
                y_size,
                marker=style["marker"],
                linestyle="-",
                markersize=17,
                markerfacecolor="none",
                color=cs(style["color_idx"]),
                label=method,
            )
            plotted_size = True

            x_time = [xv for xv, yv in zip(x, y_time) if yv is not None]
            y_time_valid = [yv for yv in y_time if yv is not None]
            if x_time:
                ax_time.plot(
                    x_time,
                    y_time_valid,
                    marker=style["marker"],
                    linestyle="-",
                    markersize=17,
                    markerfacecolor="none",
                    color=cs(style["color_idx"]),
                    label=method,
                )
                plotted_time = True

        if not plotted_size:
            ax_size.text(
                0.5,
                0.5,
                "No valid points",
                ha="center",
                va="center",
                transform=ax_size.transAxes,
                fontsize=14,
            )
        if not plotted_time:
            ax_time.text(
                0.5,
                0.5,
                "No valid points",
                ha="center",
                va="center",
                transform=ax_time.transAxes,
                fontsize=14,
            )

        ax_size.set_title(f"{dataset_title} (Size)", fontsize=30, fontweight="bold")
        ax_time.set_title(f"{dataset_title} (Time)", fontsize=30, fontweight="bold")
        ax_size.set_xlabel("Total sequence length", fontsize=30)
        ax_time.set_xlabel("Total sequence length", fontsize=30)

        for ax in (ax_size, ax_time):
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", labelsize=25)
            ax.xaxis.get_offset_text().set_size(25)
            ax.yaxis.get_offset_text().set_size(25)
            # ax.set_yscale("log")

        handles, labels = ax_size.get_legend_handles_labels()
        if handles:
            legend_handles, legend_labels = handles, labels

    axes[0].set_ylabel("Size (MB)", fontsize=30)
    axes[len(DATASETS)].set_ylabel("Time (s)", fontsize=30)

    parallel_color = cs(4)
    parallel_start = 2 * len(DATASETS)
    for i, (dataset, dataset_brief) in enumerate(PARALLEL_DATASETS):
        ax = axes[parallel_start + i]
        x, y = collect_parallel_points(dataset)

        if not x:
            ax.text(
                0.5,
                0.5,
                "No valid points",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
        else:
            ax.plot(
                x,
                y,
                marker="v",
                linestyle="-",
                markersize=20,
                markerfacecolor="none",
                color=parallel_color,
            )

        ax.set_title(f"{dataset_brief} (Parallel)", fontsize=30, fontweight="bold")
        ax.set_xlabel("# Threads", fontsize=30)
        if i == 0:
            ax.set_ylabel("Time (s)", fontsize=30)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", labelsize=25)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.set_xticklabels([str(t) for t in THREADS], fontsize=25)

    if legend_handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=6,
            fontsize=35,
            handlelength=1.2,
            markerscale=1.5
        )
        plt.tight_layout(rect=[0, 0, 1, 0.8])
    else:
        plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/scalability_index_size_vs_m.pdf")


if __name__ == "__main__":
    main()

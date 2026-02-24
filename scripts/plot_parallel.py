import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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


THREADS = [1, 2, 4, 8, 16, 32]
DATASETS = ["swissprot", "code_search_net"]
DATASETS_BRIEF = ["prot", "code"]
RESULT_ROOT = "results/VectorMaton-parallel"


def parse_build_time_seconds(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"VectorMaton-parallel index built took\s*([0-9]+)\S*", content)
    if match is None:
        return None

    return int(match.group(1)) / 1_000_000.0


def collect_points(dataset):
    xs = []
    ys = []

    for num_threads in THREADS:
        path = os.path.join(RESULT_ROOT, dataset, f"{num_threads}.log")
        if not os.path.exists(path):
            continue

        build_time_s = parse_build_time_seconds(path)
        if build_time_s is None:
            continue

        xs.append(num_threads)
        ys.append(build_time_s)

    return xs, ys


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    axes_flat = axes.flatten()

    cs = plt.colormaps["tab10"]
    color = cs(3)

    for i, dataset in enumerate(DATASETS):
        x, y = collect_points(dataset)
        ax = axes_flat[i]

        if len(x) == 0:
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
                marker="d",
                linestyle="-",
                markersize=10,
                markerfacecolor="none",
                color=color,
            )

        ax.set_title(DATASETS_BRIEF[i], fontsize=25, fontweight="bold")
        ax.set_xlabel("# Threads", fontsize=25)
        ax.set_ylabel("Index construction time (s)", fontsize=25)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/parallel_index_construction_time.pdf")


if __name__ == "__main__":
    main()

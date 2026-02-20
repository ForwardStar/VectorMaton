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


PCTS = list(range(10, 101, 10))
DATASETS = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]
DATASETS_BRIEF = ["spam", "words", "mtg", "arxiv", "prot", "code"]


def parse_metrics(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    m_match = re.search(r"Total length of strings:\s*([0-9]+)", content)
    index_match = re.search(r"Total index size:\s*([0-9]+)\s*bytes", content)
    if m_match is None or index_match is None:
        return None

    total_length_m = int(m_match.group(1))
    index_size_mb = int(index_match.group(1)) / 1_000_000.0
    return total_length_m, index_size_mb


def collect_points(dataset):
    xs = []
    ys = []
    for pct in PCTS:
        path = os.path.join("results", "VectorMaton", dataset, f"{pct}%")
        if not os.path.exists(path):
            continue

        point = parse_metrics(path)
        if point is None:
            continue

        x, y = point
        xs.append(x)
        ys.append(y)

    return xs, ys


def main():
    fig, axes = plt.subplots(1, 6, figsize=(28, 4), sharey=False)
    axes_flat = axes.flatten()

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
            cs = plt.colormaps['tab10']
            ax.plot(
                x,
                y,
                marker="d",
                linestyle="-",
                markersize=10,
                markerfacecolor="none",
                color=cs(3),
            )

        ax.set_title(DATASETS_BRIEF[i], fontsize=25, fontweight="bold")
        ax.set_xlabel("Total string length", fontsize=25)
        ax.set_ylabel("Index size (MB)", fontsize=25)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", labelsize=20)
        ax.xaxis.get_offset_text().set_x(1.05)
        ax.xaxis.get_offset_text().set_size(20)
        ax.yaxis.get_offset_text().set_size(20)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/scalability_index_size_vs_m.pdf")


if __name__ == "__main__":
    main()

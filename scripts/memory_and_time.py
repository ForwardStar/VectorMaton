import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase

plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams['hatch.linewidth'] = 2.8
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if "Libertine_R" in font:
        font_prop = fm.FontProperties(fname=font)
        font_name = font_prop.get_name()
        plt.rcParams.update({'font.family': font_name})
        rcParams['font.family'] = font_name
        rcParams['mathtext.fontset'] = 'custom'
        rcParams['mathtext.rm'] = font_name

# Configuration
datasets = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]
ds_brief = ["spam", "words", "mtg", "arxiv", "prot", "code"]
methods = ["OptQuery", "VectorMaton"]
cs = plt.colormaps['tab10']
colors = [cs(0), cs(4)]
hatches = ['\\', '/']
hatch_colors = [colors[0], colors[1]]

# Regex to extract index size
size_pattern = re.compile(r"Total index size:\s*(\d+)\s*bytes")
vector_pattern = re.compile(r"Vector size:\s*(\d+)\s*bytes")
string_pattern = re.compile(r"String size:\s*(\d+)\s*bytes")
time_pattern = re.compile(r"index built took\s*(\d+)μs")

# Store results: {method: [sizes aligned with datasets or None]}
results = {method: [] for method in methods}
time_results = {method: [] for method in methods}

for method in methods:
    for dataset in datasets:
        filepath = os.path.join("results", method, dataset, "2")
        size = None
        vector_size = None
        string_size = None

        index_time = None

        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    if method == "VectorMaton":
                        if vector_size is None:
                            m = vector_pattern.search(line)
                            if m:
                                vector_size = int(m.group(1))
                        if string_size is None:
                            m = string_pattern.search(line)
                            if m:
                                string_size = int(m.group(1))

                    if size is None:
                        m = size_pattern.search(line)
                        if m:
                            size = int(m.group(1))

                    if index_time is None:
                        m = time_pattern.search(line)
                        if m:
                            index_time = int(m.group(1))  # microseconds

        # if method == "VectorMaton" and vector_size is not None and string_size is not None:
        #     results["Data size"].append(vector_size + string_size)

        results[method].append(size)
        time_results[method].append(index_time)

# Plot
x = list(range(len(datasets)))
bar_width = 0.4

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ax_mem, ax_time = axes

class HandlerOverlayPatch(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        base_proto, hatch_proto = orig_handle
        x0, y0 = xdescent, ydescent
        artists = []
        for proto in (base_proto, hatch_proto):
            p = Rectangle(
                (x0, y0), width, height,
                facecolor=proto.get_facecolor(),
                edgecolor=proto.get_edgecolor(),
                hatch=proto.get_hatch(),
                linewidth=proto.get_linewidth(),
            )
            p.set_transform(trans)
            artists.append(p)
        return artists

def draw_bars(ax, xs, ys, i, label):
    # Layer 1: hollow bar with black border for shape readability
    ax.bar(
        xs,
        ys,
        width=bar_width,
        label=label,
        facecolor='none',
        edgecolor='black',
        linewidth=2.4,
    )

    # Layer 2: hatch-only overlay in the designated method color
    ax.bar(
        xs,
        ys,
        width=bar_width,
        facecolor='none',
        edgecolor=hatch_colors[i],
        hatch=hatches[i],
        linewidth=0.1,
    )

for i, method in enumerate(methods):
    xs, ys = [], []

    for j, size in enumerate(results[method]):
        if size is not None:
            xs.append(j + (i - 0.5) * bar_width)
            ys.append(size / (1024 * 1024))  # MB

    draw_bars(ax_mem, xs, ys, i, method)

ax_mem.set_xticks(x)
ax_mem.set_xticklabels(ds_brief, fontsize=25)
ax_mem.tick_params(axis='y', labelsize=25)
ax_mem.set_ylabel("Size (MB)", fontsize=30)
ax_mem.set_yscale("log")
ax_mem.set_xlabel("(a) Index size", fontsize=30, fontweight='bold')
ax_mem.grid(True, axis="y", linestyle="--", alpha=0.7)

for i, method in enumerate(methods):
    xs, ys = [], []

    for j, t in enumerate(time_results[method]):
        if t is not None:
            xs.append(j + (i - 0.5) * bar_width)
            ys.append(t / 1e6)  # seconds

    draw_bars(ax_time, xs, ys, i, method)

ax_time.set_xticks(x)
ax_time.set_xticklabels(ds_brief, fontsize=25)
ax_time.tick_params(axis='y', labelsize=25)
ax_time.set_ylabel("Time (s)", fontsize=30)
ax_time.set_yscale("log")
ax_time.set_xlabel("(b) Index time", fontsize=30, fontweight='bold')
ax_time.grid(True, axis="y", linestyle="--", alpha=0.7)

fig.legend(
    handles=[
        (
            Patch(facecolor='none', edgecolor='black', linewidth=2.4),
            Patch(facecolor='none', edgecolor=hatch_colors[i], linewidth=0.1, hatch=hatches[i]),
        )
        for i in range(len(methods))
    ],
    labels=methods,
    loc="upper center",
    ncol=len(methods),
    fontsize=45,
    handler_map={tuple: HandlerOverlayPatch()},
)

plt.tight_layout(rect=[0, 0, 1, 0.8])
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/memory_and_time.pdf")

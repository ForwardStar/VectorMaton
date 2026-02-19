import os
import re
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

# Configuration
datasets = ["spam", "words", "mtg", "arxiv-small", "swissprot", "code_search_net"]
ds_brief = ["spam", "words", "mtg", "arxiv", "prot", "code"]
methods = ["OptQuery", "VectorMaton"]
cs = plt.colormaps['tab10']
colors = [cs(0), cs(3), cs(4)]
hatches = ['/', '\\', 'x']

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
x = range(len(datasets))
bar_width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
ax_mem, ax_time = axes

for i, method in enumerate(methods):
    xs, ys = [], []

    for j, size in enumerate(results[method]):
        if size is not None:
            xs.append(j + (i - 0.5) * bar_width)
            ys.append(size / (1024 * 1024))  # MB

    ax_mem.bar(xs, ys, width=bar_width,
               label=method, color=colors[i], hatch=hatches[i], edgecolor=colors[i], facecolor='none')

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

    ax_time.bar(xs, ys, width=bar_width,
                label=method, color=colors[i], hatch=hatches[i], edgecolor=colors[i], facecolor='none')

ax_time.set_xticks(x)
ax_time.set_xticklabels(ds_brief, fontsize=25)
ax_time.tick_params(axis='y', labelsize=25)
ax_time.set_ylabel("Time (s)", fontsize=30)
ax_time.set_yscale("log")
ax_time.set_xlabel("(b) Index time", fontsize=30, fontweight='bold')
ax_time.grid(True, axis="y", linestyle="--", alpha=0.7)

handles, labels = ax_mem.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(methods),
    fontsize=35
)

plt.tight_layout(rect=[0, 0, 1, 0.8])
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/memory_and_time.pdf")

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

# Store results: {method: [sizes aligned with datasets or None]}
results = {method: [] for method in methods}

for method in methods:
    for dataset in datasets:
        filepath = os.path.join("results", method, dataset, "2")
        size = None
        vector_size = None
        string_size = None

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
                    match = size_pattern.search(line)
                    if match:
                        size = int(match.group(1))
                        break

        # if method == "VectorMaton" and vector_size is not None and string_size is not None:
        #     results["Data size"].append(vector_size + string_size)
        
        results[method].append(size)

# Plot
x = range(len(datasets))
bar_width = 0.3

plt.figure(figsize=(12, 5))

for i, method in enumerate(methods):
    xs = []
    ys = []

    for j, size in enumerate(results[method]):
        if size is not None:
            xs.append(j + (i - 0.5) * bar_width)
            ys.append(size / (1024 * 1024))  # convert to MB

    plt.bar(xs, ys, width=bar_width, label=method, color=colors[i], hatch=hatches[i])

plt.xticks(x, ds_brief, fontsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.ylabel("Index Size (MB)", fontsize=30)
plt.yscale("log")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(methods),
    fontsize=30
)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/memory.pdf")
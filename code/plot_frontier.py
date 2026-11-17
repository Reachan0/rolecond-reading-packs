"""Quantity-quality curve of fixed-k construction order, with RoleCond and matched top-m.

Usage: python plot_frontier.py <fixed_quantity_results.json> <output stem (no extension)>
Style follows original/plot_evidence.py.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, BLUE0 = "#1F5F80", "#B95739", "#8FAFC2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "grid.color": "#e4e7e9", "grid.linewidth": .5,
    "pdf.fonttype": 42, "mathtext.fontset": "stix", "savefig.facecolor": "white",
})

res = json.load(open(sys.argv[1]))["summary"]
stem = sys.argv[2]
ks = range(1, 9)
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), layout="constrained")
for ax, metric, title in [(axs[0], "grade", "(a) Mean grade"), (axs[1], "recall", "(b) Positive recall")]:
    x = [res[f"top_{k}"]["returned"] for k in ks]
    y = [res[f"top_{k}"][metric] for k in ks]
    ax.plot(x, y, color=BLUE, lw=1.1, marker="o", ms=3, mec="white", mew=.4, zorder=2,
            label="Construction order, fixed $k$")
    for k, xi, yi in zip(ks, x, y):
        if k >= 4:
            dy = 5 if metric == "grade" else -9
            ax.annotate(f"{k}", (xi, yi), textcoords="offset points", xytext=(3 if metric == "grade" else 0, dy),
                        ha="left" if metric == "grade" else "center", fontsize=6, color=BLUE)
    tm = res["top_m"]
    ax.scatter([tm["returned"]], [tm[metric]], marker="s", s=14, color=BLUE0, edgecolor="#333",
               lw=.4, zorder=3, label="Matched top-$m$")
    rc = res["rolecond"]
    ax.scatter([rc["returned"]], [rc[metric]], marker="D", s=18, color=ORANGE, edgecolor="#333",
               lw=.4, zorder=4, label="RoleCond")
    ax.set_xlim(3.6, 7.6)
    if metric == "grade":
        ax.set_ylim(1.5, 2.55)
    else:
        ax.set_ylim(.8, 1.02)
    ax.set_xlabel("Mean returned papers")
    ax.set_title(title, fontsize=8)
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6)
axs[1].legend(loc="lower right", frameon=False, handlelength=1.2, handletextpad=.3,
              borderaxespad=.1)
for ext in ("pdf", "png"):
    fig.savefig(f"{stem}.{ext}", dpi=400)

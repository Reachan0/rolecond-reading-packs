"""Presentation figure for the ISPA camera-ready: validity under strict and fence-stripped parsing,
and input citation coverage for both models (GPT-5.5 strict = fence-stripped; Claude fence-stripped).

Usage: python plot_presentation_both.py <relaxed_metrics.json> <output stem>
Style follows original/plot_evidence.py.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, PEACH = "#1F5F80", "#B95739", "#E8B0A0"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "grid.color": "#e4e7e9", "grid.linewidth": .5,
    "pdf.fonttype": 42, "mathtext.fontset": "stix", "savefig.facecolor": "white",
})
d = json.load(open(sys.argv[1]))
stem = sys.argv[2]
conds = ["F", "G", "R", "W"]
x = np.arange(4)
w = .36
strict, relaxed = d["strict"]["summary"], d["relaxed"]["summary"]

fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), layout="constrained")
ax = axs[0]
cs = [strict[f"claude-sonnet-4-6:{c}"]["valid"] for c in conds]
cr = [relaxed[f"claude-sonnet-4-6:{c}"]["valid"] for c in conds]
ax.bar(x - w / 2, cs, w, color=PEACH, edgecolor="white", lw=.4, zorder=2, label="Claude, strict")
ax.bar(x + w / 2, cr, w, color=ORANGE, edgecolor="white", lw=.4, zorder=2, label="Claude, fence-stripped")
ax.axhline(36, color=BLUE, lw=1.1, ls="--", zorder=3, label="GPT-5.5, both")
for i, (a, b) in enumerate(zip(cs, cr)):
    ax.text(i - w / 2, a + .8, str(a), ha="center", fontsize=6)
    ax.text(i + w / 2, b + .8, str(b), ha="center", fontsize=6)
ax.set_ylim(0, 50)
ax.set_yticks([0, 12, 24, 36])
ax.set_xticks(x, conds)
ax.set_ylabel("Valid responses (of 36)")
ax.set_title("(a) Validity", fontsize=8)
ax.legend(loc="upper center", ncol=1, frameon=False, fontsize=5.8, handlelength=1.1,
          handletextpad=.3, borderaxespad=0, labelspacing=.2)
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=.6)

ax = axs[1]
for off, model, color, label in [(-w / 2, "gpt-5.5", BLUE, "GPT-5.5"),
                                 (w / 2, "claude-sonnet-4-6", ORANGE, "Claude")]:
    m = np.array([relaxed[f"{model}:{c}"]["input_coverage"] for c in conds])
    lo, hi = np.array([relaxed[f"{model}:{c}"]["input_coverage_ci95"] for c in conds]).T
    ax.bar(x + off, m, w, color=color, edgecolor="white", lw=.4, zorder=2, label=label)
    ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="none", color="#30363a", lw=.6, capsize=1.5,
                zorder=3)
ax.set_ylim(.5, 1.12)
ax.set_xticks(x, conds)
ax.set_ylabel("Input citation coverage")
ax.set_title("(b) Coverage", fontsize=8)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=6, handlelength=1, handletextpad=.3,
          columnspacing=.8, borderaxespad=0)
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=.6)
for ext in ("pdf", "png"):
    fig.savefig(f"{stem}.{ext}", dpi=400)

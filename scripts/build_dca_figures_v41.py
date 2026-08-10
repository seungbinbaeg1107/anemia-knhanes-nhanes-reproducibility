"""Pooled decision curves. The main panel compares the random forest with the
flexible conventional comparator and the two default strategies over the
1%-10% region; the appendix panel adds the linear and age-only arms over the
full evaluated grid. Each figure is sized to its placed width.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

OUT = Path(ANEMIA_ROOT) / "outputs"
d = pd.read_csv(OUT / "pooled_comparative_dca_v41.csv")

INK = "#12263A"
BODY = "#22344A"
GRID = "#DDE3EA"

RF = ("random_forest", "Pooled random forest", "#1F3B5B", "-", 1.9)
FLEX = ("flexible_logistic", "Flexible penalized logistic regression", "#B3541E", "-", 1.7)
LIN = ("penalized_logistic_regression", "Penalized logistic regression", "#7D3C98", "-", 1.5)
AGE = ("age_only", "Age-only model", "#2A9D8F", "-", 1.5)
ALL = ("cbc_test_all", "CBC test all", "#E8912A", "--", 1.5)
NONE = ("cbc_test_none", "CBC test none", "#6B7280", ":", 1.4)

MAIN = [RF, FLEX, ALL, NONE]
APPX = [RF, FLEX, LIN, AGE, ALL, NONE]


def draw(series, hi, path, fig_w, note=""):
    sub = d[d.threshold <= hi + 1e-9]
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.52))
    for col, label, colour, ls, lw in series:
        ax.plot(sub.threshold, sub[col], label=label, color=colour,
                linestyle=ls, linewidth=lw, solid_capstyle="round")
    ax.set_xlabel("Threshold probability", fontsize=8.6, color=INK)
    ax.set_ylabel("Survey-weighted net benefit", fontsize=8.6, color=INK)
    ax.set_xlim(0, hi)
    cols = [c for c, *_ in series]
    lo = min(0.0, sub[cols].min().min())
    top = sub[cols].max().max()
    ax.set_ylim(lo - 0.004, top + 0.010)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.0, colors=BODY, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, fontsize=7.9, loc="upper right", labelcolor=BODY)
    if note:
        ax.text(0.985, 0.06, note, transform=ax.transAxes, ha="right",
                fontsize=7.6, color="#B4551F")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# main text: clinically credible region, fairest comparator only
draw(MAIN, 0.10, OUT / "figure_main_dca_pooled_1_10_v35.png", 5.85)
# appendix: full evaluated grid, every arm
draw(APPX, 0.30, OUT / "figure_appendix_dca_pooled_1_30_v35.png", 6.30,
     "CBC test-all continues below the display range")

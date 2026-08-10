"""Figure 3: internal and external discrimination, overall and by sex, as a paired
forest plot with a numeric column. Values are taken from the manuscript so the
figure and the text cannot diverge. Internal intervals are conditional
fixed-prediction cluster-bootstrap intervals; external intervals are
design-based delete-one-PSU jackknife intervals.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

OUT = Path(ANEMIA_ROOT) / "outputs"

INK = "#1A2E45"
BODY = "#3A4754"
BLUE = "#1F5C8B"
ORANGE = "#D2701C"
GRID = "#D8DEE6"
BAND = "#F4F6F9"

# group, (internal est, lo, hi), (external est, lo, hi)
DATA = [
    ("Pooled", (0.765, 0.754, 0.775), (0.708, 0.686, 0.730)),
    ("Men",    (0.846, 0.827, 0.865), (0.768, 0.723, 0.813)),
    ("Women",  (0.673, 0.658, 0.688), (0.592, 0.559, 0.624)),
]
SERIES = [
    ("KNHANES internal", BLUE, "o"),
    ("NHANES external", ORANGE, "s"),
]

fig, ax = plt.subplots(figsize=(6.6, 3.0))
fig.subplots_adjust(left=0.12, right=0.70, top=0.88, bottom=0.16)

GAP = 0.46          # distance between the two markers of one group
ROW = 1.0           # distance between groups
ys, rows = [], []
for i, (name, internal, external) in enumerate(DATA):
    centre = -i * ROW
    ys.append(centre)
    rows.append((centre + GAP / 2, internal, 0))
    rows.append((centre - GAP / 2, external, 1))
for i, centre in enumerate(ys):
    if i % 2 == 0:
        ax.axhspan(centre - ROW / 2, centre + ROW / 2, color=BAND, zorder=0)

for y, (est, lo, hi), k in rows:
    label, colour, marker = SERIES[k]
    ax.plot([lo, hi], [y, y], color=colour, linewidth=1.7,
            solid_capstyle="butt", zorder=3)
    for x in (lo, hi):                      # interval caps
        ax.plot([x, x], [y - 0.10, y + 0.10], color=colour, linewidth=1.7, zorder=3)
    ax.plot([est], [y], marker=marker, markersize=6.2, color=colour,
            markeredgecolor="white", markeredgewidth=0.8, zorder=4)

ax.set_xlim(0.53, 0.90)
ax.set_ylim(min(ys) - ROW / 2, max(ys) + ROW / 2)
ax.set_xticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
ax.set_xticklabels(["0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90"],
                   fontsize=8.6, color=BODY)
ax.set_yticks(ys)
ax.set_yticklabels([d[0] for d in DATA], fontsize=9.6, color=INK)
ax.tick_params(axis="y", length=0, pad=6)
ax.tick_params(axis="x", length=3, color=GRID, labelcolor=BODY)

ax.set_xlabel("Area under the receiver operating characteristic curve (95% CI)",
              fontsize=9.2, color=INK, labelpad=6)

ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)

# ---- numeric column, the way a forest plot is normally read ---------------
TXT_X = 1.03
ax.text(TXT_X, max(ys) + ROW / 2 + 0.04, "AUROC (95% CI)", fontsize=8.6,
        fontweight="bold", color=INK, ha="left", va="bottom",
        transform=ax.get_yaxis_transform(), clip_on=False)
for y, (est, lo, hi), k in rows:
    ax.text(TXT_X, y, "%.3f (%.3f\u2013%.3f)" % (est, lo, hi), fontsize=8.4,
            color=SERIES[k][1], ha="left", va="center",
            transform=ax.get_yaxis_transform(), clip_on=False)

handles = [Line2D([], [], color=c, marker=m, markersize=6.2, linewidth=1.7,
                  markeredgecolor="white", markeredgewidth=0.8, label=lab)
           for lab, c, m in SERIES]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
          ncol=2, frameon=False, fontsize=8.8, handletextpad=0.5,
          columnspacing=1.8, labelcolor=BODY)

fig.savefig(OUT / "figure3_sex_transportability_v38.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print("wrote", OUT / "figure3_sex_transportability_v38.png")

"""Figure 7: validation-fold permutation importance for the male worked example.
Bars give magnitude on an unbroken linear scale and an adjacent column reports
each mean with its between-fold SD to four decimals. Rows are grouped by
whether the mean exceeds one between-fold SD from zero.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

OUT = Path(ANEMIA_ROOT) / "outputs"

INK = "#12263A"
BODY = "#22344A"
BAR = "#2A6296"
RULE = "#C3CCD6"
GRID = "#E1E7ED"

LABELS = {
    "age": "Age",
    "creatinine_raw": "Serum creatinine",
    "bmi": "Body mass index",
    "educ_cat": "Education category",
    "waist_height_ratio": "Waist-to-height ratio",
    "current_smoker": "Current smoking",
    "monthly_drinker": "Monthly alcohol use",
    "kidney_disease": "Kidney disease",
    "rheumatic_disease": "Rheumatoid arthritis",
    "thyroid_disease": "Thyroid disease",
}

d = pd.read_csv(OUT / "rf_permutation_importance_summary_v6.csv")
d = d.sort_values("mean_auc_decrease", ascending=False).reset_index(drop=True)
d["label"] = d["feature"].map(LABELS)
assert d["label"].notna().all(), "unmapped predictor name"

reproducible = d["mean_auc_decrease"] - d["between_fold_sd"] > 0
groups = [
    ("Dominant contribution", d.index == 0),
    ("Modest but reproducible", reproducible & (d.index != 0)),
    ("No reproducible contribution", ~reproducible),
]
assert [int(m.sum()) for _, m in groups] == [1, 4, 5], \
    'group sizes changed: %s' % [int(m.sum()) for _, m in groups]

FIG_W, FIG_H = 5.85, 3.05          # 5.85in is the placed width in the manuscript
NAME_PT, VAL_PT, HEAD_PT, TICK_PT = 8.6, 8.2, 8.2, 8.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.245, right=0.795, top=0.985, bottom=0.135)

# rows run top to bottom: a header, then the predictors it covers
rows = []
for title, mask in groups:
    rows.append(("head", title, None))
    for _, r in d[mask].iterrows():
        rows.append(("item", r["label"], r))

ys = [-i for i in range(len(rows))]
NAME_X, VAL_X = -0.02, 1.03         # axes fractions, outside the bar area
HEAD_X = -0.375

for y, (kind, text, r) in zip(ys, rows):
    if kind == "head":
        ax.plot([HEAD_X, 1.26], [y - 0.52, y - 0.52], transform=ax.get_yaxis_transform(),
                color=RULE, linewidth=0.8, clip_on=False, zorder=2)
        ax.text(HEAD_X, y, text, transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=HEAD_PT, fontweight="bold", color=INK,
                clip_on=False)
        continue
    ax.barh(y, r["mean_auc_decrease"], height=0.52, color=BAR, zorder=3)
    ax.text(NAME_X, y, text, transform=ax.get_yaxis_transform(), va="center",
            ha="right", fontsize=NAME_PT, color=BODY, clip_on=False)
    v, sd = r["mean_auc_decrease"], r["between_fold_sd"]
    txt = ("\u2212%.4f (%.4f)" % (abs(v), sd)) if v < 0 else ("%.4f (%.4f)" % (v, sd))
    ax.text(VAL_X, y, txt, transform=ax.get_yaxis_transform(), va="center",
            ha="left", fontsize=VAL_PT, color=BODY, clip_on=False)

ax.text(VAL_X, ys[0] + 1.15, "Mean (SD)", transform=ax.get_yaxis_transform(),
        va="center", ha="left", fontsize=VAL_PT, fontweight="bold", color=INK,
        clip_on=False)

ax.set_ylim(min(ys) - 0.8, max(ys) + 1.5)
ax.set_yticks([])
ax.set_xlim(0, 0.185)
ax.set_xticks([0, 0.05, 0.10, 0.15])
ax.set_xticklabels(["0", "0.05", "0.10", "0.15"], fontsize=TICK_PT, color=BODY)
ax.tick_params(axis="x", length=3, color=GRID, labelcolor=BODY)
ax.set_xlabel("Mean decrease in held-out AUROC after permutation",
              fontsize=NAME_PT, color=INK, labelpad=5)

ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(RULE)

fig.savefig(OUT / "figure7_permutation_importance_v38.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print("wrote", OUT / "figure7_permutation_importance_v38.png")

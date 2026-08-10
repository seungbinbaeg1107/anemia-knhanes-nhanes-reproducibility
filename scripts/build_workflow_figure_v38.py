"""Figure 1: intended-use workflow. Layout is in inches with the axes pinned to
the figure, and the figure is sized to the width it is placed at, so the point
sizes here are the ones that print. Every string is measured against the box
that holds it and the script raises rather than overflowing.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

OUT = Path(ANEMIA_ROOT) / "outputs"

INK = "#12263A"          # labels and panel titles
DETAIL = "#22344A"       # body text: dark enough to read at 8pt on white
BLUE = "#2A6296"
TEAL = "#177A62"
RED = "#9C3535"
GREY = "#7B8794"
HAIR = "#B9C5D2"

FIG_W, FIG_H = 6.30, 1.90          # 6.30in is the placed width in the manuscript
LABEL_PT, DETAIL_PT = 8.6, 8.2
TITLE_PT, SUB_PT, ITEM_PT = 8.6, 8.2, 7.9

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
rend = fig.canvas.get_renderer()


def text_width(txt, size, bold=False):
    """Width in inches of a string as it will be drawn."""
    t = fig.text(0, 0, txt, fontsize=size, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(renderer=rend).width / fig.dpi
    t.remove()
    return w


# ---------------------------------------------------------------- geometry
PAD = 0.02
TOP, BOTTOM = FIG_H - PAD, PAD
COL_W = 3.55                       # step column
GAP = 0.10
BX = PAD + COL_W + GAP
BW = FIG_W - BX - PAD              # panel column, 2.61in
LABEL_X = PAD + 0.16
DETAIL_X = LABEL_X + 0.98

BOX_H = 0.30
BOX_GAP = ((TOP - BOTTOM) - 5 * BOX_H) / 4
STEP = BOX_H + BOX_GAP
YS = [TOP - BOX_H - i * STEP for i in range(5)]
STEPS = [
    ("1  Trigger", "Creatinine result already available", "#E8F0F8", BLUE),
    ("2  Eligible", "No CBC ordered/resulted; tube ready", "#E8F0F8", BLUE),
    ("3  Model", "Random forest, 10 predictors + sex", "#E8F0F8", BLUE),
    ("4  Action", "Add CBC to existing specimen", "#DFEFE9", TEAL),
    # The constraint is laboratory capacity and reagent cost. Downstream harms
    # (false positives, further workup, incidental findings) were not studied.
    ("5  Constraint", "Capacity and reagent cost", "#EFF1F4", GREY),
]

for (label, detail, fill, edge), y in zip(STEPS, YS):
    ax.add_patch(FancyBboxPatch((PAD, y), COL_W, BOX_H,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=1.0, edgecolor=edge, facecolor=fill,
                                zorder=2))
    assert text_width(label, LABEL_PT, bold=True) <= DETAIL_X - LABEL_X - 0.06, \
        'step label runs into the detail column: "%s"' % label
    assert DETAIL_X + text_width(detail, DETAIL_PT) <= PAD + COL_W - 0.14, \
        'step detail overflows the box: "%s"' % detail
    ax.text(LABEL_X, y + BOX_H / 2, label, ha="left", va="center",
            fontsize=LABEL_PT, fontweight="bold", color=INK, zorder=3)
    ax.text(DETAIL_X, y + BOX_H / 2, detail, ha="left", va="center",
            fontsize=DETAIL_PT, color=DETAIL, zorder=3)

for y in YS[:-1]:
    ax.add_patch(FancyArrowPatch((PAD + COL_W / 2, y - 0.006),
                                 (PAD + COL_W / 2, y - BOX_GAP + 0.006),
                                 arrowstyle="-|>", mutation_scale=6,
                                 linewidth=0.9, color=GREY, zorder=1))

# ---- what these two surveys could and could not test --------------------
PH = ((TOP - BOTTOM) - 0.06) / 2


def panel(y, edge, face, title, colour, subtitle, items):
    ax.add_patch(FancyBboxPatch((BX, y), BW, PH,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=1.0, edgecolor=edge, facecolor=face,
                                zorder=0))
    lines = [(title, TITLE_PT, True, colour), (subtitle, SUB_PT, False, DETAIL)]
    lines += [(t, ITEM_PT, False, DETAIL) for t in items]
    heights = [pt / 72 for _, pt, _, _ in lines]
    leads = [0.052, 0.062] + [0.028] * (len(items) - 1)
    block = sum(heights) + sum(leads)
    cy = y + PH / 2 + block / 2
    for (txt, pt, bold, col), h in zip(lines, heights):
        assert text_width(txt, pt, bold) <= BW - 0.16, \
            'panel line too wide: "%s"' % txt
        ax.text(BX + BW / 2, cy - h / 2, txt, ha="center", va="center",
                fontsize=pt, fontweight="bold" if bold else "normal", color=col)
        cy -= h
        if leads:
            cy -= leads.pop(0)


panel(BOTTOM + PH + 0.06, HAIR, "#F7FAFD", "Evaluated here", TEAL,
      "Steps 3 and 4, in both surveys",
      ["discrimination  ·  calibration",
       "net benefit  ·  fixed thresholds",
       "referral burden  ·  covariate shift"])

panel(BOTTOM, "#E0C6C6", "#FDFAFA", "Not evaluated", RED,
      "Steps 1, 2 and 5, beyond survey data",
      ["specimen and trigger availability",
       "alert burden  ·  downstream testing",
       "downstream harms  ·  outcomes  ·  cost"])

fig.savefig(OUT / "figure1_workflow_v38.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote", OUT / "figure1_workflow_v38.png")

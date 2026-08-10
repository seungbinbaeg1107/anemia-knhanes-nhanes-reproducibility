"""Recompute the survey-weighted external estimates, their design-based intervals,
and the paired male-female contrasts from participant-level predictions, using
estimators written here rather than imported from the analysis scripts.
Verifies the evaluation, not the model fitting.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

OUT = Path(ANEMIA_ROOT) / "outputs"

nh = pd.read_parquet(OUT / "nh_analytic_full_v35.parquet").reset_index(drop=True)
pred = pd.read_csv(OUT / "sex_group_external_predictions_v8.csv")
dev = pd.read_csv(OUT / "sex_group_development_predictions_v8.csv")

pooled = pred[pred.group == "pooled"].sort_values("row_index").reset_index(drop=True)
assert len(pooled) == len(nh)
assert (pooled.outcome.to_numpy(int) == nh.label.astype(int).to_numpy()).all()

w = nh.survey_weight.to_numpy(float)
strata = nh.strata.astype(str).to_numpy()
psu = nh.psu.astype(str).to_numpy()
y = nh.label.astype(int).to_numpy()
p = pooled.prediction.to_numpy(float)
male_rows = set(pred.loc[pred.group == "male", "row_index"])
female_rows = set(pred.loc[pred.group == "female", "row_index"])
is_m = pooled.row_index.isin(male_rows).to_numpy()
is_f = pooled.row_index.isin(female_rows).to_numpy()


def cal(yv, pv, wv):
    lp = np.log(np.clip(pv, 1e-6, 1 - 1e-6) / np.clip(1 - pv, 1e-6, 1 - 1e-6))
    obj = lambda b: np.sum(wv * (np.logaddexp(0, b[0] + b[1] * lp)
                                 - yv * (b[0] + b[1] * lp)))
    return minimize(obj, np.array([0.0, 1.0]), method="BFGS").x


def reps(mask):
    """Stratified delete-one-PSU replicate weights, built from scratch."""
    valid = np.isfinite(w) & (w > 0) & mask
    base = np.where(valid, w, 0.0)
    for s in pd.unique(strata[valid]):
        ins = valid & (strata == s)
        us = pd.unique(psu[ins])
        m = len(us)
        if m < 2:
            continue
        for om in us:
            wr = base.copy()
            wr[ins & (psu == om)] = 0.0
            wr[ins & (psu != om)] *= m / (m - 1)
            yield s, m, wr


def jse(vals):
    df = pd.DataFrame(vals, columns=["s", "m", "v"]).dropna()
    var = sum(((int(g["m"].iloc[0]) - 1) / int(g["m"].iloc[0]))
              * np.sum((g["v"].to_numpy(float) - g["v"].mean()) ** 2)
              for _, g in df.groupby("s"))
    return float(np.sqrt(max(var, 0.0)))


def ci(point, vals):
    se = jse(vals)
    return point, point - 1.96 * se, point + 1.96 * se


rows, bad = [], 0


def cmp(label, got, want, tol):
    global bad
    ok = abs(got - want) <= tol
    rows.append((ok, label, got, want))
    if not ok:
        bad += 1


# ---- external AUROC by group, with design-based intervals
def auroc_ci(pv, mask):
    ok = mask & np.isfinite(w) & (w > 0)
    a = roc_auc_score(y[ok], pv[ok], sample_weight=w[ok])
    r = [(s, m, roc_auc_score(y[wr > 0], pv[wr > 0], sample_weight=wr[wr > 0]))
         for s, m, wr in reps(mask) if len(np.unique(y[wr > 0])) > 1]
    return ci(a, r)

for name, mask, want in (("pooled", np.ones(len(y), bool), (0.708, 0.686, 0.730)),
                         ("pooled model in men", is_m, (0.781, 0.734, 0.829)),
                         ("pooled model in women", is_f, (0.585, 0.554, 0.617))):
    est, lo, hi = auroc_ci(p, mask)
    cmp("external AUROC %s" % name, est, want[0], 0.0006)
    cmp("  lower bound", lo, want[1], 0.0011)
    cmp("  upper bound", hi, want[2], 0.0011)

# the sex-specific models, whose predictions live in their own rows
for g, mask, want in (("male", is_m, (0.768, 0.723, 0.813)),
                      ("female", is_f, (0.592, 0.559, 0.624))):
    sub = pred[pred.group == g].set_index("row_index")["prediction"]
    ps = pooled.row_index.map(sub).to_numpy(float)
    ps = np.where(np.isfinite(ps), ps, 0.0)
    est, lo, hi = auroc_ci(ps, mask)
    cmp("external AUROC %s-specific model" % g, est, want[0], 0.0006)
    cmp("  lower bound", lo, want[1], 0.0011)
    cmp("  upper bound", hi, want[2], 0.0011)

# ---- pooled calibration
okall = np.isfinite(w) & (w > 0)
b = cal(y[okall], p[okall], w[okall])
ri = [(s, m, cal(y[wr > 0], p[wr > 0], wr[wr > 0])[0]) for s, m, wr in reps(okall)]
rs = [(s, m, cal(y[wr > 0], p[wr > 0], wr[wr > 0])[1]) for s, m, wr in reps(okall)]
cmp("calibration intercept", b[0], -0.345, 0.0006)
cmp("calibration slope", b[1], 0.876, 0.0006)
cmp("  intercept lower", ci(b[0], ri)[1], -0.551, 0.0011)
cmp("  slope upper", ci(b[1], rs)[2], 0.971, 0.0011)

# ---- paired male-female contrasts, same replicate applied to both subgroups
def paired(fn):
    point = fn(np.isfinite(w) & (w > 0) & is_m, w) - fn(np.isfinite(w) & (w > 0) & is_f, w)
    out = []
    for s, m, wr in reps(np.isfinite(w) & (w > 0)):
        mm, ff = (wr > 0) & is_m, (wr > 0) & is_f
        if len(np.unique(y[mm])) < 2 or len(np.unique(y[ff])) < 2:
            continue
        out.append((s, m, fn(mm, wr) - fn(ff, wr)))
    return ci(point, out)


auc_of = lambda mask, wv: roc_auc_score(y[mask], p[mask], sample_weight=wv[mask])
d_auc, lo_a, hi_a = paired(auc_of)
cmp("male-female AUROC difference", d_auc, 0.196, 0.0011)
cmp("  lower bound", lo_a, 0.136, 0.0011)
cmp("  upper bound", hi_a, 0.257, 0.0011)

THR = 0.0900230
fnr_of = lambda mask, wv: (np.sum(wv[mask & (y == 1) & (p < THR)])
                           / np.sum(wv[mask & (y == 1)]))
d_fnr, lo_f, hi_f = paired(fnr_of)
cmp("male-female FNR difference", d_fnr, 0.405, 0.0011)
cmp("  lower bound", lo_f, 0.312, 0.0011)
cmp("  upper bound", hi_f, 0.499, 0.0011)

# ---- internal out-of-fold AUROCs
for g, want in (("pooled", 0.765), ("male", 0.846), ("female", 0.673)):
    s = dev[dev.group == g]
    cmp("internal OOF AUROC %s" % g,
        roc_auc_score(s.outcome.values, s.prediction.values), want, 0.0006)

print("recomputed from participant-level data with independent estimators\n")
for ok, label, got, want in rows:
    print("%-5s %-32s recomputed %8.4f   reported %8.3f"
          % ("PASS" if ok else "FAIL", label, got, want))
print("\n%s  (%d of %d)"
      % ("ALL RECOMPUTATIONS AGREE" if bad == 0 else "MISMATCH", bad, len(rows)))
raise SystemExit(1 if bad else 0)

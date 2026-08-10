"""Design-based male-female performance contrasts and an in-sample external
recalibration sensitivity analysis. No model is refitted.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

OUT = Path(ANEMIA_ROOT) / "outputs"
PRED = OUT / "sex_group_external_predictions_v8.csv"


# ----------------------------------------------------------------- estimators
def calibration_fit(y, p, w):
    """Weighted logistic recalibration of the linear predictor (Cox)."""
    y = np.asarray(y, int)
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))

    def objective(beta):
        eta = beta[0] + beta[1] * lp
        return np.sum(w * (np.logaddexp(0, eta) - y * eta))

    return np.asarray(minimize(objective, np.array([0.0, 1.0]), method="BFGS").x, float)


def apply_recal(p, beta):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    eta = beta[0] + beta[1] * lp
    return 1.0 / (1.0 + np.exp(-eta))


def group_metrics(y, p, w, thresholds):
    """Survey-weighted performance; thresholds is {name: cutoff}."""
    out = {
        "prevalence": float(np.average(y, weights=w)),
        "auc": float(roc_auc_score(y, p, sample_weight=w)),
        "brier": float(np.average((y - p) ** 2, weights=w)),
    }
    cal = calibration_fit(y, p, w)
    out["calibration_intercept"] = float(cal[0])
    out["calibration_slope"] = float(cal[1])
    for name, thr in thresholds.items():
        flag = p >= thr
        tp = w[(y == 1) & flag].sum()
        fp = w[(y == 0) & flag].sum()
        fn = w[(y == 1) & ~flag].sum()
        tn = w[(y == 0) & ~flag].sum()
        out[f"sensitivity_{name}"] = float(tp / (tp + fn))
        out[f"specificity_{name}"] = float(tn / (tn + fp))
        out[f"fnr_{name}"] = float(fn / (tp + fn))
        out[f"ppv_{name}"] = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        out[f"flagged_{name}"] = float((tp + fp) / w.sum())
    return out


def replicate_weights(frame):
    """Yield (stratum, m, replicate weight vector) for the delete-one-PSU jackknife."""
    w = frame.survey_weight.to_numpy(float)
    strata = frame.strata.astype(str).to_numpy()
    psu = frame.psu.astype(str).to_numpy()
    valid = np.isfinite(w) & (w > 0)
    base = np.where(valid, w, 0.0)
    for stratum in pd.unique(strata[valid]):
        in_s = valid & (strata == stratum)
        psus = pd.unique(psu[in_s])
        m = len(psus)
        if m < 2:
            continue
        for omitted in psus:
            wr = base.copy()
            wr[in_s & (psu == omitted)] = 0.0
            wr[in_s & (psu != omitted)] *= m / (m - 1)
            yield stratum, m, wr


def jackknife_se(point, replicates):
    """Stratified delete-one-PSU jackknife SE from a list of (stratum, m, value)."""
    df = pd.DataFrame(replicates, columns=["stratum", "m", "value"]).dropna()
    var = 0.0
    for _, g in df.groupby("stratum"):
        m = int(g["m"].iloc[0])
        v = g["value"].to_numpy(float)
        var += ((m - 1) / m) * np.sum((v - v.mean()) ** 2)
    return float(np.sqrt(max(var, 0.0)))


# ----------------------------------------------------------------- load data
pred = pd.read_csv(PRED)
pooled = pred[pred.group == "pooled"].reset_index(drop=True)
male_idx = set(pred.loc[pred.group == "male", "row_index"])
female_idx = set(pred.loc[pred.group == "female", "row_index"])
assert male_idx.isdisjoint(female_idx)
assert male_idx | female_idx == set(pooled.row_index)

is_male = pooled.row_index.isin(male_idx).to_numpy()
is_female = pooled.row_index.isin(female_idx).to_numpy()

y_pool = pooled.outcome.to_numpy(int)
p_pool = pooled.prediction.to_numpy(float)

# Development-derived pooled operating points (Table S6, unweighted OOF basis)
POOLED_THR = {"sens90": 0.09002304336179877, "spec90": 0.18319670021932172}

# ------------------------------------------------- Panel A: pooled model by sex
rows = []
qa = {}

point_m = group_metrics(y_pool[is_male], p_pool[is_male],
                        pooled.survey_weight.to_numpy(float)[is_male], POOLED_THR)
point_f = group_metrics(y_pool[is_female], p_pool[is_female],
                        pooled.survey_weight.to_numpy(float)[is_female], POOLED_THR)
point_all = group_metrics(y_pool, p_pool, pooled.survey_weight.to_numpy(float), POOLED_THR)
qa["pooled_model_overall_auc"] = point_all["auc"]

rep_m, rep_f, rep_d = {}, {}, {}
for stratum, m, wr in replicate_weights(pooled):
    for mask, store in ((is_male, rep_m), (is_female, rep_f)):
        sub = mask & (wr > 0)
        vals = group_metrics(y_pool[sub], p_pool[sub], wr[sub], POOLED_THR)
        for k, v in vals.items():
            store.setdefault(k, []).append((stratum, m, v))
    for k in rep_m:
        rep_d.setdefault(k, []).append(
            (stratum, m, rep_m[k][-1][2] - rep_f[k][-1][2]))

for metric in point_m:
    diff = point_m[metric] - point_f[metric]
    se = jackknife_se(diff, rep_d[metric])
    rows.append({
        "panel": "A. Pooled (primary) model evaluated by sex",
        "metric": metric,
        "men": point_m[metric],
        "men_ci_low": point_m[metric] - 1.96 * jackknife_se(point_m[metric], rep_m[metric]),
        "men_ci_high": point_m[metric] + 1.96 * jackknife_se(point_m[metric], rep_m[metric]),
        "women": point_f[metric],
        "women_ci_low": point_f[metric] - 1.96 * jackknife_se(point_f[metric], rep_f[metric]),
        "women_ci_high": point_f[metric] + 1.96 * jackknife_se(point_f[metric], rep_f[metric]),
        "difference_men_minus_women": diff,
        "diff_ci_low": diff - 1.96 * se,
        "diff_ci_high": diff + 1.96 * se,
    })

# ------------------------------- Panel B: sex-specific models in their own group
male = pred[pred.group == "male"].reset_index(drop=True)
female = pred[pred.group == "female"].reset_index(drop=True)
# each sex-specific model has its own development thresholds (Table S6)
THR_M = {"own_sens90": 0.0255143226621153, "own_spec90": 0.1469586592660943}
THR_F = {"own_sens90": 0.09667004230825556, "own_spec90": 0.20070442261402458}

pm = group_metrics(male.outcome.to_numpy(int), male.prediction.to_numpy(float),
                   male.survey_weight.to_numpy(float), THR_M)
pf = group_metrics(female.outcome.to_numpy(int), female.prediction.to_numpy(float),
                   female.survey_weight.to_numpy(float), THR_F)
qa["sex_specific_male_auc"] = pm["auc"]
qa["sex_specific_female_auc"] = pf["auc"]

rm, rf_ = {}, {}
for stratum, m, wr in replicate_weights(male):
    sub = wr > 0
    for k, v in group_metrics(male.outcome.to_numpy(int)[sub],
                              male.prediction.to_numpy(float)[sub], wr[sub], THR_M).items():
        rm.setdefault(k, []).append((stratum, m, v))
for stratum, m, wr in replicate_weights(female):
    sub = wr > 0
    for k, v in group_metrics(female.outcome.to_numpy(int)[sub],
                              female.prediction.to_numpy(float)[sub], wr[sub], THR_F).items():
        rf_.setdefault(k, []).append((stratum, m, v))

for metric in ["prevalence", "auc", "brier", "calibration_intercept", "calibration_slope"]:
    rows.append({
        "panel": "B. Sex-specific models in their own group (exploratory)",
        "metric": metric,
        "men": pm[metric],
        "men_ci_low": pm[metric] - 1.96 * jackknife_se(pm[metric], rm[metric]),
        "men_ci_high": pm[metric] + 1.96 * jackknife_se(pm[metric], rm[metric]),
        "women": pf[metric],
        "women_ci_low": pf[metric] - 1.96 * jackknife_se(pf[metric], rf_[metric]),
        "women_ci_high": pf[metric] + 1.96 * jackknife_se(pf[metric], rf_[metric]),
        "difference_men_minus_women": pm[metric] - pf[metric],
        "diff_ci_low": np.nan,   # independent subsamples: not a paired contrast
        "diff_ci_high": np.nan,
    })

pd.DataFrame(rows).to_csv(OUT / "sex_contrast_external_v31.csv", index=False)

# ------------------------------------------- recalibration sensitivity analysis
recal_rows = []
for name, frame in (("pooled", pooled), ("male", male), ("female", female)):
    y = frame.outcome.to_numpy(int)
    p = frame.prediction.to_numpy(float)
    w = frame.survey_weight.to_numpy(float)
    beta = calibration_fit(y, p, w)
    p_rc = apply_recal(p, beta)
    base = group_metrics(y, p, w, {})
    rc = group_metrics(y, p_rc, w, {})
    recal_rows.append({
        "group": name,
        "recalibration_intercept_alpha": float(beta[0]),
        "recalibration_slope_beta": float(beta[1]),
        "auc_unrecalibrated": base["auc"],
        "auc_recalibrated": rc["auc"],
        "brier_unrecalibrated": base["brier"],
        "brier_recalibrated": rc["brier"],
        "calibration_intercept_unrecalibrated": base["calibration_intercept"],
        "calibration_intercept_recalibrated": rc["calibration_intercept"],
        "calibration_slope_unrecalibrated": base["calibration_slope"],
        "calibration_slope_recalibrated": rc["calibration_slope"],
    })
pd.DataFrame(recal_rows).to_csv(OUT / "recalibration_sensitivity_v31.csv", index=False)

# ------------------------- pooled net benefit, unrecalibrated vs recalibrated
y = pooled.outcome.to_numpy(int)
w = pooled.survey_weight.to_numpy(float)
p = pooled.prediction.to_numpy(float)
beta_pool = calibration_fit(y, p, w)
p_rc = apply_recal(p, beta_pool)
W = w.sum()
dca = []
for thr in np.round(np.arange(0.01, 0.3001, 0.01), 4):
    row = {"threshold": float(thr)}
    for label, pv in (("random_forest", p), ("random_forest_recalibrated", p_rc)):
        flag = pv >= thr
        tp = w[(y == 1) & flag].sum()
        fp = w[(y == 0) & flag].sum()
        row[label] = float(tp / W - (fp / W) * (thr / (1 - thr)))
    tp = w[y == 1].sum()
    fp = w[y == 0].sum()
    row["cbc_test_all"] = float(tp / W - (fp / W) * (thr / (1 - thr)))
    row["cbc_test_none"] = 0.0
    dca.append(row)
pd.DataFrame(dca).to_csv(OUT / "dca_clinical_range_v31.csv", index=False)

json.dump(qa, open(OUT / "sex_contrast_qa_v31.json", "w"), indent=2)
print(json.dumps(qa, indent=2))
print("\nreproduction targets: pooled 0.708, male 0.768, female 0.592")

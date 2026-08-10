"""Missing-data sensitivity analyses: complete case, chained-equation multiple
imputation, missingness indicators, and the position of the transported
development medians in the external distribution.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

warnings.filterwarnings("ignore")

OUT = Path(ANEMIA_ROOT) / "outputs"
FEATURES = ["age", "bmi", "waist_height_ratio", "creatinine_raw", "educ_cat",
            "kidney_disease", "thyroid_disease", "rheumatic_disease",
            "current_smoker", "monthly_drinker"]

kn = pd.read_parquet(OUT / "kn_analytic_full_v35.parquet")
nh = pd.read_parquet(OUT / "nh_analytic_full_v35.parquet")
pred = pd.read_csv(OUT / "sex_group_external_predictions_v8.csv")

pooled = pred[pred.group == "pooled"].reset_index(drop=True)
assert len(pooled) == len(nh)
assert (pooled.outcome.to_numpy(int) == nh.label.astype(int).to_numpy()).all(), "row order"

male_idx = set(pred.loc[pred.group == "male", "row_index"])
female_idx = set(pred.loc[pred.group == "female", "row_index"])
is_male = pooled.row_index.isin(male_idx).to_numpy()
is_female = pooled.row_index.isin(female_idx).to_numpy()
assert is_male.sum() == 2583 and is_female.sum() == 3148


# ----------------------------------------------------------------- estimators
def calibration_fit(y, p, w):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))

    def obj(b):
        eta = b[0] + b[1] * lp
        return np.sum(w * (np.logaddexp(0, eta) - y * eta))

    return np.asarray(minimize(obj, np.array([0.0, 1.0]), method="BFGS").x, float)


def replicate_weights(w, strata, psu, mask):
    valid = np.isfinite(w) & (w > 0) & mask
    base = np.where(valid, w, 0.0)
    for s in pd.unique(strata[valid]):
        in_s = valid & (strata == s)
        psus = pd.unique(psu[in_s])
        m = len(psus)
        if m < 2:
            continue
        for omitted in psus:
            wr = base.copy()
            wr[in_s & (psu == omitted)] = 0.0
            wr[in_s & (psu != omitted)] *= m / (m - 1)
            yield s, m, wr


def jack_se(reps):
    df = pd.DataFrame(reps, columns=["s", "m", "v"]).dropna()
    var = 0.0
    for _, g in df.groupby("s"):
        m = int(g["m"].iloc[0])
        v = g["v"].to_numpy(float)
        var += ((m - 1) / m) * np.sum((v - v.mean()) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def evaluate(mask, p_col):
    """Survey-weighted AUROC / calibration on a subset, with jackknife CIs."""
    w = nh.survey_weight.to_numpy(float)
    strata = nh.strata.astype(str).to_numpy()
    psu = nh.psu.astype(str).to_numpy()
    y = nh.label.astype(int).to_numpy()
    p = p_col
    ok = mask & np.isfinite(w) & (w > 0)
    auc = roc_auc_score(y[ok], p[ok], sample_weight=w[ok])
    cal = calibration_fit(y[ok], p[ok], w[ok])
    ra, ri, rs = [], [], []
    for s, m, wr in replicate_weights(w, strata, psu, mask):
        sub = wr > 0
        if len(np.unique(y[sub])) < 2:
            continue
        ra.append((s, m, roc_auc_score(y[sub], p[sub], sample_weight=wr[sub])))
        c = calibration_fit(y[sub], p[sub], wr[sub])
        ri.append((s, m, c[0]))
        rs.append((s, m, c[1]))
    return {
        "n": int(ok.sum()), "events": int(y[ok].sum()),
        "auc": float(auc), "auc_se": jack_se(ra),
        "cal_intercept": float(cal[0]), "cal_intercept_se": jack_se(ri),
        "cal_slope": float(cal[1]), "cal_slope_se": jack_se(rs),
    }


# ------------------------------------------- 1. complete-case, all three groups
complete = nh[FEATURES].notna().all(axis=1).to_numpy()
print("NHANES complete cases: %d / %d (%.1f%%)"
      % (complete.sum(), len(nh), 100 * complete.mean()))

rows = []
GROUPS = {"pooled": np.ones(len(nh), bool), "male": is_male, "female": is_female}
PRED_COL = {
    "pooled": pooled.prediction.to_numpy(float),
    "male": None,
    "female": None,
}
# sex-specific model predictions, mapped back onto the pooled row order
for g in ("male", "female"):
    sub = pred[pred.group == g]
    lut = dict(zip(sub.row_index, sub.prediction))
    PRED_COL[g] = pooled.row_index.map(lut).to_numpy(float)

for gname, gmask in GROUPS.items():
    for label, mask in (("all eligible", gmask), ("complete cases", gmask & complete)):
        r = evaluate(mask, PRED_COL[gname])
        rows.append({"group": gname, "cohort": label, **r})
        print("  %-7s %-14s n=%4d events=%3d AUROC %.3f (%.3f-%.3f)"
              % (gname, label, r["n"], r["events"], r["auc"],
                 r["auc"] - 1.96 * r["auc_se"], r["auc"] + 1.96 * r["auc_se"]))

# paired complete-case vs full difference, same replicates
for gname, gmask in GROUPS.items():
    p = PRED_COL[gname]
    y = nh.label.astype(int).to_numpy()
    w = nh.survey_weight.to_numpy(float)
    full = gmask & np.isfinite(w) & (w > 0)
    cc = full & complete
    d = (roc_auc_score(y[cc], p[cc], sample_weight=w[cc])
         - roc_auc_score(y[full], p[full], sample_weight=w[full]))
    reps = []
    for s, m, wr in replicate_weights(w, nh.strata.astype(str).to_numpy(),
                                      nh.psu.astype(str).to_numpy(), gmask):
        a = wr > 0
        b = a & complete
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        reps.append((s, m,
                     roc_auc_score(y[b], p[b], sample_weight=wr[b])
                     - roc_auc_score(y[a], p[a], sample_weight=wr[a])))
    se = jack_se(reps)
    rows.append({"group": gname, "cohort": "complete-case minus all eligible",
                 "n": int(cc.sum()), "events": int(y[cc].sum()),
                 "auc": d, "auc_se": se,
                 "cal_intercept": np.nan, "cal_intercept_se": np.nan,
                 "cal_slope": np.nan, "cal_slope_se": np.nan})
    print("  %-7s complete-case - all: %+.3f (%+.3f to %+.3f)"
          % (gname, d, d - 1.96 * se, d + 1.96 * se))

pd.DataFrame(rows).to_csv(OUT / "complete_case_external_v35.csv", index=False)

# ------------------------- 2. what was imputed, and what it was imputed with
desc = []
for f in FEATURES:
    med = kn[f].median()
    k_obs, n_obs = kn[f].dropna(), nh[f].dropna()
    desc.append({
        "predictor": f,
        "knhanes_median_transported": float(med),
        "knhanes_missing_pct": 100 * float(kn[f].isna().mean()),
        "nhanes_missing_pct": 100 * float(nh[f].isna().mean()),
        "knhanes_observed_mean": float(k_obs.mean()),
        "nhanes_observed_mean": float(n_obs.mean()),
        "nhanes_observed_median": float(n_obs.median()),
        "imputed_minus_nhanes_median": float(med - n_obs.median()),
        "nhanes_pctile_of_transported_median":
            100 * float((n_obs < med).mean()),
    })
pd.DataFrame(desc).to_csv(OUT / "imputed_value_description_v35.csv", index=False)
print("\nTransported KNHANES median vs NHANES observed distribution:")
print(pd.DataFrame(desc)[["predictor", "knhanes_median_transported",
                          "nhanes_observed_median", "nhanes_missing_pct",
                          "nhanes_pctile_of_transported_median"]]
      .round(3).to_string(index=False))

# ------------------------------------------- 3. missingness co-occurrence
co = []
for src_name, frame in (("KNHANES", kn), ("NHANES", nh)):
    miss = frame[FEATURES].isna()
    counts = miss.sum(axis=1)
    co.append({"source": src_name, "n": len(frame),
               "any_missing_pct": 100 * float((counts > 0).mean()),
               "mean_missing_predictors": float(counts.mean()),
               "max_missing_predictors": int(counts.max()),
               "complete_case_pct": 100 * float((counts == 0).mean())})
pd.DataFrame(co).to_csv(OUT / "missingness_cooccurrence_v35.csv", index=False)
print("\n", pd.DataFrame(co).round(2).to_string(index=False))

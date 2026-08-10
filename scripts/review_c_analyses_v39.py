"""Three sensitivity analyses: performance with participants aged 80 years or
older excluded from both surveys, development fitted with survey weights as
observation weights, and the development-process cluster bootstrap for the
male and female analysis groups.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

warnings.filterwarnings("ignore")

OUT = Path(ANEMIA_ROOT) / "outputs"
RESULT = OUT / "review_c_analyses_v39.json"

FEATURES = ["age", "bmi", "waist_height_ratio", "creatinine_raw", "educ_cat",
            "kidney_disease", "thyroid_disease", "rheumatic_disease",
            "current_smoker", "monthly_drinker"]
GRID = {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]}
PUBLISHED_INTERNAL = {"pooled": 0.765, "male": 0.846, "female": 0.673}
PUBLISHED_EXTERNAL = {"pooled": 0.708, "male": 0.768, "female": 0.592}
B_SEX = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# ------------------------------------------------------------------ data
kn = pd.read_parquet(OUT / "kn_analytic_cache_v34.parquet").reset_index(drop=True)
nh = pd.read_parquet(OUT / "nh_analytic_full_v35.parquet").reset_index(drop=True)

# development weights live with the stored predictions, not in the cache
_p = pd.read_csv(OUT / "sex_group_development_predictions_v8.csv")
_p = _p[_p.group == "pooled"].sort_values("row_index").reset_index(drop=True)
assert len(_p) == len(kn)
assert (_p.cluster.astype(str).values == kn.cluster.astype(str).values).all(), "row order"
assert (_p.outcome.astype(int).values == kn.label.astype(int).values).all(), "row order"
kn["survey_weight"] = _p.survey_weight.to_numpy(float)

print("KNHANES %d rows / %d events   NHANES %d rows / %d events"
      % (len(kn), int(kn.label.sum()), len(nh), int(nh.label.sum())), flush=True)

results = json.loads(RESULT.read_text()) if RESULT.exists() else {}


def save():
    json.dump(results, open(RESULT, "w"), indent=2)


# ------------------------------------------------------------- machinery
def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=500,
                                                    random_state=42, n_jobs=1))])


def frames(kn_df, nh_df, group):
    """Development and external matrices for one analysis group."""
    if group == "male":
        d, e, feats = kn_df[kn_df.sex == 1], nh_df[nh_df.sex == 1], FEATURES
    elif group == "female":
        d, e, feats = kn_df[kn_df.sex == 0], nh_df[nh_df.sex == 0], FEATURES
    else:
        d, e, feats = kn_df, nh_df, ["sex"] + FEATURES
    return (d[feats].to_numpy(float), d.label.astype(int).to_numpy(),
            d.cluster.to_numpy().astype(str), d.survey_weight.to_numpy(float),
            e[feats].to_numpy(float), e.reset_index(drop=True))


def nested_oof(X, y, g, seed_outer=42, seed_inner=1, w=None):
    """One complete grouped nested CV. w, if given, weights the model fit."""
    oof = np.full(len(X), np.nan)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed_outer)
    try:
        splits_outer = list(outer.split(X, y, g))
    except ValueError:
        return oof
    for tr, te in splits_outer:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return np.full(len(X), np.nan)
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed_inner)
        try:
            splits = list(inner.split(X[tr], y[tr], g[tr]))
        except ValueError:
            return np.full(len(X), np.nan)
        gs = GridSearchCV(pipe(), GRID, scoring="roc_auc", cv=splits, n_jobs=-1)
        if w is None:
            gs.fit(X[tr], y[tr])
        else:
            gs.fit(X[tr], y[tr], clf__sample_weight=w[tr])
        oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    return oof


def final_fit(X, y, g, w=None):
    """Refit on the whole development set, tuning once, as the focal models do."""
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=1)
    gs = GridSearchCV(pipe(), GRID, scoring="roc_auc",
                      cv=list(inner.split(X, y, g)), n_jobs=-1)
    if w is None:
        gs.fit(X, y)
    else:
        gs.fit(X, y, clf__sample_weight=w)
    return gs.best_estimator_, gs.best_params_


# --------------------------------------------- survey-weighted external eval
def calibration_fit(y, p, w):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))

    def obj(b):
        eta = b[0] + b[1] * lp
        return np.sum(w * (np.logaddexp(0, eta) - y * eta))

    return np.asarray(minimize(obj, np.array([0.0, 1.0]), method="BFGS").x, float)


def replicate_weights(w, strata, psu):
    valid = np.isfinite(w) & (w > 0)
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
    for _, grp in df.groupby("s"):
        m = int(grp["m"].iloc[0])
        v = grp["v"].to_numpy(float)
        var += ((m - 1) / m) * np.sum((v - v.mean()) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def external_eval(ext_df, p):
    w = ext_df.survey_weight.to_numpy(float)
    strata = ext_df.strata.astype(str).to_numpy()
    psu = ext_df.psu.astype(str).to_numpy()
    y = ext_df.label.astype(int).to_numpy()
    ok = np.isfinite(w) & (w > 0)
    auc = roc_auc_score(y[ok], p[ok], sample_weight=w[ok])
    cal = calibration_fit(y[ok], p[ok], w[ok])
    ra, ri, rs = [], [], []
    for s, m, wr in replicate_weights(w, strata, psu):
        sub = wr > 0
        if len(np.unique(y[sub])) < 2:
            continue
        ra.append((s, m, roc_auc_score(y[sub], p[sub], sample_weight=wr[sub])))
        c = calibration_fit(y[sub], p[sub], wr[sub])
        ri.append((s, m, c[0]))
        rs.append((s, m, c[1]))
    se = jack_se(ra)
    return {"n": int(ok.sum()), "events": int(y[ok].sum()),
            "auc": float(auc), "auc_se": se,
            "auc_ci": [float(auc - 1.96 * se), float(auc + 1.96 * se)],
            "cal_intercept": float(cal[0]), "cal_intercept_se": jack_se(ri),
            "cal_slope": float(cal[1]), "cal_slope_se": jack_se(rs)}


# ==================================================================== C1
if "c1_age_topcode" not in results:
    print("\n########## C1  age >=80 excluded ##########", flush=True)
    t0 = time.time()
    kn_r = kn[kn.age < 80].reset_index(drop=True)
    nh_r = nh[nh.age < 80].reset_index(drop=True)
    print("  KNHANES %d -> %d (%d removed);  NHANES %d -> %d (%d removed)"
          % (len(kn), len(kn_r), len(kn) - len(kn_r),
             len(nh), len(nh_r), len(nh) - len(nh_r)), flush=True)
    entry = {"knhanes_removed": int(len(kn) - len(kn_r)),
             "nhanes_removed": int(len(nh) - len(nh_r)), "groups": {}}
    for group in ["pooled", "male", "female"]:
        X, y, g, w, Xe, ext = frames(kn_r, nh_r, group)
        oof = nested_oof(X, y, g)
        ok = np.isfinite(oof)
        internal = float(roc_auc_score(y[ok], oof[ok]))
        model, params = final_fit(X, y, g)
        p = model.predict_proba(Xe)[:, 1]
        ext_res = external_eval(ext, p)
        entry["groups"][group] = {
            "n_dev": int(len(X)), "events_dev": int(y.sum()),
            "internal_oof_auc": internal,
            "published_internal": PUBLISHED_INTERNAL[group],
            "external": ext_res, "published_external": PUBLISHED_EXTERNAL[group],
            "final_params": {k: str(v) for k, v in params.items()}}
        print("  %-7s internal %.4f (published %.3f) | external %.4f (%.4f-%.4f)"
              " (published %.3f)"
              % (group, internal, PUBLISHED_INTERNAL[group], ext_res["auc"],
                 ext_res["auc_ci"][0], ext_res["auc_ci"][1],
                 PUBLISHED_EXTERNAL[group]), flush=True)
    entry["minutes"] = round((time.time() - t0) / 60, 1)
    results["c1_age_topcode"] = entry
    save()

# ==================================================================== C2
if "c2_weighted_development" not in results:
    print("\n########## C2  survey-weighted development fit ##########", flush=True)
    t0 = time.time()
    entry = {"note": ("Random forest refit with KNHANES examination weights as "
                      "observation weights. Hyperparameter selection kept the "
                      "unweighted inner AUROC, so only the fit is reweighted."),
             "groups": {}}
    for group in ["pooled", "male", "female"]:
        X, y, g, w, Xe, ext = frames(kn, nh, group)
        oof = nested_oof(X, y, g, w=w)
        ok = np.isfinite(oof)
        unw = float(roc_auc_score(y[ok], oof[ok]))
        wtd = float(roc_auc_score(y[ok], oof[ok], sample_weight=w[ok]))
        # the sensitivity-oriented operating point, re-derived under weighting
        order = np.argsort(-oof[ok])
        ys, ws = y[ok][order], w[ok][order]
        tp = np.cumsum(ys * ws) / np.sum(ys * ws)
        cut = np.searchsorted(tp, 0.90)
        thr = float(oof[ok][order][min(cut, len(ys) - 1)])
        flag = oof[ok] >= thr
        sens = float(np.sum(w[ok][flag] * y[ok][flag]) / np.sum(w[ok] * y[ok]))
        spec = float(np.sum(w[ok][~flag] * (1 - y[ok][~flag]))
                     / np.sum(w[ok] * (1 - y[ok])))
        referred = float(np.sum(w[ok][flag]) / np.sum(w[ok]))
        model, params = final_fit(X, y, g, w=w)
        p = model.predict_proba(Xe)[:, 1]
        ext_res = external_eval(ext, p)
        entry["groups"][group] = {
            "internal_oof_auc_unweighted": unw,
            "internal_oof_auc_weighted": wtd,
            "published_internal": PUBLISHED_INTERNAL[group],
            "weighted_sensitivity_threshold": thr,
            "weighted_sensitivity": sens, "weighted_specificity": spec,
            "weighted_referred_fraction": referred,
            "external": ext_res, "published_external": PUBLISHED_EXTERNAL[group],
            "final_params": {k: str(v) for k, v in params.items()}}
        print("  %-7s internal unw %.4f / wtd %.4f (published %.3f) | external "
              "%.4f (%.4f-%.4f) (published %.3f)"
              % (group, unw, wtd, PUBLISHED_INTERNAL[group], ext_res["auc"],
                 ext_res["auc_ci"][0], ext_res["auc_ci"][1],
                 PUBLISHED_EXTERNAL[group]), flush=True)
    entry["minutes"] = round((time.time() - t0) / 60, 1)
    results["c2_weighted_development"] = entry
    save()

# ==================================================================== C3
print("\n########## C3  development bootstrap, men and women (B=%d) ##########"
      % B_SEX, flush=True)
for group in ["male", "female"]:
    key = "c3_development_bootstrap_%s" % group
    if key in results:
        print("skip %s (already done)" % group, flush=True)
        continue
    X, y, g, w, _, _ = frames(kn, nh, group)
    rng = np.random.default_rng(20260807)
    clusters = pd.unique(g)
    idx_by = {c: np.flatnonzero(g == c) for c in clusters}
    boot, t0 = [], time.time()
    csv = OUT / ("development_bootstrap_%s_v39.csv" % group)
    for b in range(B_SEX):
        draw = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([idx_by[c] for c in draw])
        # each copy of a duplicated cluster keeps the ORIGINAL label, so grouped
        # CV sends every copy to the same fold. Relabelling the copies would put
        # identical participants in train and validation and inflate the AUROC.
        gb = np.concatenate([np.full(len(idx_by[c]), str(c)) for c in draw])
        oof = nested_oof(X[idx], y[idx], gb)
        ok = np.isfinite(oof)
        auc = (float(roc_auc_score(y[idx][ok], oof[ok]))
               if ok.any() and len(np.unique(y[idx][ok])) > 1 else np.nan)
        boot.append(auc)
        pd.DataFrame({"replicate": range(len(boot)), "auc": boot}).to_csv(csv, index=False)
        if (b + 1) % 5 == 0:
            f = np.asarray([v for v in boot if np.isfinite(v)], float)
            print("  %s %3d/%d  median %.4f  2.5-97.5%% %.4f-%.4f  [%.0f min]"
                  % (group, b + 1, B_SEX, np.median(f), np.percentile(f, 2.5),
                     np.percentile(f, 97.5), (time.time() - t0) / 60), flush=True)
    f = np.asarray([v for v in boot if np.isfinite(v)], float)
    results[key] = {"B": int(len(f)), "median": float(np.median(f)),
                    "ci_low": float(np.percentile(f, 2.5)),
                    "ci_high": float(np.percentile(f, 97.5)),
                    "sd": float(f.std(ddof=1)),
                    "published_conditional_point": PUBLISHED_INTERNAL[group],
                    "minutes": round((time.time() - t0) / 60, 1)}
    save()

save()
print("\nDONE ->", RESULT, flush=True)

"""Quantify the uncertainty the conditional fixed-prediction intervals omit, in
two ways: repeated grouped nested cross-validation over random fold partitions,
and a cluster bootstrap that reruns the complete development process, including
hyperparameter selection, inside every replicate.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

warnings.filterwarnings("ignore")

OUT = Path(ANEMIA_ROOT) / "outputs"
CACHE = OUT / "kn_analytic_cache_v34.parquet"
RESULT = OUT / "development_uncertainty_v34.json"

R = int(sys.argv[1]) if len(sys.argv) > 1 else 25
B_POOLED = int(sys.argv[2]) if len(sys.argv) > 2 else 100

FEATURES = ["age", "bmi", "waist_height_ratio", "creatinine_raw", "educ_cat",
            "kidney_disease", "thyroid_disease", "rheumatic_disease",
            "current_smoker", "monthly_drinker"]
GRID = {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]}
PUBLISHED = {"pooled": 0.765, "male": 0.846, "female": 0.673}

kn = pd.read_parquet(CACHE)
print("rows %d events %d clusters %d"
      % (len(kn), int(kn.label.sum()), kn.cluster.nunique()), flush=True)


def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=500,
                                                    random_state=42, n_jobs=1))])


def frame_for(group):
    if group == "male":
        d, feats = kn.loc[kn.sex == 1], FEATURES
    elif group == "female":
        d, feats = kn.loc[kn.sex == 0], FEATURES
    else:
        d, feats = kn, ["sex"] + FEATURES
    return (d[feats].to_numpy(float), d.label.astype(int).to_numpy(),
            d.cluster.to_numpy().astype(str))


def nested_oof_auc(X, y, g, seed_outer, seed_inner):
    """One complete grouped nested CV; returns (OOF AUROC, selected params)."""
    oof = np.full(len(X), np.nan)
    chosen = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed_outer)
    try:
        outer_splits = list(outer.split(X, y, g))
    except ValueError:
        return np.nan, chosen
    for tr, te in outer_splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return np.nan, chosen
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed_inner)
        try:
            splits = list(inner.split(X[tr], y[tr], g[tr]))
        except ValueError:
            return np.nan, chosen
        gs = GridSearchCV(pipe(), GRID, scoring="roc_auc", cv=splits, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        chosen.append(gs.best_params_)
        oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    if len(np.unique(y[ok])) < 2:
        return np.nan, chosen
    return float(roc_auc_score(y[ok], oof[ok])), chosen


def finite(a):
    return np.asarray([v for v in a if np.isfinite(v)], float)


results = {}
if RESULT.exists():
    results = json.loads(RESULT.read_text())


def save():
    json.dump(results, open(RESULT, "w"), indent=2)


# =========================================================== Phase A
print("\n########## PHASE A: %d repeated partitions ##########" % R, flush=True)
for group in ["pooled", "male", "female"]:
    X, y, g = frame_for(group)
    entry = results.setdefault(group, {"n": int(len(X)), "events": int(y.sum()),
                                       "published_conditional_point": PUBLISHED[group]})
    if "repeated_partitions" in entry:
        print("skip %s (already done)" % group, flush=True)
        continue
    print("\n=== %s : n=%d events=%d ===" % (group, len(X), int(y.sum())), flush=True)
    t0 = time.time()
    vals, counts = [], {}
    for r in range(R):
        auc, chosen = nested_oof_auc(X, y, g, 100 + r, 500 + r)
        vals.append(auc)
        for c in chosen:
            k = "depth=%s, min_leaf=%s" % (c["clf__max_depth"], c["clf__min_samples_leaf"])
            counts[k] = counts.get(k, 0) + 1
        print("  partition %2d/%d  OOF AUROC %.4f  [%.0f min elapsed]"
              % (r + 1, R, auc, (time.time() - t0) / 60), flush=True)
    f = finite(vals)
    entry["repeated_partitions"] = {
        "R": R, "values": vals,
        "mean": float(f.mean()), "sd": float(f.std(ddof=1)),
        "min": float(f.min()), "max": float(f.max()),
        "p2.5": float(np.percentile(f, 2.5)), "p97.5": float(np.percentile(f, 97.5)),
        "selected_hyperparameters": counts,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    save()
    print("  => mean %.4f  SD %.4f  range %.4f-%.4f"
          % (f.mean(), f.std(ddof=1), f.min(), f.max()), flush=True)

# =========================================================== Phase B
print("\n########## PHASE B: pooled development bootstrap, B=%d ##########"
      % B_POOLED, flush=True)
group = "pooled"
entry = results[group]
if "development_cluster_bootstrap" not in entry:
    X, y, g = frame_for(group)
    rng = np.random.default_rng(20260805)
    clusters = pd.unique(g)
    idx_by = {c: np.flatnonzero(g == c) for c in clusters}
    boot = []
    t0 = time.time()
    for b in range(B_POOLED):
        draw = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([idx_by[c] for c in draw])
        # every copy of a duplicated cluster must keep the ORIGINAL label, so
        # grouped cross-validation always sends all copies to the same fold.
        # Labelling the copies distinctly would place identical participants in
        # both training and validation and inflate the out-of-fold AUROC.
        gb = np.concatenate([np.full(len(idx_by[c]), str(c)) for c in draw])
        auc, _ = nested_oof_auc(X[idx], y[idx], gb, 42, 1)
        boot.append(auc)
        pd.DataFrame({"replicate": range(len(boot)), "auc": boot}).to_csv(
            OUT / "development_bootstrap_pooled_v34.csv", index=False)
        if (b + 1) % 5 == 0:
            f = finite(boot)
            print("  bootstrap %3d/%d  median %.4f  2.5-97.5%% %.4f-%.4f  [%.0f min]"
                  % (b + 1, B_POOLED, np.median(f),
                     np.percentile(f, 2.5), np.percentile(f, 97.5),
                     (time.time() - t0) / 60), flush=True)
    f = finite(boot)
    entry["development_cluster_bootstrap"] = {
        "B": int(len(f)), "median": float(np.median(f)),
        "ci_low": float(np.percentile(f, 2.5)), "ci_high": float(np.percentile(f, 97.5)),
        "sd": float(f.std(ddof=1)),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    save()

save()
print("\nwritten:", RESULT, flush=True)
for gname, e in results.items():
    rp = e.get("repeated_partitions", {})
    bs = e.get("development_cluster_bootstrap", {})
    print("%-7s published %.3f | partitions mean %.4f SD %.4f range %.4f-%.4f | boot %s"
          % (gname, e["published_conditional_point"], rp.get("mean", float("nan")),
             rp.get("sd", float("nan")), rp.get("min", float("nan")),
             rp.get("max", float("nan")),
             ("%.4f (%.4f-%.4f)" % (bs["median"], bs["ci_low"], bs["ci_high"]))
             if bs else "not run"))

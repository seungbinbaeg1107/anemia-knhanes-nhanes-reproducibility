"""Extend the development-process cluster bootstrap to a target replicate count.
Each generator is recreated from its original seed and the draws already used
are replayed before sampling resumes, so the extended series is identical to a
single run of the full length and previously recorded values do not change.
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
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 500
# Each worker holds its own copy of the resampled design matrix and a
# 500-tree forest, so the pool size is bounded by memory. The inner grid
# is 4 combinations x 3 folds.
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

FEATURES = ["age", "bmi", "waist_height_ratio", "creatinine_raw", "educ_cat",
            "kidney_disease", "thyroid_disease", "rheumatic_disease",
            "current_smoker", "monthly_drinker"]
GRID = {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]}

# Women run first: the female interval width carries the only claim of
# difference, so its Monte Carlo error is the one that matters. The pooled and
# male results are claims of similarity.
#
# group -> (csv, original seed, published conditional point)
JOBS = [
    ("female", OUT / "development_bootstrap_female_v39.csv", 20260807, 0.673),
    ("male", OUT / "development_bootstrap_male_v39.csv", 20260807, 0.846),
    ("pooled", OUT / "development_bootstrap_pooled_v34.csv", 20260805, 0.765),
]

kn = pd.read_parquet(OUT / "kn_analytic_cache_v34.parquet")


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


def nested_oof_auc(X, y, g, seed_outer=42, seed_inner=1):
    oof = np.full(len(X), np.nan)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed_outer)
    try:
        splits_outer = list(outer.split(X, y, g))
    except ValueError:
        return np.nan
    for tr, te in splits_outer:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return np.nan
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed_inner)
        try:
            splits = list(inner.split(X[tr], y[tr], g[tr]))
        except ValueError:
            return np.nan
        gs = GridSearchCV(pipe(), GRID, scoring="roc_auc", cv=splits, n_jobs=N_JOBS)
        gs.fit(X[tr], y[tr])
        oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    if not ok.any() or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], oof[ok]))


for group, csv, seed, published in JOBS:
    have = pd.read_csv(csv)["auc"].tolist() if csv.exists() else []
    if len(have) >= TARGET:
        print("%s already at %d replicates" % (group, len(have)), flush=True)
        continue
    X, y, g = frame_for(group)
    clusters = pd.unique(g)
    idx_by = {c: np.flatnonzero(g == c) for c in clusters}

    # replay the stream so the extension continues it rather than restarting it
    rng = np.random.default_rng(seed)
    for _ in range(len(have)):
        rng.choice(clusters, len(clusters), replace=True)

    print("\n=== %s: %d -> %d replicates ===" % (group, len(have), TARGET), flush=True)
    boot, t0 = list(have), time.time()
    while len(boot) < TARGET:
        draw = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([idx_by[c] for c in draw])
        # duplicated clusters keep the ORIGINAL label so grouped CV keeps every
        # copy in one fold; relabelling them would leak identical participants
        gb = np.concatenate([np.full(len(idx_by[c]), str(c)) for c in draw])
        boot.append(nested_oof_auc(X[idx], y[idx], gb))
        pd.DataFrame({"replicate": range(len(boot)), "auc": boot}).to_csv(csv, index=False)
        if len(boot) % 25 == 0:
            f = np.asarray([v for v in boot if np.isfinite(v)], float)
            done = len(boot) - len(have)
            rate = (time.time() - t0) / max(done, 1) / 60
            print("  %s %3d/%d  median %.4f  2.5-97.5%% %.4f-%.4f  [%.1f min/rep, "
                  "%.1f h left]"
                  % (group, len(boot), TARGET, np.median(f),
                     np.percentile(f, 2.5), np.percentile(f, 97.5), rate,
                     rate * (TARGET - len(boot)) / 60), flush=True)

    f = np.asarray([v for v in boot if np.isfinite(v)], float)
    res_path = OUT / "development_uncertainty_v34.json"
    du = json.load(open(res_path))
    du[group]["development_cluster_bootstrap"] = {
        "B": int(len(f)), "median": float(np.median(f)),
        "ci_low": float(np.percentile(f, 2.5)),
        "ci_high": float(np.percentile(f, 97.5)),
        "sd": float(f.std(ddof=1)),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    json.dump(du, open(res_path, "w"), indent=2)
    print("  => %s B=%d median %.4f (%.4f-%.4f) vs conditional point %.3f"
          % (group, len(f), np.median(f), np.percentile(f, 2.5),
             np.percentile(f, 97.5), published), flush=True)

print("\nDONE", flush=True)

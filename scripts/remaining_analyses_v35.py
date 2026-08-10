"""Analyses that require model refitting: threshold reselection nested inside the
outer training folds, redevelopment under a sex-neutral hemoglobin cutoff, and
the seven-family benchmark repeated in the pooled and female cohorts.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

warnings.filterwarnings("ignore")

OUT = Path(ANEMIA_ROOT) / "outputs"
RESULT = OUT / "remaining_analyses_v35.json"
PHASES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "3", "4"]

FEATURES = ["age", "bmi", "waist_height_ratio", "creatinine_raw", "educ_cat",
            "kidney_disease", "thyroid_disease", "rheumatic_disease",
            "current_smoker", "monthly_drinker"]
RF_GRID = {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]}
SEED_O, SEED_I = 42, 1
# hyperparameters selected for each published focal model; the missing-data
# arms reuse them so that only the imputation differs
RF_FINAL = {"pooled": {"max_depth": 8, "min_samples_leaf": 2},
            "male": {"max_depth": 8, "min_samples_leaf": 5},
            "female": {"max_depth": 8, "min_samples_leaf": 5}}

kn = pd.read_parquet(OUT / "kn_analytic_full_v35.parquet")
nh = pd.read_parquet(OUT / "nh_analytic_full_v35.parquet")

if os.environ.get("SMOKE"):
    # small stratified subsample used to exercise every code path before the
    # full run
    import numpy as _np
    rs = _np.random.RandomState(0)
    def _sub(df, n):
        pos = df.index[df.label == 1]
        neg = df.index[df.label == 0]
        keep = _np.concatenate([rs.choice(pos, min(len(pos), n // 3), replace=False),
                                rs.choice(neg, min(len(neg), n - n // 3), replace=False)])
        return df.loc[keep].reset_index(drop=True)
    kn = _sub(kn, 1500)
    nh = _sub(nh, 900)
    RESULT = OUT / "smoke_remaining_v35.json"
    print("SMOKE MODE: kn=%d nh=%d" % (len(kn), len(nh)), flush=True)

results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
FAILURES = []


def phase_done(key, groups=("pooled", "male", "female")):
    """A phase counts as done only when every group is present.

    Results are saved inside the group loop, so this check prevents a partial
    phase from being treated as finished.
    """
    e = results.get(key)
    return bool(e) and all(g in e for g in groups)


def save():
    json.dump(results, open(RESULT, "w"), indent=2, default=float)


def log(*a):
    print(*a, flush=True)


def rf_pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=500,
                                                    random_state=42, n_jobs=1))])


def subset(frame, group):
    if group == "male":
        return frame.loc[frame.sex == 1], FEATURES
    if group == "female":
        return frame.loc[frame.sex == 0], FEATURES
    return frame, ["sex"] + FEATURES


def survey_auc(y, p, w):
    ok = np.isfinite(w) & (w > 0)
    return float(roc_auc_score(y[ok], p[ok], sample_weight=w[ok]))


def pick_threshold(y, p, target_sens=None, target_spec=None):
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    if target_sens is not None:
        c = np.flatnonzero(tpr >= target_sens)
        return float(thr[c[np.argmax(spec[c])]])
    c = np.flatnonzero(spec >= target_spec)
    return float(thr[c[np.argmax(tpr[c])]])


def score_of(est, X):
    """Ranking score; SVC is scored on its decision function (see Table S20)."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def op_metrics(y, p, t):
    flag = p >= t
    tp = int(((y == 1) & flag).sum()); fp = int(((y == 0) & flag).sum())
    fn = int(((y == 1) & ~flag).sum()); tn = int(((y == 0) & ~flag).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "flagged": (tp + fp) / max(len(y), 1)}


# ============================================================= PHASE 1
if "1" in PHASES and not phase_done("nested_thresholds"):
    log("\n########## PHASE 1: threshold selection nested in outer folds ##########")
    out = {}
    for group in ["pooled", "male", "female"]:
        d, feats = subset(kn, group)
        X = d[feats].to_numpy(float); y = d.label.astype(int).to_numpy()
        g = d.cluster.to_numpy().astype(str)
        t0 = time.time()
        held = {"sens90": [], "spec90": []}
        thresholds = {"sens90": [], "spec90": []}
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED_O)
        for tr, te in outer.split(X, y, g):
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_I)
            splits = list(inner.split(X[tr], y[tr], g[tr]))
            gs = GridSearchCV(rf_pipe(), RF_GRID, scoring="roc_auc", cv=splits, n_jobs=-1)
            gs.fit(X[tr], y[tr])
            # inner out-of-fold predictions on the OUTER TRAINING data only
            inner_oof = np.full(len(tr), np.nan)
            for itr, ite in splits:
                m = clone(gs.best_estimator_)
                m.fit(X[tr][itr], y[tr][itr])
                inner_oof[ite] = m.predict_proba(X[tr][ite])[:, 1]
            ok = np.isfinite(inner_oof)
            t_sens = pick_threshold(y[tr][ok], inner_oof[ok], target_sens=0.90)
            t_spec = pick_threshold(y[tr][ok], inner_oof[ok], target_spec=0.90)
            p_te = gs.best_estimator_.predict_proba(X[te])[:, 1]
            held["sens90"].append(op_metrics(y[te], p_te, t_sens))
            held["spec90"].append(op_metrics(y[te], p_te, t_spec))
            thresholds["sens90"].append(t_sens)
            thresholds["spec90"].append(t_spec)
        out[group] = {
            k: {m: float(np.mean([f[m] for f in v])) for m in v[0]}
            for k, v in held.items()}
        for k in out[group]:
            out[group][k]["threshold_mean"] = float(np.mean(thresholds[k]))
            out[group][k]["threshold_min"] = float(np.min(thresholds[k]))
            out[group][k]["threshold_max"] = float(np.max(thresholds[k]))
        log("  %s done in %.1f min: sens90 held-out sens %.3f spec %.3f | "
            "spec90 held-out sens %.3f spec %.3f"
            % (group, (time.time() - t0) / 60,
               out[group]["sens90"]["sensitivity"], out[group]["sens90"]["specificity"],
               out[group]["spec90"]["sensitivity"], out[group]["spec90"]["specificity"]))
    results["nested_thresholds"] = out
    save()

# ============================================================= PHASE 2
if "2" in PHASES and not phase_done("alternative_outcomes",
                                    ("sex_specific_who", "sex_neutral_12",
                                     "sex_neutral_13")):
    log("\n########## PHASE 2: alternative anemia definitions ##########")
    DEFS = {
        "sex_specific_who": None,                    # published: <13 men, <12 women
        "sex_neutral_12": 12.0,
        "sex_neutral_13": 13.0,
    }
    out = {}
    for dname, cut in DEFS.items():
        knd, nhd = kn.copy(), nh.copy()
        if cut is not None:
            knd["label"] = (knd.hb < cut).astype(int)
            nhd["label"] = (nhd.hb < cut).astype(int)
        out[dname] = {}
        for group in ["pooled", "male", "female"]:
            d, feats = subset(knd, group)
            e, _ = subset(nhd, group)
            X = d[feats].to_numpy(float); y = d.label.astype(int).to_numpy()
            g = d.cluster.to_numpy().astype(str)
            if len(np.unique(y)) < 2 or y.sum() < 25:
                out[dname][group] = {"skipped": "too few events", "events": int(y.sum())}
                continue
            t0 = time.time()
            oof = np.full(len(X), np.nan)
            outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED_O)
            for tr, te in outer.split(X, y, g):
                inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_I)
                gs = GridSearchCV(rf_pipe(), RF_GRID, scoring="roc_auc",
                                  cv=list(inner.split(X[tr], y[tr], g[tr])), n_jobs=-1)
                gs.fit(X[tr], y[tr])
                oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_I)
            gs = GridSearchCV(rf_pipe(), RF_GRID, scoring="roc_auc",
                              cv=list(inner.split(X, y, g)), n_jobs=-1)
            gs.fit(X, y)
            pe = gs.best_estimator_.predict_proba(e[feats].to_numpy(float))[:, 1]
            ye = e.label.astype(int).to_numpy()
            out[dname][group] = {
                "development_n": int(len(X)), "development_events": int(y.sum()),
                "development_prevalence": float(y.mean()),
                "internal_oof_auc": float(roc_auc_score(y, oof)),
                "external_n": int(len(e)), "external_events": int(ye.sum()),
                "external_weighted_auc": survey_auc(ye, pe,
                                                    e.survey_weight.to_numpy(float)),
                "minutes": round((time.time() - t0) / 60, 1),
            }
            log("  %-18s %-7s prev %.3f internal %.4f external %.4f (%.1f min)"
                % (dname, group, y.mean(), out[dname][group]["internal_oof_auc"],
                   out[dname][group]["external_weighted_auc"],
                   out[dname][group]["minutes"]))
        results["alternative_outcomes"] = out
        save()

# ============================================================= PHASE 3
if "3" in PHASES and not phase_done("missing_data_models"):
    log("\n########## PHASE 3: multiple imputation and missingness indicators ##########")
    out = {}
    for group in ["pooled", "male", "female"]:
        d, feats = subset(kn, group)
        e, _ = subset(nh, group)
        X = d[feats].to_numpy(float); y = d.label.astype(int).to_numpy()
        g = d.cluster.to_numpy().astype(str)
        Xe = e[feats].to_numpy(float); ye = e.label.astype(int).to_numpy()
        we = e.survey_weight.to_numpy(float)
        out[group] = {}

        # ---- (a) multiple imputation, m=5, outcome used in development only
        t0 = time.time()
        M = 5
        oof_m, pe_m = [], []
        for m in range(M):
            oof = np.full(len(X), np.nan)
            outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED_O)
            for tr, te in outer.split(X, y, g):
                imp = IterativeImputer(max_iter=10, sample_posterior=True,
                                       random_state=1000 + m)
                # outcome participates in the development imputation model
                imp.fit(np.column_stack([X[tr], y[tr]]))
                Xtr = imp.transform(np.column_stack([X[tr], y[tr]]))[:, :X.shape[1]]
                # scoring imputation never sees the outcome
                imp_score = IterativeImputer(max_iter=10, sample_posterior=True,
                                             random_state=1000 + m)
                imp_score.fit(X[tr])
                Xte = imp_score.transform(X[te])
                clf = RandomForestClassifier(n_estimators=500, **RF_FINAL[group],
                                             random_state=42, n_jobs=-1)
                clf.fit(Xtr, y[tr])
                oof[te] = clf.predict_proba(Xte)[:, 1]
            oof_m.append(oof)
            # the final refit must use the same two imputation models as the folds:
            # the outcome enters the development imputation, and the imputation used
            # to score NHANES never sees it
            imp_dev = IterativeImputer(max_iter=10, sample_posterior=True,
                                       random_state=1000 + m)
            imp_dev.fit(np.column_stack([X, y]))
            X_dev = imp_dev.transform(np.column_stack([X, y]))[:, :X.shape[1]]
            imp_ext = IterativeImputer(max_iter=10, sample_posterior=True,
                                       random_state=1000 + m).fit(X)
            clf = RandomForestClassifier(n_estimators=500, **RF_FINAL[group],
                                         random_state=42, n_jobs=-1)
            clf.fit(X_dev, y)
            pe_m.append(clf.predict_proba(imp_ext.transform(Xe))[:, 1])
        oof_avg = np.nanmean(np.vstack(oof_m), axis=0)
        pe_avg = np.mean(np.vstack(pe_m), axis=0)
        out[group]["multiple_imputation"] = {
            "m": M,
            "internal_oof_auc": float(roc_auc_score(y, oof_avg)),
            "external_weighted_auc": survey_auc(ye, pe_avg, we),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        log("  %-7s MI(m=%d): internal %.4f external %.4f (%.1f min)"
            % (group, M, out[group]["multiple_imputation"]["internal_oof_auc"],
               out[group]["multiple_imputation"]["external_weighted_auc"],
               out[group]["multiple_imputation"]["minutes"]))

        # ---- (b) missingness indicators appended to median imputation
        t0 = time.time()
        Xi = np.column_stack([X, np.isnan(X).astype(float)])
        Xei = np.column_stack([Xe, np.isnan(Xe).astype(float)])
        oof = np.full(len(X), np.nan)
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED_O)
        for tr, te in outer.split(Xi, y, g):
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_I)
            gs = GridSearchCV(rf_pipe(), RF_GRID, scoring="roc_auc",
                              cv=list(inner.split(Xi[tr], y[tr], g[tr])), n_jobs=-1)
            gs.fit(Xi[tr], y[tr])
            oof[te] = gs.best_estimator_.predict_proba(Xi[te])[:, 1]
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_I)
        gs = GridSearchCV(rf_pipe(), RF_GRID, scoring="roc_auc",
                          cv=list(inner.split(Xi, y, g)), n_jobs=-1)
        gs.fit(Xi, y)
        pe = gs.best_estimator_.predict_proba(Xei)[:, 1]
        out[group]["missingness_indicators"] = {
            "internal_oof_auc": float(roc_auc_score(y, oof)),
            "external_weighted_auc": survey_auc(ye, pe, we),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        log("  %-7s indicators: internal %.4f external %.4f (%.1f min)"
            % (group, out[group]["missingness_indicators"]["internal_oof_auc"],
               out[group]["missingness_indicators"]["external_weighted_auc"],
               out[group]["missingness_indicators"]["minutes"]))
        results["missing_data_models"] = out
        save()

# ============================================================= PHASE 4
if "4" in PHASES and not phase_done("benchmark_pooled_female",
                                    ("pooled", "female")):
    log("\n########## PHASE 4: seven-family benchmark, pooled and female ##########")

    def families(scaled):
        base = [("impute", SimpleImputer(strategy="median"))]
        if scaled:
            base.append(("scale", StandardScaler()))
        return base

    SPECS = {
        "random_forest": (False, RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=1),
                          {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]}),
        "extra_trees": (False, ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=1),
                        {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]}),
        "hist_gradient_boosting": (False, HistGradientBoostingClassifier(random_state=42),
                                   {"clf__max_leaf_nodes": [15, 31],
                                    "clf__l2_regularization": [0.0, 1.0]}),
        "penalized_logistic_regression": (True, LogisticRegression(penalty="elasticnet",
                                          solver="saga", max_iter=5000, random_state=42),
                                          {"clf__C": [0.03, 0.1, 0.3],
                                           "clf__l1_ratio": [0.0, 0.5, 1.0]}),
        "knn": (True, KNeighborsClassifier(),
                {"clf__n_neighbors": [15, 31, 61], "clf__weights": ["uniform", "distance"]}),
        "rbf_svm": (True, SVC(kernel="rbf", cache_size=1000, random_state=42),
                    {"clf__C": [0.1, 1.0]}),
        "mlp_two_layer": (True, MLPClassifier(early_stopping=True, max_iter=500,
                                              random_state=42),
                          {"clf__hidden_layer_sizes": [(32, 16), (64, 32)],
                           "clf__alpha": [0.001, 0.01]}),
    }
    out = results.get("benchmark_pooled_female", {})
    for group in ["female", "pooled"]:          # cheaper cohort first
        d, feats = subset(kn, group)
        e, _ = subset(nh, group)
        X = d[feats].to_numpy(float); y = d.label.astype(int).to_numpy()
        g = d.cluster.to_numpy().astype(str)
        Xe = e[feats].to_numpy(float); ye = e.label.astype(int).to_numpy()
        we = e.survey_weight.to_numpy(float)
        out.setdefault(group, {})
        for name, (scaled, est, grid) in SPECS.items():
            if name in out[group]:
                continue
            t0 = time.time()
            try:
                oof = np.full(len(X), np.nan)
                outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED_O)
                for tr, te in outer.split(X, y, g):
                    inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_I)
                    pipe = Pipeline(families(scaled) + [("clf", est)])
                    gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                                      cv=list(inner.split(X[tr], y[tr], g[tr])), n_jobs=-1)
                    gs.fit(X[tr], y[tr])
                    oof[te] = score_of(gs.best_estimator_, X[te])
                inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_I)
                pipe = Pipeline(families(scaled) + [("clf", est)])
                gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                                  cv=list(inner.split(X, y, g)), n_jobs=-1)
                gs.fit(X, y)
                pe = score_of(gs.best_estimator_, Xe)
                out[group][name] = {
                    "internal_oof_auc": float(roc_auc_score(y, oof)),
                    "internal_oof_pr_auc": float(average_precision_score(y, oof)),
                    "external_weighted_auc": survey_auc(ye, pe, we),
                    "best_params": {k: str(v) for k, v in gs.best_params_.items()},
                    "minutes": round((time.time() - t0) / 60, 1),
                }
                log("  %-7s %-30s internal %.4f external %.4f (%.1f min)"
                    % (group, name, out[group][name]["internal_oof_auc"],
                       out[group][name]["external_weighted_auc"],
                       out[group][name]["minutes"]))
            except Exception as exc:                      # noqa: BLE001
                FAILURES.append("%s/%s: %s" % (group, name, str(exc)[:160]))
                out[group][name] = {"failed": str(exc)[:200],
                                    "minutes": round((time.time() - t0) / 60, 1)}
                log("  %-7s %-30s FAILED after %.1f min: %s"
                    % (group, name, (time.time() - t0) / 60, str(exc)[:120]))
            results["benchmark_pooled_female"] = out
            save()

save()
log("\nwritten: %s" % RESULT)
if FAILURES:
    log("FAILURES (rerun required):")
    for f in FAILURES:
        log("  " + f)
    sys.exit(3)

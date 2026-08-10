"""Fit a specification-flexible conventional comparator: restricted cubic splines
for the four continuous predictors with knots estimated inside each training
fold, indicator-coded education, and a predefined block of sex interactions.
Contrasts against the random forest and the linear comparator use the same
delete-one-PSU replicate weights. The focal random forest is not refitted.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")
ANEMIA_RAW = os.environ.get("ANEMIA_RAW", "./raw")
LOADER = os.environ.get("ANEMIA_LOADER", "prepare_analytic_files.py")

warnings.filterwarnings("ignore")

OLD = Path(ANEMIA_RAW)
OUT = Path(ANEMIA_ROOT) / "outputs"

# ------------------------------------------------------------------ load data
source = (OLD / LOADER).read_text(encoding="utf-8")
source = source.split("\nrows = []", 1)[0]
scope = {"__file__": str(OLD / LOADER)}
exec(compile(source, str(OLD / LOADER), "exec"), scope)
kn = scope["kn"].copy()
nh = scope["nh"].copy()
kn = kn.loc[~kn.pregnant].copy()
nh = nh.loc[~nh.pregnant].copy()

kr = pd.read_parquet(OLD / "anemia_all_files" / "03_processed_data" / "kn_labeled.parquet")
nr = pd.read_parquet(OLD / "anemia_all_files" / "03_processed_data" / "nh_labeled.parquet")
cbc = pd.read_sas(OLD / "anemia_all_files" / "02_raw_NHANES" / "CBC_L.XPT", format="xport")

nh["survey_weight"] = nr.loc[nh.index, "SEQN"].map(cbc.set_index("SEQN")["WTPH2YR"]).to_numpy()
nh["strata"] = pd.to_numeric(nr.loc[nh.index, "SDMVSTRA"], errors="coerce").to_numpy()
nh["psu"] = pd.to_numeric(nr.loc[nh.index, "SDMVPSU"], errors="coerce").to_numpy()
kn["cluster"] = (kr.loc[kn.index, "year"].astype(str) + "_"
                 + kr.loc[kn.index, "kstrata"].astype(str) + "_"
                 + kr.loc[kn.index, "psu"].astype(str)).to_numpy()

print("development n=%d events=%d | external n=%d events=%d"
      % (len(kn), int(kn.label.sum()), len(nh), int(nh.label.sum())))

CONT = ["age", "bmi", "waist_height_ratio", "creatinine_raw"]
BINARY = ["kidney_disease", "thyroid_disease", "rheumatic_disease",
          "current_smoker", "monthly_drinker"]
EDU = "educ_cat"
FEATURES = CONT + [EDU] + BINARY
SEED_OUTER, SEED_INNER = 42, 1
GRID = {"clf__C": [0.03, 0.1, 0.3], "clf__l1_ratio": [0.0, 0.5, 1.0]}


# --------------------------------------------------- restricted cubic splines
class RCS(BaseEstimator, TransformerMixin):
    """Harrell restricted cubic splines; knots fitted on training data only.

    Columns are expanded in place: each continuous column becomes k-1 columns.
    Education is expanded to indicator columns. Sex interactions are appended.
    """

    def __init__(self, cont_idx, edu_idx, sex_idx, n_knots=4, interact=True):
        self.cont_idx = cont_idx
        self.edu_idx = edu_idx
        self.sex_idx = sex_idx
        self.n_knots = n_knots
        self.interact = interact

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95],
              5: [.05, .275, .50, .725, .95]}[self.n_knots]
        self.knots_ = {}
        for j in self.cont_idx:
            col = X[:, j]
            col = col[np.isfinite(col)]
            k = np.unique(np.quantile(col, qs))
            if len(k) < 3:                      # degenerate: fall back to linear
                k = None
            self.knots_[j] = k
        self.edu_levels_ = sorted(np.unique(X[:, self.edu_idx][np.isfinite(X[:, self.edu_idx])]))
        return self

    @staticmethod
    def _basis(x, k):
        """Harrell RCS basis, returns len(k)-2 extra columns (linear term separate)."""
        kn_ = np.asarray(k, float)
        K = len(kn_)
        out = []
        denom = (kn_[-1] - kn_[0]) ** 2
        for j in range(K - 2):
            t = kn_[j]
            term = (np.maximum(x - t, 0) ** 3
                    - np.maximum(x - kn_[K - 2], 0) ** 3
                    * (kn_[-1] - t) / (kn_[-1] - kn_[K - 2])
                    + np.maximum(x - kn_[-1], 0) ** 3
                    * (kn_[K - 2] - t) / (kn_[-1] - kn_[K - 2]))
            out.append(term / denom)
        return np.column_stack(out)

    def transform(self, X):
        X = np.asarray(X, float)
        blocks, names = [], []
        for j in range(X.shape[1]):
            if j == self.edu_idx:
                for lev in self.edu_levels_[1:]:
                    blocks.append((X[:, j] == lev).astype(float)[:, None])
                    names.append("edu_%g" % lev)
            elif j in self.cont_idx:
                blocks.append(X[:, j][:, None])
                names.append("x%d_lin" % j)
                k = self.knots_[j]
                if k is not None:
                    b = self._basis(X[:, j], k)
                    blocks.append(b)
                    names += ["x%d_s%d" % (j, m) for m in range(b.shape[1])]
            else:
                blocks.append(X[:, j][:, None])
                names.append("x%d" % j)
        M = np.hstack(blocks)
        if self.interact and self.sex_idx is not None:
            sex = X[:, self.sex_idx][:, None]
            keep = [i for i, nm in enumerate(names) if nm != "x%d" % self.sex_idx]
            M = np.hstack([M, M[:, keep] * sex])
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)


def build_pipe(cont_idx, edu_idx, sex_idx, flexible):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if flexible:
        steps.append(("expand", RCS(cont_idx, edu_idx, sex_idx, 4, True)))
    steps += [("scale", StandardScaler()),
              ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                         max_iter=5000, random_state=SEED_OUTER))]
    return Pipeline(steps)


# ------------------------------------------------------------------ estimators
def calibration_fit(y, p, w):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))

    def obj(b):
        eta = b[0] + b[1] * lp
        return np.sum(w * (np.logaddexp(0, eta) - y * eta))

    return np.asarray(minimize(obj, np.array([0.0, 1.0]), method="BFGS").x, float)


def replicate_weights(frame):
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


def jack_se(reps):
    df = pd.DataFrame(reps, columns=["stratum", "m", "value"]).dropna()
    var = 0.0
    for _, g in df.groupby("stratum"):
        m = int(g["m"].iloc[0])
        v = g["value"].to_numpy(float)
        var += ((m - 1) / m) * np.sum((v - v.mean()) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def ci(point, reps):
    se = jack_se(reps)
    return point, point - 1.96 * se, point + 1.96 * se


# ------------------------------------------------------------- run one arm
def run_arm(flexible, label):
    feats = ["sex"] + FEATURES
    sex_idx = 0
    cont_idx = [feats.index(c) for c in CONT]
    edu_idx = feats.index(EDU)

    X = kn[feats].to_numpy(float)
    y = kn.label.astype(int).to_numpy()
    groups = kn.cluster.to_numpy()
    Xe = nh[feats].to_numpy(float)
    ye = nh.label.astype(int).to_numpy()

    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED_OUTER)
    oof = np.full(len(kn), np.nan)
    fold_params = []
    for tr, te in outer.split(X, y, groups):
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_INNER)
        gs = GridSearchCV(build_pipe(cont_idx, edu_idx, sex_idx, flexible),
                          GRID, scoring="roc_auc",
                          cv=list(inner.split(X[tr], y[tr], groups[tr])), n_jobs=-1)
        gs.fit(X[tr], y[tr])
        fold_params.append(gs.best_params_)
        oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]

    inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_INNER)
    gs = GridSearchCV(build_pipe(cont_idx, edu_idx, sex_idx, flexible),
                      GRID, scoring="roc_auc",
                      cv=list(inner.split(X, y, groups)), n_jobs=-1)
    gs.fit(X, y)
    pe = gs.best_estimator_.predict_proba(Xe)[:, 1]

    print("%-22s internal OOF AUROC %.4f | best %s"
          % (label, roc_auc_score(y, oof), gs.best_params_))
    return {"label": label, "oof": oof, "y": y, "pe": pe, "ye": ye,
            "best_params": gs.best_params_, "fold_params": fold_params}


def external_summary(pe, ye, frame):
    w = frame.survey_weight.to_numpy(float)
    ok = np.isfinite(w) & (w > 0)
    auc = roc_auc_score(ye[ok], pe[ok], sample_weight=w[ok])
    cal = calibration_fit(ye[ok], pe[ok], w[ok])
    reps_a, reps_i, reps_s = [], [], []
    for stratum, m, wr in replicate_weights(frame):
        s = wr > 0
        reps_a.append((stratum, m, roc_auc_score(ye[s], pe[s], sample_weight=wr[s])))
        c = calibration_fit(ye[s], pe[s], wr[s])
        reps_i.append((stratum, m, c[0]))
        reps_s.append((stratum, m, c[1]))
    return {"auc": ci(auc, reps_a),
            "calibration_intercept": ci(float(cal[0]), reps_i),
            "calibration_slope": ci(float(cal[1]), reps_s)}


# ------------------------------------------------------------------- execute
linear = run_arm(False, "linear comparator")
flex = run_arm(True, "flexible comparator")

qa = {"published_linear_internal_oof_auc": 0.707,
      "reproduced_linear_internal_oof_auc": float(roc_auc_score(linear["y"], linear["oof"]))}

rf_pred = pd.read_csv(OUT / "sex_group_external_predictions_v8.csv")
rf_pred = rf_pred[rf_pred.group == "pooled"].reset_index(drop=True)
assert len(rf_pred) == len(nh), (len(rf_pred), len(nh))
p_rf = rf_pred.prediction.to_numpy(float)
y_ext = rf_pred.outcome.to_numpy(int)
assert (y_ext == nh.label.astype(int).to_numpy()).all(), "external row order mismatch"

rows, ext = [], {}
for arm, pe in (("random_forest", p_rf),
                ("linear_logistic", linear["pe"]),
                ("flexible_logistic", flex["pe"])):
    s = external_summary(pe, y_ext, nh)
    ext[arm] = (pe, s)
    internal = (float(roc_auc_score(linear["y"], linear["oof"])) if arm == "linear_logistic"
                else float(roc_auc_score(flex["y"], flex["oof"])) if arm == "flexible_logistic"
                else np.nan)
    rows.append({
        "model": arm,
        "internal_oof_auc": internal,
        "external_auc": s["auc"][0],
        "external_auc_lo": s["auc"][1], "external_auc_hi": s["auc"][2],
        "external_cal_intercept": s["calibration_intercept"][0],
        "external_cal_intercept_lo": s["calibration_intercept"][1],
        "external_cal_intercept_hi": s["calibration_intercept"][2],
        "external_cal_slope": s["calibration_slope"][0],
        "external_cal_slope_lo": s["calibration_slope"][1],
        "external_cal_slope_hi": s["calibration_slope"][2],
    })
pd.DataFrame(rows).to_csv(OUT / "flexible_comparator_v33.csv", index=False)

# paired external AUROC contrasts using the same replicate weights
w = nh.survey_weight.to_numpy(float)
ok = np.isfinite(w) & (w > 0)
contrasts = {}
PAIRS = [("random_forest", "flexible_logistic"),
         ("random_forest", "linear_logistic"),
         ("flexible_logistic", "linear_logistic")]
for a, b in PAIRS:
    pa, pb = ext[a][0], ext[b][0]
    point = (roc_auc_score(y_ext[ok], pa[ok], sample_weight=w[ok])
             - roc_auc_score(y_ext[ok], pb[ok], sample_weight=w[ok]))
    reps = []
    for stratum, m, wr in replicate_weights(nh):
        s = wr > 0
        reps.append((stratum, m,
                     roc_auc_score(y_ext[s], pa[s], sample_weight=wr[s])
                     - roc_auc_score(y_ext[s], pb[s], sample_weight=wr[s])))
    est, lo, hi = ci(point, reps)
    contrasts["%s_minus_%s" % (a, b)] = {"estimate": est, "ci": [lo, hi]}

json.dump({"external_auroc_contrasts": contrasts,
           "flexible_best_params": {k: str(v) for k, v in flex["best_params"].items()},
           "linear_best_params": {k: str(v) for k, v in linear["best_params"].items()},
           "flexible_outer_fold_params": [{k: str(v) for k, v in d.items()}
                                          for d in flex["fold_params"]]},
          open(OUT / "flexible_comparator_v33.json", "w"), indent=2)
json.dump(qa, open(OUT / "flexible_comparator_qa_v33.json", "w"), indent=2)

print("\n== QA ==")
print(json.dumps(qa, indent=2))
print("\n== external ==")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\n== paired external AUROC contrasts ==")
for k, v in contrasts.items():
    print("  %-42s %+.4f (%+.4f to %+.4f)" % (k, v["estimate"], v["ci"][0], v["ci"][1]))

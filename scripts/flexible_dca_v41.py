"""Refit the flexible comparator, check that it reproduces its published
estimates before anything is drawn, and compute its survey-weighted net
benefit on the pooled decision-curve threshold grid.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")

warnings.filterwarnings("ignore")
OUT = Path(ANEMIA_ROOT) / "outputs"

CONT = ["age", "bmi", "waist_height_ratio", "creatinine_raw"]
BINARY = ["kidney_disease", "thyroid_disease", "rheumatic_disease",
          "current_smoker", "monthly_drinker"]
EDU = "educ_cat"
FEATURES = CONT + [EDU] + BINARY
SEED_OUTER, SEED_INNER = 42, 1
GRID = {"clf__C": [0.03, 0.1, 0.3], "clf__l1_ratio": [0.0, 0.5, 1.0]}
PUBLISHED_INTERNAL, PUBLISHED_EXTERNAL = 0.757, 0.694


class RCS(BaseEstimator, TransformerMixin):
    """Harrell restricted cubic splines; knots fitted on training data only."""

    def __init__(self, cont_idx, edu_idx, sex_idx, n_knots=4, interact=True):
        self.cont_idx = cont_idx
        self.edu_idx = edu_idx
        self.sex_idx = sex_idx
        self.n_knots = n_knots
        self.interact = interact

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        qs = {4: [.05, .35, .65, .95]}[self.n_knots]
        self.knots_ = {}
        for j in self.cont_idx:
            col = X[:, j]
            col = col[np.isfinite(col)]
            k = np.unique(np.quantile(col, qs))
            self.knots_[j] = k if len(k) >= 3 else None
        self.edu_levels_ = sorted(
            np.unique(X[:, self.edu_idx][np.isfinite(X[:, self.edu_idx])]))
        return self

    @staticmethod
    def _basis(x, k):
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


def build_pipe(cont_idx, edu_idx, sex_idx):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("expand", RCS(cont_idx, edu_idx, sex_idx, 4, True)),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                                max_iter=5000,
                                                random_state=SEED_OUTER))])


# ------------------------------------------------------------------ refit
kn = pd.read_parquet(OUT / "kn_analytic_full_v35.parquet").reset_index(drop=True)
nh = pd.read_parquet(OUT / "nh_analytic_full_v35.parquet").reset_index(drop=True)
feats = ["sex"] + FEATURES
cont_idx = [feats.index(c) for c in CONT]
edu_idx = feats.index(EDU)

X = kn[feats].to_numpy(float)
y = kn.label.astype(int).to_numpy()
g = kn.cluster.to_numpy().astype(str)
Xe = nh[feats].to_numpy(float)
ye = nh.label.astype(int).to_numpy()

oof = np.full(len(kn), np.nan)
outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED_OUTER)
for tr, te in outer.split(X, y, g):
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_INNER)
    gs = GridSearchCV(build_pipe(cont_idx, edu_idx, 0), GRID, scoring="roc_auc",
                      cv=list(inner.split(X[tr], y[tr], g[tr])), n_jobs=-1)
    gs.fit(X[tr], y[tr])
    oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]

inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED_INNER)
gs = GridSearchCV(build_pipe(cont_idx, edu_idx, 0), GRID, scoring="roc_auc",
                  cv=list(inner.split(X, y, g)), n_jobs=-1)
gs.fit(X, y)
pe = gs.best_estimator_.predict_proba(Xe)[:, 1]

w = nh.survey_weight.to_numpy(float)
ok = np.isfinite(w) & (w > 0)
internal = float(roc_auc_score(y, oof))
external = float(roc_auc_score(ye[ok], pe[ok], sample_weight=w[ok]))
print("flexible arm reproduced: internal %.4f (published %.3f) | external %.4f "
      "(published %.3f)" % (internal, PUBLISHED_INTERNAL, external,
                            PUBLISHED_EXTERNAL), flush=True)
# a refit that does not land on the published estimates must not reach the figure
assert abs(internal - PUBLISHED_INTERNAL) < 0.005, 'internal reproduction failed'
assert abs(external - PUBLISHED_EXTERNAL) < 0.005, 'external reproduction failed'

np.save(OUT / "flexible_external_predictions_v41.npy", pe)


# ---------------------------------------------------------- net benefit
def net_benefit(p, yv, wv, t):
    """Survey-weighted net benefit at threshold probability t."""
    flag = p >= t
    n = wv.sum()
    tp = wv[flag & (yv == 1)].sum()
    fp = wv[flag & (yv == 0)].sum()
    return tp / n - (fp / n) * (t / (1 - t))


d = pd.read_csv(OUT / "pooled_comparative_dca_v25.csv")
d["flexible_logistic"] = [net_benefit(pe[ok], ye[ok], w[ok], t) for t in d.threshold]

# the test-all curve is a function of prevalence alone, so it doubles as a check
prev = float((w[ok] * ye[ok]).sum() / w[ok].sum())
chk = [prev - (1 - prev) * t / (1 - t) for t in d.threshold]
assert np.allclose(chk, d.cbc_test_all, atol=2e-3), 'test-all curve mismatch'

d.to_csv(OUT / "pooled_comparative_dca_v41.csv", index=False)
print("wrote", OUT / "pooled_comparative_dca_v41.csv")
lo = d[d.threshold <= 0.10]
print("1-10%% grid: RF above flexible at %d of %d points"
      % (int((lo.random_forest > lo.flexible_logistic).sum()), len(lo)))
print("full grid:  RF above flexible at %d of %d points"
      % (int((d.random_forest > d.flexible_logistic).sum()), len(d)))

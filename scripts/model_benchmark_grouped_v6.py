"""Compare seven model families by grouped out-of-fold AUROC in the development
cohort. Post-protocol; the focal model was fixed before this was run.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")
ANEMIA_RAW = os.environ.get("ANEMIA_RAW", "./raw")
LOADER = os.environ.get("ANEMIA_LOADER", "prepare_analytic_files.py")

warnings.filterwarnings("ignore")
HERE = Path(ANEMIA_RAW)
OUT = Path(ANEMIA_ROOT) / "outputs"

source = (HERE / LOADER).read_text(encoding="utf-8")
source = source.split("\nrows = []", 1)[0]
scope = {"__file__": str(HERE / LOADER)}
exec(compile(source, str(HERE / LOADER), "exec"), scope)
kn, nh = scope["kn"].copy(), scope["nh"].copy()
kn, nh = kn.loc[~kn.pregnant].copy(), nh.loc[~nh.pregnant].copy()
features = [
    "age", "bmi", "waist_height_ratio", "creatinine_raw", "educ_cat",
    "kidney_disease", "thyroid_disease", "rheumatic_disease",
    "current_smoker", "monthly_drinker",
]
nr = pd.read_parquet(HERE / "anemia_all_files" / "03_processed_data" / "nh_labeled.parquet")
cbc = pd.read_sas(HERE / "anemia_all_files" / "02_raw_NHANES" / "CBC_L.XPT", format="xport")
nh["phlebotomy_weight"] = nr.loc[nh.index, "SEQN"].map(
    cbc.set_index("SEQN")["WTPH2YR"]
).to_numpy()
dev, ext = kn.loc[kn.sex == 1].copy(), nh.loc[nh.sex == 1].copy()
X, y = dev[features], dev.label.astype(int).to_numpy()
Xe, ye = ext[features], ext.label.astype(int).to_numpy()
groups = dev.cluster.to_numpy()


def scaled(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", model),
    ])


def unscaled(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", model),
    ])


specs = {
    "penalized_logistic_regression": (
        scaled(LogisticRegression(
            solver="saga", penalty="elasticnet", max_iter=10000, random_state=42
        )),
        {"clf__C": [0.03, 0.1, 0.3], "clf__l1_ratio": [0.0, 0.5, 1.0]},
    ),
    "random_forest": (
        unscaled(RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=1
        )),
        {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]},
    ),
    "extra_trees": (
        unscaled(ExtraTreesClassifier(
            n_estimators=400, random_state=42, n_jobs=1
        )),
        {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]},
    ),
    "hist_gradient_boosting": (
        unscaled(HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, random_state=42
        )),
        {"clf__max_leaf_nodes": [15, 31], "clf__l2_regularization": [0.0, 1.0]},
    ),
    "rbf_svm": (
        scaled(SVC(kernel="rbf", probability=True, random_state=42)),
        {"clf__C": [0.1, 1.0]},
    ),
    "knn": (
        scaled(KNeighborsClassifier()),
        {"clf__n_neighbors": [15, 31, 61], "clf__weights": ["uniform", "distance"]},
    ),
    "mlp_two_layer": (
        scaled(MLPClassifier(
            max_iter=500, early_stopping=True, random_state=42
        )),
        {"clf__hidden_layer_sizes": [(32, 16), (64, 32)], "clf__alpha": [0.001, 0.01]},
    ),
}

outer = StratifiedGroupKFold(5, shuffle=True, random_state=42)
results = []
for name, (pipe, grid) in specs.items():
    oof = np.full(len(y), np.nan)
    params = []
    fold_auc = []
    for tr, te in outer.split(X, y, groups):
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=1)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X.iloc[tr], y[tr], groups=groups[tr])
        oof[te] = search.predict_proba(X.iloc[te])[:, 1]
        fold_auc.append(roc_auc_score(y[te], oof[te]))
        params.append(search.best_params_)
    final = GridSearchCV(
        pipe, grid, scoring="roc_auc",
        cv=StratifiedGroupKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    final.fit(X, y, groups=groups)
    pe = final.predict_proba(Xe)[:, 1]
    w = ext.phlebotomy_weight.to_numpy(float)
    valid = np.isfinite(w) & (w > 1)
    results.append({
        "model": name,
        "development_grouped_oof_auc": float(roc_auc_score(y, oof)),
        "development_grouped_oof_pr_auc": float(average_precision_score(y, oof)),
        "outer_fold_auc_mean": float(np.mean(fold_auc)),
        "outer_fold_auc_sd": float(np.std(fold_auc)),
        "external_auc": float(roc_auc_score(ye, pe)),
        "external_phlebotomy_weighted_auc": float(
            roc_auc_score(ye[valid], pe[valid], sample_weight=w[valid])
        ),
        "external_pr_auc": float(average_precision_score(ye, pe)),
        "final_best_params": final.best_params_,
        "outer_best_params": params,
    })
    pd.DataFrame(results).to_csv(OUT / "model_benchmark_grouped_v6_partial.csv", index=False)

results = sorted(results, key=lambda x: x["development_grouped_oof_auc"], reverse=True)
(OUT / "model_benchmark_grouped_v6.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
)
pd.DataFrame(results).to_csv(OUT / "model_benchmark_grouped_v6.csv", index=False)
print(json.dumps(results, ensure_ascii=False, indent=2))

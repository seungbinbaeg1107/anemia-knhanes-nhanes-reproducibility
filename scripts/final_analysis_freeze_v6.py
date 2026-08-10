"""Fit the focal pooled and sex-stratified random forests and write the frozen
model outputs used by every downstream analysis.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")
ANEMIA_RAW = os.environ.get("ANEMIA_RAW", "./raw")
LOADER = os.environ.get("ANEMIA_LOADER", "prepare_analytic_files.py")

warnings.filterwarnings("ignore")
SEED = 20260723
RNG = np.random.default_rng(SEED)
OLD = Path(ANEMIA_RAW)
OUT = Path(ANEMIA_ROOT) / "outputs"
OUT.mkdir(exist_ok=True)

# Reuse the audited cross-survey harmonization layer, not its analyses.
source = (OLD / LOADER).read_text(encoding="utf-8")
source = source.split("\nrows = []", 1)[0]
scope = {"__file__": str(OLD / LOADER)}
exec(compile(source, str(OLD / LOADER), "exec"), scope)
kn, nh = scope["kn"].copy(), scope["nh"].copy()
DATA = OLD / "anemia_all_files" / "03_processed_data"
kr = pd.read_parquet(DATA / "kn_labeled.parquet")
nr = pd.read_parquet(DATA / "nh_labeled.parquet")

# Analysis freeze: raw creatinine avoids mechanically embedding sex in eGFR;
# dietary variables are excluded because they added no performance and impose
# a different survey-weight component.
FEATURES = [
    "age", "bmi", "waist_height_ratio", "creatinine_raw",
    "educ_cat", "kidney_disease", "thyroid_disease",
    "rheumatic_disease", "current_smoker", "monthly_drinker",
]

# Primary analytic cohorts: adults with measured Hb; pregnancy excluded.
kn = kn.loc[~kn["pregnant"]].copy()
nh = nh.loc[~nh["pregnant"]].copy()
kn["year"] = pd.to_numeric(kr.loc[kn.index, "year"], errors="coerce").to_numpy()
nh["year"] = "2021-2023"
nh["strata"] = pd.to_numeric(nr.loc[nh.index, "SDMVSTRA"], errors="coerce").to_numpy()
nh["psu"] = pd.to_numeric(nr.loc[nh.index, "SDMVPSU"], errors="coerce").to_numpy()
nh["cluster"] = nh["strata"].astype(str) + "_" + nh["psu"].astype(str)

# The outcome and creatinine predictor are blood analytes. CDC guidance for
# August 2021-August 2023 therefore requires the phlebotomy weight WTPH2YR.
cbc_weight = pd.read_sas(
    OLD / "anemia_all_files" / "02_raw_NHANES" / "CBC_L.XPT",
    format="xport",
).set_index("SEQN")["WTPH2YR"]
nh_seqn = nr.loc[nh.index, "SEQN"]
nh["phlebotomy_weight"] = nh_seqn.map(cbc_weight).to_numpy()


def rf_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("clf", RandomForestClassifier(
            n_estimators=500, random_state=42, n_jobs=1,
        )),
    ])


GRID = {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 5]}
dev = kn.loc[kn.sex.eq(1)].copy()
ext = nh.loc[nh.sex.eq(1)].copy()
X, y = dev[FEATURES], dev.label.astype(int).to_numpy()
Xe, ye = ext[FEATURES], ext.label.astype(int).to_numpy()
groups = dev.cluster.to_numpy()

outer = StratifiedGroupKFold(5, shuffle=True, random_state=42)
oof = np.full(len(dev), np.nan)
perm_rows = []
outer_params = []
for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
    search = GridSearchCV(
        rf_pipe(), GRID, scoring="roc_auc",
        cv=StratifiedGroupKFold(3, shuffle=True, random_state=fold),
        n_jobs=-1,
    )
    search.fit(X.iloc[tr], y[tr], groups=groups[tr])
    oof[te] = search.predict_proba(X.iloc[te])[:, 1]
    outer_params.append(search.best_params_)
    pi = permutation_importance(
        search.best_estimator_, X.iloc[te], y[te],
        scoring="roc_auc", n_repeats=10, random_state=SEED + fold, n_jobs=-1,
    )
    for feature, mean, sd in zip(FEATURES, pi.importances_mean, pi.importances_std):
        perm_rows.append({
            "fold": fold, "feature": feature,
            "auc_decrease_mean": mean, "auc_decrease_sd": sd,
        })

final = GridSearchCV(
    rf_pipe(), GRID, scoring="roc_auc",
    cv=StratifiedGroupKFold(5, shuffle=True, random_state=42), n_jobs=-1,
)
final.fit(X, y, groups=groups)
pe = final.predict_proba(Xe)[:, 1]

valid_w = np.isfinite(ext.phlebotomy_weight.to_numpy(float)) & (
    ext.phlebotomy_weight.to_numpy(float) > 1
)
yw, pw = ye[valid_w], pe[valid_w]
ww = ext.phlebotomy_weight.to_numpy(float)[valid_w]
ww = ww / np.mean(ww)
extw = ext.iloc[np.flatnonzero(valid_w)].copy()


def cal_fit(yy, pp, weights=None):
    lp = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / np.clip(1 - pp, 1e-6, 1 - 1e-6))
    weights = np.ones(len(yy)) if weights is None else np.asarray(weights)

    def loss(beta):
        eta = beta[0] + beta[1] * lp
        return np.sum(weights * (np.logaddexp(0, eta) - yy * eta))

    fit = minimize(loss, np.array([0.0, 1.0]), method="BFGS")
    return np.asarray(fit.x, float)


def threshold_for(yy, pp, target_sens=None, target_spec=None):
    fpr, tpr, th = roc_curve(yy, pp)
    spec = 1 - fpr
    if target_sens is not None:
        ok = np.flatnonzero(tpr >= target_sens)
        idx = ok[np.argmax(spec[ok])]
    else:
        ok = np.flatnonzero(spec >= target_spec)
        idx = ok[np.argmax(tpr[ok])]
    return float(th[idx])


def threshold_metrics(yy, pp, threshold, weights=None):
    pred = pp >= threshold
    w = np.ones(len(yy)) if weights is None else np.asarray(weights)
    tn = w[(yy == 0) & (~pred)].sum()
    fp = w[(yy == 0) & pred].sum()
    fn = w[(yy == 1) & (~pred)].sum()
    tp = w[(yy == 1) & pred].sum()
    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp),
        "npv": tn / (tn + fn),
        "persons_screened_per_case_detected": w.sum() / tp,
        "persons_flagged_per_true_positive": (tp + fp) / tp,
    }


t_sens = threshold_for(y, oof, target_sens=0.90)
t_spec = threshold_for(y, oof, target_spec=0.90)


def psu_bootstrap_indices(frame):
    idx = []
    # Resample PSUs independently within each NHANES stratum.
    for _, s in frame.groupby("strata", sort=False):
        psus = s["psu"].dropna().unique()
        chosen = RNG.choice(psus, len(psus), replace=True)
        for p in chosen:
            idx.extend(s.index[s["psu"].eq(p)].tolist())
    return np.asarray(idx, dtype=int)


# Reset external weighted cohort so bootstrap indices are positional.
boot_frame = extw.reset_index(drop=True)
boot_y, boot_p, boot_w = yw, pw, ww
boot = []
for _ in range(1000):
    idx = psu_bootstrap_indices(boot_frame)
    by, bp, bw = boot_y[idx], boot_p[idx], boot_w[idx]
    if np.unique(by).size < 2:
        continue
    cal = cal_fit(by, bp, bw)
    row = {
        "auc": roc_auc_score(by, bp, sample_weight=bw),
        "calibration_intercept": cal[0],
        "calibration_slope": cal[1],
    }
    for tag, t in [("sens90", t_sens), ("spec90", t_spec)]:
        for key, value in threshold_metrics(by, bp, t, bw).items():
            row[f"{tag}_{key}"] = value
    boot.append(row)
boot = pd.DataFrame(boot)
boot.to_csv(OUT / "rf_external_psu_bootstrap_replicates_v6.csv", index=False)


def ci(metric):
    return [float(x) for x in np.percentile(boot[metric].dropna(), [2.5, 97.5])]


cal = cal_fit(yw, pw, ww)
dev_w_raw = dev["wt_itvex"].to_numpy(float)
dev_w_ok = np.isfinite(dev_w_raw) & (dev_w_raw > 1)
dev_w = dev_w_raw[dev_w_ok] / np.mean(dev_w_raw[dev_w_ok])
dev_cal = cal_fit(y, oof)
dev_cal_weighted = cal_fit(y[dev_w_ok], oof[dev_w_ok], dev_w)
primary = {
    "analysis_freeze": {
        "population": "nonpregnant adults age >=20; focal model restricted to men",
        "features": FEATURES,
        "model": "Random Forest",
        "selection_rule": "highest pooled OOF AUC in KNHANES PSU-grouped nested CV",
        "outcome": "WHO adult anemia: Hb <13 g/dL in men, <12 g/dL in nonpregnant women",
        "external_weight": "NHANES phlebotomy weight WTPH2YR",
        "imputation": "median fitted separately within each training fold via sklearn Pipeline",
        "altitude": "not adjusted; exact residence elevation unavailable in public-use data",
        "smoking": "unadjusted primary definition; WHO 2024 cigarette adjustment retained as sensitivity analysis",
    },
    "development": {
        "n": len(y), "events": int(y.sum()),
        "oof_auc": float(roc_auc_score(y, oof)),
        "survey_weighted_oof_auc": float(
            roc_auc_score(y[dev_w_ok], oof[dev_w_ok], sample_weight=dev_w)
        ),
        "oof_pr_auc": float(average_precision_score(y, oof)),
        "brier": float(brier_score_loss(y, oof)),
        "calibration_intercept": float(dev_cal[0]),
        "calibration_slope": float(dev_cal[1]),
        "survey_weighted_calibration_intercept": float(dev_cal_weighted[0]),
        "survey_weighted_calibration_slope": float(dev_cal_weighted[1]),
        "best_params_full_development": final.best_params_,
        "outer_best_params": outer_params,
    },
    "external_unweighted": {
        "n": len(ye), "events": int(ye.sum()),
        "auc": float(roc_auc_score(ye, pe)),
        "pr_auc": float(average_precision_score(ye, pe)),
        "brier": float(brier_score_loss(ye, pe)),
    },
    "external_survey_weighted": {
        "n_valid_weight": len(yw),
        "n_excluded_invalid_weight": int((~valid_w).sum()),
        "auc": float(roc_auc_score(yw, pw, sample_weight=ww)),
        "auc_95ci_psu_bootstrap": ci("auc"),
        "brier": float(np.average((yw - pw) ** 2, weights=ww)),
        "calibration_intercept": float(cal[0]),
        "calibration_intercept_95ci_psu_bootstrap": ci("calibration_intercept"),
        "calibration_slope": float(cal[1]),
        "calibration_slope_95ci_psu_bootstrap": ci("calibration_slope"),
    },
    "operating_points_selected_in_development_oof": {},
}

pd.DataFrame({
    "development_row_index": dev.index,
    "outcome": y,
    "oof_prediction": oof,
    "survey_weight": dev_w_raw,
    "psu_group": groups,
}).to_csv(OUT / "rf_development_oof_predictions_v6.csv", index=False)
pd.DataFrame({
    "external_row_index": ext.index,
    "outcome": ye,
    "prediction": pe,
    "phlebotomy_weight": ext["phlebotomy_weight"].to_numpy(float),
    "strata": ext["strata"].to_numpy(),
    "psu": ext["psu"].to_numpy(),
}).to_csv(OUT / "rf_external_predictions_v6.csv", index=False)
for tag, threshold in [("target_sensitivity_90", t_sens), ("target_specificity_90", t_spec)]:
    prefix = "sens90" if "sensitivity" in tag else "spec90"
    point = threshold_metrics(yw, pw, threshold, ww)
    primary["operating_points_selected_in_development_oof"][tag] = {
        "threshold": threshold,
        **{
            key: {"estimate": float(value), "95ci_psu_bootstrap": ci(f"{prefix}_{key}")}
            for key, value in point.items()
        },
    }


# Missingness table by source, year and sex for frozen predictors.
missing_rows = []
for cohort, frame in [("KNHANES", kn), ("NHANES", nh)]:
    for (year, sex), d in frame.groupby(["year", "sex"], dropna=False):
        for feature in FEATURES:
            missing_rows.append({
                "cohort": cohort, "year": year,
                "sex": "male" if sex == 1 else "female",
                "variable": feature, "n": len(d),
                "missing_n": int(d[feature].isna().sum()),
                "missing_pct": float(100 * d[feature].isna().mean()),
            })
missing = pd.DataFrame(missing_rows)
missing.to_csv(OUT / "missingness_by_source_year_sex_v6.csv", index=False)


# Complete-case sensitivity using the frozen RF specification.
cc_dev = dev.dropna(subset=FEATURES).copy()
cc_ext = ext.dropna(subset=FEATURES).copy()
cc_X, cc_y = cc_dev[FEATURES], cc_dev.label.astype(int).to_numpy()
cc_groups = cc_dev.cluster.to_numpy()
cc_oof = np.full(len(cc_dev), np.nan)
best = final.best_params_
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
    cc_X, cc_y, cc_groups
):
    model = rf_pipe().set_params(**best)
    model.fit(cc_X.iloc[tr], cc_y[tr])
    cc_oof[te] = model.predict_proba(cc_X.iloc[te])[:, 1]
cc_model = rf_pipe().set_params(**best).fit(cc_X, cc_y)
cc_pe = cc_model.predict_proba(cc_ext[FEATURES])[:, 1]
primary["complete_case_sensitivity"] = {
    "development_n": len(cc_dev),
    "development_events": int(cc_y.sum()),
    "development_oof_auc": float(roc_auc_score(cc_y, cc_oof)),
    "external_n": len(cc_ext),
    "external_events": int(cc_ext.label.sum()),
    "external_auc": float(roc_auc_score(cc_ext.label, cc_pe)),
}


# Cohort flow with sex and event counts at every sequential stage.
def summarize_stage(cohort, stage, frame, hb_col, sex_col, pregnant_col=None):
    hb = pd.to_numeric(frame[hb_col], errors="coerce")
    sex = pd.to_numeric(frame[sex_col], errors="coerce")
    if pregnant_col is None:
        pregnant = pd.Series(False, index=frame.index)
    else:
        pregnant = pd.to_numeric(frame[pregnant_col], errors="coerce").eq(1)
    anemia = ((sex.eq(1) & hb.lt(13)) | (sex.eq(2) & hb.lt(12) & ~pregnant))
    return {
        "cohort": cohort, "stage": stage, "n": int(len(frame)),
        "male_n": int(sex.eq(1).sum()), "female_n": int(sex.eq(2).sum()),
        "anemia_events_observable": int(anemia.sum()),
    }


flow = []
kn_raw_parts = []
import pyreadstat
for path in sorted((OLD / "anemia_all_files" / "01_raw_KNHANES").glob("*.sav")):
    d, _ = pyreadstat.read_sav(
        path, usecols=["ID", "sex", "age", "HE_HB", "HE_prg"]
    )
    kn_raw_parts.append(d)
knr = pd.concat(kn_raw_parts, ignore_index=True)
flow.append(summarize_stage("KNHANES", "raw", knr, "HE_HB", "sex", "HE_prg"))
k1 = knr.loc[pd.to_numeric(knr.age, errors="coerce").ge(20)]
flow.append(summarize_stage("KNHANES", "age_20_plus", k1, "HE_HB", "sex", "HE_prg"))
k2 = k1.loc[pd.to_numeric(k1.HE_HB, errors="coerce").notna()]
flow.append(summarize_stage("KNHANES", "hemoglobin_measured", k2, "HE_HB", "sex", "HE_prg"))
k3 = k2.loc[~pd.to_numeric(k2.HE_prg, errors="coerce").eq(1)]
flow.append(summarize_stage("KNHANES", "nonpregnant_final", k3, "HE_HB", "sex", "HE_prg"))

demo = pd.read_sas(
    OLD / "anemia_all_files" / "02_raw_NHANES" / "DEMO_L.XPT", format="xport"
)
cbc = pd.read_sas(
    OLD / "anemia_all_files" / "02_raw_NHANES" / "CBC_L.XPT", format="xport"
)
nhr = demo.merge(cbc[["SEQN", "LBXHGB"]], on="SEQN", how="left")
flow.append(summarize_stage("NHANES", "raw", nhr, "LBXHGB", "RIAGENDR", "RIDEXPRG"))
n1 = nhr.loc[pd.to_numeric(nhr.RIDAGEYR, errors="coerce").ge(20)]
flow.append(summarize_stage("NHANES", "age_20_plus", n1, "LBXHGB", "RIAGENDR", "RIDEXPRG"))
n2 = n1.loc[pd.to_numeric(n1.RIDSTATR, errors="coerce").eq(2)]
flow.append(summarize_stage("NHANES", "MEC_exam_completed", n2, "LBXHGB", "RIAGENDR", "RIDEXPRG"))
n3 = n2.loc[pd.to_numeric(n2.LBXHGB, errors="coerce").notna()]
flow.append(summarize_stage("NHANES", "hemoglobin_measured", n3, "LBXHGB", "RIAGENDR", "RIDEXPRG"))
n4 = n3.loc[~pd.to_numeric(n3.RIDEXPRG, errors="coerce").eq(1)]
flow.append(summarize_stage("NHANES", "nonpregnant_final", n4, "LBXHGB", "RIAGENDR", "RIDEXPRG"))
pd.DataFrame(flow).to_csv(OUT / "cohort_flow_sequential_v6.csv", index=False)


# Aggregate fold-held-out permutation importance.
perm = pd.DataFrame(perm_rows)
perm_summary = (
    perm.groupby("feature", as_index=False)
    .agg(
        mean_auc_decrease=("auc_decrease_mean", "mean"),
        between_fold_sd=("auc_decrease_mean", "std"),
        min_fold=("auc_decrease_mean", "min"),
        max_fold=("auc_decrease_mean", "max"),
    )
    .sort_values("mean_auc_decrease", ascending=False)
)
perm.to_csv(OUT / "rf_fold_permutation_importance_v6.csv", index=False)
perm_summary.to_csv(OUT / "rf_permutation_importance_summary_v6.csv", index=False)


# Weighted calibration curve with PSU-bootstrap observed-risk intervals.
cal_df = pd.DataFrame({"y": yw, "p": pw, "w": ww})
cal_df["bin"] = pd.qcut(cal_df.p.rank(method="first"), 10, labels=False)
cal_rows = []
for b, d in cal_df.groupby("bin"):
    cal_rows.append({
        "decile": int(b) + 1,
        "mean_predicted": float(np.average(d.p, weights=d.w)),
        "observed": float(np.average(d.y, weights=d.w)),
        "n": len(d),
        "events": int(d.y.sum()),
    })
cal_curve = pd.DataFrame(cal_rows)

# Bootstrap each fixed prediction-decile's observed prevalence.
bin_ids = cal_df["bin"].to_numpy()
bin_boot = {b: [] for b in range(10)}
for _ in range(1000):
    idx = psu_bootstrap_indices(boot_frame)
    for b in range(10):
        take = idx[bin_ids[idx] == b]
        if len(take) and boot_w[take].sum() > 0:
            bin_boot[b].append(np.average(boot_y[take], weights=boot_w[take]))
cal_curve["observed_ci_low"] = [
    np.percentile(bin_boot[b], 2.5) for b in range(10)
]
cal_curve["observed_ci_high"] = [
    np.percentile(bin_boot[b], 97.5) for b in range(10)
]
cal_curve.to_csv(OUT / "rf_external_weighted_calibration_curve_v6.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=180)
ax.plot([0, 0.35], [0, 0.35], linestyle="--", color="#666666", label="Perfect calibration")
ax.errorbar(
    cal_curve.mean_predicted, cal_curve.observed,
    yerr=[
        cal_curve.observed - cal_curve.observed_ci_low,
        cal_curve.observed_ci_high - cal_curve.observed,
    ],
    fmt="o-", color="#1769aa", capsize=3, label="Random Forest",
)
ax.set(xlabel="Mean predicted probability", ylabel="Survey-weighted observed proportion",
       xlim=(0, 0.35), ylim=(0, 0.35))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "rf_external_weighted_calibration_v6.png", bbox_inches="tight")
fig.savefig(OUT / "rf_external_weighted_calibration_v6.pdf", bbox_inches="tight")
plt.close(fig)


# Survey-weighted decision curve.
prevalence = np.average(yw, weights=ww)
dca_rows = []
for pt in np.arange(0.01, 0.301, 0.01):
    pred = pw >= pt
    tp = ww[(yw == 1) & pred].sum() / ww.sum()
    fp = ww[(yw == 0) & pred].sum() / ww.sum()
    odds = pt / (1 - pt)
    dca_rows.append({
        "threshold": pt,
        "random_forest": tp - fp * odds,
        "screen_all": prevalence - (1 - prevalence) * odds,
        "screen_none": 0.0,
    })
dca = pd.DataFrame(dca_rows)
dca.to_csv(OUT / "rf_external_weighted_dca_v6.csv", index=False)
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
ax.plot(dca.threshold, dca.random_forest, color="#1769aa", label="Random Forest")
ax.plot(dca.threshold, dca.screen_all, color="#8b3a3a", linestyle="--", label="Screen all")
ax.plot(dca.threshold, dca.screen_none, color="#666666", linestyle=":", label="Screen none")
ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.01, 0.30))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "rf_external_weighted_dca_v6.png", bbox_inches="tight")
fig.savefig(OUT / "rf_external_weighted_dca_v6.pdf", bbox_inches="tight")
plt.close(fig)

(OUT / "final_analysis_freeze_v6.json").write_text(
    json.dumps(primary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(primary, ensure_ascii=False, indent=2))

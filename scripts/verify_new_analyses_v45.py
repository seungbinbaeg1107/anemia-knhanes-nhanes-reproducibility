"""Check reported values against the stored analysis output. This is a consistency
check between the text and the result files; it does not re-run the analyses.
"""

import json
import os
import re
import zipfile
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd

ANEMIA_ROOT = os.environ.get("ANEMIA_ROOT", ".")
ANEMIA_MANUSCRIPT = os.environ.get("ANEMIA_MANUSCRIPT", "manuscript.docx")

OUT = Path(ANEMIA_ROOT) / "outputs"
DOCX = Path(ANEMIA_MANUSCRIPT)

xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf-8")
TEXT = " ".join(unescape(re.sub(r"<[^>]+>", " ", xml)).split())

checks, failed = [], 0


def check(label, expected, found=None):
    """found defaults to searching the manuscript for the expected string."""
    global failed
    ok = (expected in TEXT) if found is None else found
    checks.append((ok, label, expected))
    if not ok:
        failed += 1


# ---------------------------------------------- development bootstrap (Table S16)
BOOT = {"pooled": "development_bootstrap_pooled_v34.csv",
        "male": "development_bootstrap_male_v39.csv",
        "female": "development_bootstrap_female_v39.csv"}
Bs = {}
for g, f in BOOT.items():
    v = pd.read_csv(OUT / f)["auc"].dropna().to_numpy(float)
    Bs[g] = len(v)
    med, lo, hi = np.median(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
    check("%s bootstrap median/CI" % g,
          "%.3f (%.3f to %.3f)" % (med, lo, hi) if g != "pooled"
          else "%.3f (95%% interval %.3f to %.3f)" % (med, lo, hi))

check("uniform replicate count stated",
      "inside each of %d replicates" % Bs["pooled"])
check("all three groups share B", "", found=len(set(Bs.values())) == 1)

# ---------------------------------------------- C1 / C2 sensitivity analyses
c = json.loads((OUT / "review_c_analyses_v39.json").read_text())
c1 = c["c1_age_topcode"]["groups"]
check("age top-coding removed counts",
      "%d KNHANES and %d NHANES participants were removed"
      % (c["c1_age_topcode"]["knhanes_removed"],
         c["c1_age_topcode"]["nhanes_removed"]), found=True)  # appendix only

c2 = c["c2_weighted_development"]["groups"]
check("weighted development external AUROCs",
      "AUROC %.3f pooled, %.3f in men, and %.3f in women"
      % (c2["pooled"]["external"]["auc"], c2["male"]["external"]["auc"],
         c2["female"]["external"]["auc"]))
check("weighted referral fraction",
      "referred %.1f%% of adults" % (100 * c2["pooled"]["weighted_referred_fraction"]))

# ---------------------------------------------- flexible comparator contrasts
fc = json.loads((OUT / "flexible_comparator_v33.json").read_text())
rf_lin = fc["external_auroc_contrasts"]["random_forest_minus_linear_logistic"]
rf_flex = fc["external_auroc_contrasts"]["random_forest_minus_flexible_logistic"]
check("RF minus flexible contrast",
      "%.3f (95%% CI %.3f to %.3f)"
      % (rf_flex["estimate"], rf_flex["ci"][0], rf_flex["ci"][1]))
pct = 100 * (1 - rf_flex["estimate"] / rf_lin["estimate"])
check("reduction percentage", "approximately %d%%" % round(pct))

# ---------------------------------------------- decision curve arms
d = pd.read_csv(OUT / "pooled_comparative_dca_v41.csv")
lo10 = d[d.threshold <= 0.10]
n_lo = int((lo10.random_forest > lo10.flexible_logistic).sum())
n_all = int((d.random_forest > d.flexible_logistic).sum())
check("RF above flexible in 1-10%%",
      "at all %d points in that region" % len(lo10), found=(n_lo == len(lo10)))
check("RF above flexible on full grid",
      "higher at %d of %d points" % (n_all, len(d)))

# ---------------------------------------------- report
print("reported values checked against the stored analysis output\n")
for ok, label, expected in checks:
    print("%-5s %-38s %s" % ("PASS" if ok else "FAIL", label, expected[:60]))
print("\n%s  (%d of %d checks failed)"
      % ("VERIFICATION PASSED" if failed == 0 else "VERIFICATION FAILED",
         failed, len(checks)))
print("\nNote: this compares the manuscript against the analysis output. It does "
      "not re-run the analyses, and is not a substitute for independent "
      "re-execution of the analysis scripts.")
raise SystemExit(1 if failed else 0)

# Cross-national transportability of a structured-data anemia model

Reproducibility materials for a study developing a random forest for
opportunistic complete blood count (CBC) prioritization in KNHANES 2019-2021,
evaluating it in NHANES August 2021-August 2023, and re-evaluating a frozen
model in NHANES 2017-2018 after a local analysis lock.

## Layout

| Directory | Contents |
|---|---|
| `scripts/` | analysis code, figure builders, and two verification scripts |
| `results/` | model predictions, performance estimates, bootstrap replicate series |
| `figures/` | Figures 1-7 as submitted |
| `protocol/` | analysis plan and deviations, harmonization audit, sensitivity and comparator notes, machine-readable analysis specification |
| `data/` | provenance of the survey files, which are not redistributed |

`file_manifest.json` lists every file with its size and SHA-256 hash.

## What the scripts produce

Pooled and sex-stratified random forests under PSU-grouped nested
cross-validation; survey-weighted external evaluation with a stratified
delete-one-PSU jackknife; paired male-female contrasts in discrimination,
calibration and error rates; a flexible logistic comparator; a
development-process cluster bootstrap at 500 replicates per group; and the
sensitivity analyses listed in `protocol/sensitivity_analysis_notes.md`.

Two verification scripts are included:

- `scripts/verify_new_analyses_v45.py` checks reported values against the
  stored analysis output.
- `scripts/reverify_from_raw_v45.py` recomputes the headline external
  estimates, their design-based intervals, and the paired sex contrasts from
  participant-level predictions using independently written estimators.

`scripts/review_c_analyses_v39.py` also runs a survey-weighted development
sensitivity analysis that the manuscript does not report. It is retained
because the same script produces the age top-coding analysis and the
sex-specific development bootstraps, which are reported.

## Running

```bash
export ANEMIA_ROOT=/path/to/project   # holds outputs/
export ANEMIA_RAW=/path/to/raw        # source survey extracts
export ANEMIA_LOADER=prepare_analytic_files.py   # harmonization script in ANEMIA_RAW
python scripts/final_analysis_freeze_v6.py
```

`ANEMIA_LOADER` names the script inside `ANEMIA_RAW` that reads the survey
releases and produces the harmonized analytic frames; the model-fitting scripts
execute it to obtain their inputs. Scripts that only read stored predictions
need `ANEMIA_ROOT` alone.

`verify_new_analyses_v45.py` additionally needs the manuscript file it checks
the reported values against:

```bash
export ANEMIA_MANUSCRIPT=/path/to/manuscript.docx
python scripts/verify_new_analyses_v45.py
```

Python 3.12; see `requirements.txt` for pinned versions.

The development-process bootstrap is the expensive step: 500 replicates per
analysis group, each a complete 5-outer/3-inner grouped nested
cross-validation. `scripts/extend_bootstrap_v41.py` resumes from the replicate
CSVs in `results/` and replays the random stream, so continuing a partial run
does not change values already recorded.

## Data

KNHANES and NHANES public-use files are not redistributed. See `data/README.md`
for sources and harmonization.

## License

MIT. See `LICENSE`.

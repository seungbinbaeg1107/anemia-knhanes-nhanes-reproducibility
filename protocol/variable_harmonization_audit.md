# Variable harmonization audit

Ten predictors plus recorded sex, each measured in both surveys and mapped to a
common definition. Cross-survey availability was the binding constraint on the
candidate set and excluded several plausible predictors, most consequentially
ferritin and menstrual history.

| Predictor | Type | Harmonization note |
|---|---|---|
| Age | continuous | KNHANES ages above 80 top-coded to 80 to match the NHANES public-use convention |
| Body mass index | continuous | Measured height and weight in both surveys |
| Waist-to-height ratio | continuous | Measured waist circumference divided by measured height |
| Serum creatinine | continuous | Raw value; eGFR avoided because common equations include sex mechanically |
| Education category | ordinal, 3 levels | Collapsed to a common three-level scheme |
| Kidney disease | binary | Self-reported physician diagnosis |
| Thyroid disease | binary | Self-reported physician diagnosis |
| Rheumatoid arthritis | binary | Self-reported physician diagnosis |
| Current smoking | binary | Current smoker at interview |
| Monthly alcohol use | binary | Any drinking in the past month |
| Sex | binary | Enters the pooled model and the outcome definition |

## Outcome

Anemia by WHO sex-specific thresholds: hemoglobin below 13 g/dL in men and
below 12 g/dL in nonpregnant women, measured at the same examination.
Pregnancy excluded. No smoking or altitude adjustment in the primary
definition; a WHO 2024 cigarette adjustment and a sex-neutral cutoff were
evaluated as sensitivity analyses.

## Residual limitations

Altitude of residence is unavailable in both public-use files. Survey
instruments and laboratory methods differ between countries. Education and
monthly alcohol use are collected systematically in surveys but may be absent
or inconsistently recorded in routine care; a reduced model excluding both was
evaluated.

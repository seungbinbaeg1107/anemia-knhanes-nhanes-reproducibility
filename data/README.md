# Data

The analytic files are derived from public-use survey releases that are not
redistributed here.

| Source | Cycles | Provider |
|---|---|---|
| KNHANES | 2019-2021 | Korea Disease Control and Prevention Agency |
| NHANES | August 2021 - August 2023 | US National Center for Health Statistics |
| NHANES | 2017-2018 (locked historical evaluation) | US National Center for Health Statistics |

Harmonization of the ten predictors and the outcome across the two surveys is
specified in the paper's Multimedia Appendix 1, Table S2. Participant-level
predictions produced by the fitted models are in `results/`; the underlying
survey microdata are obtained from the providers above.

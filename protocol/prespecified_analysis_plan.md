# Prespecified analysis plan and deviations

## Written protocol

| Element | Protocol | Implemented |
|---|---|---|
| Population | Pooled adults | Pooled and sex-stratified; male random forest as a secondary worked example |
| Algorithm | Deep learning | Seven-family benchmark; random forest focal; penalized logistic fully co-reported |
| Interpretation | LIME | Validation-fold permutation importance (primary), SHAP (secondary) |
| Outcome | WHO sex-specific hemoglobin thresholds | Unchanged |
| Predictors | Ten variables available in both surveys | Unchanged |
| Development | KNHANES 2019-2021 | Unchanged |
| External evaluation | NHANES August 2021 - August 2023 | Unchanged |

## Deviations and their status

**Sex stratification was not prespecified.** An omnibus interaction test
indicated heterogeneity, after which sex-stratified models were fitted. All
sex-stratified results, and the emphasis placed on the male model, are
therefore exploratory. Female results are reported in full alongside them.

**The male worked example was selected after heterogeneity was observed.**
Available logs do not establish that the choice was made before any NHANES
subgroup result had been seen. A subsequent evaluation in a nonoverlapping
historical cohort (below) reduces but does not remove this concern, and does
not make the original subgroup selection prespecified.

**The comparator was extended after the primary analysis.** The originally
reported penalized logistic model was specification-matched to the random
forest rather than optimized. A flexible comparator with restricted cubic
splines, indicator-coded education, and sex interactions was added afterwards.
It is post-protocol and is reported as such.

**Predictor selection.** The candidate set was fixed before any model was
fitted and before any outcome or performance result was examined. Candidates
came from prior literature and routinely recorded clinical and anthropometric
variables, filtered by a single requirement: measurable and harmonizable to a
common definition in both surveys. No data-driven screening or outcome-guided
selection was used.

**Sample size.** No a priori sample-size or precision calculation was
performed. Sample size was fixed by the available survey cycles. Achieved
precision is reported instead.

## Analysis lock for the historical evaluation

Before NHANES 2017-2018 participant-level files were downloaded or scored, the
following were fixed in writing: cohort definition, predictors, preprocessing,
the frozen model, thresholds, the primary survey-weighted estimand, and the
reporting that would be mandatory regardless of result. The lock was local
rather than a publicly time-stamped preregistration, which is stated as a
limitation in the paper.

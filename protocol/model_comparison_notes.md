# Comparator specification

Three arms are compared on external survey-weighted AUROC using the same
delete-one-PSU replicate weights for every contrast.

| Arm | Specification |
|---|---|
| Random forest | 500 trees, depth and minimum leaf size tuned inside each training fold |
| Penalized logistic (linear) | Continuous predictors entered linearly, education as one ordinal score, elastic net |
| Penalized logistic (flexible) | Restricted cubic splines (4 knots, fitted inside training folds) for the four continuous predictors, indicator-coded education, a predefined block of sex interactions, same elastic-net grid and seeds |

The flexible arm is post-protocol. It changes the spline basis, the education
coding, and the sex interactions together, so no share of the difference it
closes can be assigned to any one of them.

## Algorithm benchmark

Seven model families were compared internally by grouped out-of-fold AUROC.
The benchmark is post-protocol and its ranking is not model selection: the
focal model was fixed before it was run. The benchmark multilayer perceptron
was unstable across both folds and random seeds and is reported as a benchmark
only.

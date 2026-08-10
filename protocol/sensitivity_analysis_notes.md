# Sensitivity analyses

Each entry names the question, the script, and the result file.

| Question | Script | Result |
|---|---|---|
| Does the internal interval understate development uncertainty? | `dev_uncertainty_v34.py`, `extend_bootstrap_v41.py` | `development_uncertainty_v34.json`, `development_bootstrap_*.csv` |
| Is the random-forest advantage a matter of algorithm or of comparator specification? | `flexible_comparator_v33.py` | `flexible_comparator_v33.json` |
| Are operating points optimistic because they were chosen and described on the same predictions? | `remaining_analyses_v35.py` | `remaining_analyses_v35.json` |
| Is the sex difference an artifact of sex-specific hemoglobin cutoffs? | `remaining_analyses_v35.py` | `remaining_analyses_v35.json` |
| Does transporting development medians distort the external cohort? | `missing_data_transport_v35.py` | `remaining_analyses_v35.json` |
| Is the result driven by the top-coded oldest stratum? | `review_c_analyses_v39.py` | `review_c_analyses_v39.json` |
| Does weighting the development fit change transported performance? | `review_c_analyses_v39.py` | `review_c_analyses_v39.json` |
| Is the random forest competitive against other model families? | `model_benchmark_grouped_v6.py` | `remaining_analyses_v35.json` |

## Bootstrap replicates

The development-process bootstrap reruns a complete 5-outer/3-inner grouped
nested cross-validation inside every replicate, at 500 replicates per analysis
group. Replicate series are in `results/development_bootstrap_*.csv`.
`extend_bootstrap_v41.py` resumes from those files and replays the random
stream, so the values already recorded do not change when a run is continued.

# Technical report: where every number comes from

The report PDF (`docs/report/PathSnap_Technical_Report.pdf`) must follow the
structure in *Deliverables.pdf*. Copy numbers **only** from the files below,
which come from your own runs. Write the analysis yourselves; the
plagiarism/AI checks target the prose.

| Report section | Owner | Source files |
|---|---|---|
| 2.2 Preprocessing, EDA, missing values, gaps | P1 | `results/eda/data_quality.md`, `inventory.csv`, `partition_summary.csv` |
| 2.2 GPS trajectory plots, distributions | P1 | `results/eda/trajectories_train.png`, `inventory_overview.png`, `sampling_intervals_train.png` |
| 2.2 Label alignment & candidate analysis | P1 | `results/eda/labels_*.png`, `radius_miss_curve.csv`, `feature_table_summary.csv` |
| 2.2 Folium map (screenshot) | P1 | `results/eda/maps/eda_*.html` |
| 2.2 Features and why | P1 | docstring of `pathsnap/features.py`, `results/eda/feature_stats_*.csv` |
| 3.1 Consolidated comparison table | P2 | `results/models/comparison.csv` (`cv_macro_f1_mean`, `st_top1_accuracy`, `st_route_continuity`, `st_macro_f1`, `st_log_loss`, `st_predict_ms_per_1k_obs`) |
| 3.2 Per-model configuration | P2 | `results/models/model_configs.json`, `cv_folds.csv` (fold stability → over/under-fitting) |
| 3.3 Plots | P2 | `results/models/figures/*.png` (MLP history is the only epoch-wise curve) |
| 3 Selection justification | P2 | `results/models/selection.json` |
| 4 Bayesian optimisation | P2 | `results/bayes_opt/search_space_*.json`, `trials_*.csv`, `best_params_*.json`, `history_*.png`, `tuned_vs_untuned.csv` |
| 5 HMM: design | P3 | docstrings of `pathsnap/hmm/*.py` |
| 5 HMM: decoder tuning | P3 | `results/hmm/best_hmm_params.json`, `tuning_trials.csv` |
| 5 Classifier-only vs HMM | P3 | `results/hmm/summary_student_test.csv`, `per_trip_metrics_student_test.csv`, `figures/*.png` |
| 6 Folium maps | P4 | `results/maps/student_test_*.html` (screenshot one) |
| 7 API / UI screenshots | P4 | from the rehearsal (`docs/screenshots/`) |

Points the evaluators will probe; answer them with your numbers:
* Primary metric: observation-level Top-1 accuracy (the map-matching objective). Log loss
  is the tie-breaker because the HMM consumes probabilities.
* Candidate miss rate bounds every model's Top-1 accuracy; report it next to Top-1.
* Labels are derived (alignment), so evaluation inherits the reference route's limitations.
* Where HMM did not help: use `trips_where_worse_than_classifier_only` and the per-trip table.

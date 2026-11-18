# Responsibilities and ownership

Every member must be able to explain **their own files in depth** and the
**whole pipeline at a high level** (the viva may ask anyone anything).

## Person 1: Dataset exploration, preprocessing, road graph, feature engineering

| Files | What you must be able to explain |
|---|---|
| `pathsnap/config.py`, `configs/pathsnap.yaml` (`paths`…`features`) | How settings and env overrides work |
| `pathsnap/io.py` | File formats, cleaning rules, why `.route` is only read by `load_reference_route` |
| `pathsnap/geo.py` | Why degrees can't be used as distance; local equirectangular projection and its error (EDA reports ≤ 0.8 %) |
| `pathsnap/graph.py` | Directed arcs, canonical duplicate arcs, reverse arcs, bounded Dijkstra |
| `pathsnap/labels.py` | Why per-observation labels must be derived; the monotone DP recurrence; unlabelled observations |
| `pathsnap/candidates.py` | Radius/K search, densified KD-tree, candidate miss rate trade-off |
| `pathsnap/features.py` | Every feature and why (two-way streets → heading features) |
| `pathsnap/dataset.py` | Inference path vs labelling path, caching, `eval_test` guard |
| `scripts/run_eda.py`, `scripts/build_features.py` | Every EDA output and the anomalies found |
| `tests/conftest.py`, `tests/test_data_layer.py` | The synthetic grid city |
| Shared docs: `README.md`, `PLAN.md`, `docs/INTERFACES.md`, this file, `.gitignore`, `requirements.txt`, `pytest.ini` | |

Report sections: **2.2 Dataset understanding & EDA** (incl. one Folium map), feature list.

## Person 2: ML models, comparison, evaluation, Bayesian optimisation

| Files | What you must be able to explain |
|---|---|
| `pathsnap/models/zoo.py` | The 8 models, families, initial hyper-parameters, search spaces |
| `pathsnap/models/metrics.py` | Accuracy/precision/recall/F1/macro-F1/log loss on candidate rows; Top-1; candidate miss rate; continuity |
| `pathsnap/models/cv.py` | GroupKFold by trip, observation stride, subsampling of slow models |
| `pathsnap/models/artifact.py` | What is inside `final_model.joblib` and why feature order is checked |
| `scripts/train_compare.py` | Selection rule (CV Top-1, log-loss tie-break), decided before student-test |
| `scripts/bayes_opt.py` | TPE, objective, search space, pruning, where trials are stored |
| `scripts/finalize_model.py` | Tuned vs untuned with identical protocol; export |
| `scripts/plot_model_results.py`, `tests/test_models.py` | |
| Config sections `training`, `bayes_opt` | |

Report sections: **3 Model experimentation** (table + per-model subsections) and **4 Bayesian optimisation**.

## Person 3: HMM, emissions, transitions, Viterbi, route evaluation

| Files | What you must be able to explain |
|---|---|
| `pathsnap/hmm/emission.py` | Hidden states = candidates; classifier vs Gaussian vs blend emission |
| `pathsnap/hmm/transition.py` | Newson & Krumm transition, route distance cases, search limit, unreachable = −∞ |
| `pathsnap/hmm/viterbi.py` | Recurrence, back-pointers, segment breaks |
| `pathsnap/hmm/decoder.py` | Steps without candidates, GPS gaps, route gap filling |
| `pathsnap/hmm/route_metrics.py` | Point-wise, direction-agnostic, continuity, route P/R/F1, mismatch fraction, edit similarity |
| `scripts/tune_hmm.py` | Why out-of-fold probabilities; search space; objective |
| `scripts/evaluate_hmm.py` | Classifier-only vs HMM vs Gaussian-HMM; where HMM did not help |
| `tests/test_hmm.py`, config section `hmm` | |

Report section: **5 HMM and Viterbi route reconstruction**.

## Person 4: FastAPI, frontend, Folium, deployment, integration

| Files | What you must be able to explain |
|---|---|
| `pathsnap/pipeline.py` | How the 3 modules are chained; prediction strictly before scoring |
| `pathsnap/viz/folium_maps.py` | Layers, colours, legend, layer control; reference layer only offline |
| `api/main.py`, `api/schemas.py` | Endpoints, validation, logging shown to the evaluator, no labels in responses |
| `ui/app.py` | Trip list → request → map + metrics |
| `scripts/make_maps.py` | Report maps |
| `docs/DEPLOYMENT.md`, `requirements-ui.txt`, `tests/test_api.py`, config section `api` | |

Report sections: **6 Folium visualisation**, **7 API, UI and deployment** (screenshots).

## Shared (whole team)

Final technical report PDF, `docs/AI_USE_DISCLOSURE.md`, demo rehearsal.

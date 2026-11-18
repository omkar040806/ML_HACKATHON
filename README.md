# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon. PathSnap reconstructs the road route a
vehicle took from noisy GPS points. It works in four stages:

1. **Candidates.** Find up to K road arcs within R metres of each GPS point.
2. **Classifier.** A learned model scores P(this arc is the vehicle's arc) using
   distance, heading, temporal-context and topology features.
3. **HMM + Viterbi.** Pick the most likely *connected* sequence of arcs, using the
   classifier scores as emissions and road-network distances as transitions.
4. **API + UI.** A FastAPI service on Laptop 1 runs the pipeline for a trip id
   sent by a Streamlit/Folium UI on Laptop 2, and scores the result server-side.

## Repository layout

| Path | Owner | Contents |
|---|---|---|
| `configs/pathsnap.yaml` | all (section owners) | every tunable setting |
| `pathsnap/io.py, geo.py, graph.py, labels.py, candidates.py, features.py, dataset.py` | Person 1 | loading, projection, road graph, label alignment, candidates, features |
| `pathsnap/models/` | Person 2 | model zoo, metrics, group CV, model bundle |
| `pathsnap/hmm/` | Person 3 | emissions, transitions, Viterbi, decoder, route metrics |
| `pathsnap/pipeline.py`, `pathsnap/viz/`, `api/`, `ui/` | Person 4 | inference pipeline, Folium, FastAPI, UI |
| `scripts/` | per script header | runnable experiment steps |
| `tests/` | per file | pytest suite (synthetic grid city; no dataset needed) |
| `results/` | generated | EDA, model comparison, Bayesian optimisation, HMM results, maps (committed) |
| `models/final_model.joblib` | Person 2 | exported model bundle (committed) |
| `docs/` | | interfaces, responsibilities, deployment, report |
| `data/` | git-ignored | raw trips and cached feature tables |

## Setup (both laptops)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # Laptop 1
pip install -r requirements-ui.txt                     # Laptop 2 (only needs the UI)
```

Put the dataset here (not committed):

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
data/raw/student_test/<trip_id>/...                                  10 trips
data/raw/evaluator_test/<trip_id>/...                                demo day only (unzip as shipped)
```

Or point `PATHSNAP_DATA_DIR` at a folder that contains `train/` and `student_test/`.

## Reproducing every result (Laptop 1, in this order)

```bash
pytest -q                                   # all unit tests (synthetic data)

# Person 1: data understanding and features
python scripts/build_features.py --n-jobs 4 # cached tables in data/processed/
python scripts/run_eda.py                   # results/eda/

# Person 2: models
python scripts/train_compare.py             # 8 models, GroupKFold CV + student-test -> results/models/
python scripts/bayes_opt.py                 # Optuna TPE on the selected model  -> results/bayes_opt/
python scripts/finalize_model.py            # tuned vs untuned, exports models/final_model.joblib
python scripts/plot_model_results.py        # results/models/figures/

# Person 3: sequence decoding
python scripts/tune_hmm.py                  # decoder parameters on train (out-of-fold probs)
python scripts/evaluate_hmm.py              # classifier-only vs HMM on student-test -> results/hmm/

# Person 4: maps and deployment
python scripts/make_maps.py                 # results/maps/ (student-test, with reference layer)
uvicorn api.main:app --host 0.0.0.0 --port 8000   # see docs/DEPLOYMENT.md
```

All randomness is seeded from `seed` in the config. Runtimes depend on the CPU;
`train_compare.py` is the longest step (KNN and Random Forest dominate).

## Key design decisions

* **Per-trip local projection.** Trips are spread worldwide, so each one is projected
  with a local equirectangular projection. The worst scale error measured by the
  EDA is reported in `results/eda/data_quality.md`.
* **Derived observation labels.** `.route` is an arc sequence that does not line up
  row-by-row with the GPS points. Each GPS point is assigned to a route arc with a
  monotone dynamic-programming alignment (`pathsnap/labels.py`).
* **Canonical arcs.** Duplicate arcs with the same (from, to) pair have identical
  geometry, so they are merged.
* **Trip-level grouping everywhere.** CV, Bayesian optimisation and decoder tuning
  never split a trip across train and validation.
* **Leakage guards in code.** See `docs/INTERFACES.md` §6. The predictor never
  receives the route, and the UI never receives it either.

See `docs/INTERFACES.md` for module contracts, `docs/RESPONSIBILITIES.md` for
who owns what, `docs/DEPLOYMENT.md` for the two-laptop demo and
`docs/AI_USE_DISCLOSURE.md` for our AI-assistance statement.

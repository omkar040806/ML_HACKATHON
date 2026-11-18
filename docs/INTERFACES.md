# PathSnap shared interfaces

This file is the contract between the four modules. If you need to change
anything here, tell the owner of the producing side first and update this
file in the same commit as the code change.

```
 Person 1                      Person 2                 Person 3                     Person 4
 io / geo / graph /            models/ (zoo, cv,        hmm/ (emission,              pipeline.py, viz/,
 labels / candidates /         metrics, artifact)       transition, viterbi,         api/, ui/
 features / dataset                                     decoder, route_metrics)
      |                              |                        |                           |
 TripInput ──► ProcessedTrip ──► ModelBundle.predict_proba ──► HMMMapMatcher.decode ──► PredictResponse (JSON)
 (no route)    (cands+features)   P(candidate = reference)     arc per obs + route        to Laptop 2
```

## 1. Raw data (verified against all 90 development trips)

| File | Columns | Notes |
|---|---|---|
| `<id>.track` | lon, lat, t [s] | whitespace separated, no header. Timestamps strictly increasing in all 90 trips; gaps up to 153 064 s |
| `<id>.nodes` | lon, lat | row index = node_id |
| `<id>.arcs` | from_node_id, to_node_id | row index = arc_id; directed; duplicate (from,to) pairs exist |
| `<id>.route` | arc_id | chronological reference arcs; **not aligned to GPS rows** |

Folder layout: `data/raw/<partition>/<trip_id>/<trip_id>.<ext>`; partitions
`train`, `student_test`, and the evaluator partition (logical name `eval_test`,
folder `evaluator_test`, demo day only).

## 2. Person 1 → everyone

```python
pathsnap.io.load_trip_input(trip_dir) -> TripInput        # reads track, nodes, arcs ONLY
pathsnap.io.load_reference_route(trip_dir) -> ndarray|None  # labels / scoring only
pathsnap.dataset.prepare_trip(TripInput, cfg) -> ProcessedTrip   # inference path, no labels
pathsnap.dataset.attach_labels(ProcessedTrip, route, cfg)         # adds labels in place
pathsnap.dataset.build_partition_tables(cfg, "train"|"student_test") -> (cands_df, obs_df)
pathsnap.features.FEATURE_COLUMNS   # ordered list; the model bundle checks it
```

`ProcessedTrip`: `trip, graph (RoadGraph), x, y, t (projected metres / seconds), cands, obs`.

**Candidate table** (`cands`, one row per observation × candidate arc, sorted by `obs_idx`, then distance):

| column | meaning |
|---|---|
| `trip_id`, `obs_idx` | observation key (row in the cleaned track) |
| `arc_id` | **canonical** arc id (duplicates merged, see `graph.py`) |
| `dist_m`, `proj_t`, `foot_x`, `foot_y` | point-to-segment distance, position along arc in [0,1], projected foot |
| `rank`, `n_cands` | distance rank, number of candidates of this observation |
| `FEATURE_COLUMNS` | 20 features, see `features.py` |
| `arc_from`, `arc_to` | node ids (for continuity metrics) |
| `label`, `obs_labeled` | only after `attach_labels`: 1 if this arc is the aligned reference arc |

**Observation table** (`obs`): `trip_id, obs_idx, t, lon, lat, x, y, n_cands` and,
after labelling, `ref_arc, ref_dist, route_pos, labeled, ref_in_cands`.

`RoadGraph` methods used downstream: `arc_from, arc_to, arc_len, canonical,
reverse_arc, node_lat, node_lon, proj, bounded_dijkstra(), shortest_path_arcs()`.

## 3. Person 2 → Person 3 / Person 4

`models/final_model.joblib` contains a `pathsnap.models.artifact.ModelBundle`:

```python
bundle = load_bundle(path)            # checks FEATURE_COLUMNS match
prob = bundle.predict_proba(cands)    # ndarray (len(cands),), P(candidate is reference)
bundle.model_name, bundle.params, bundle.metadata
```

## 4. Person 3 → Person 4

```python
matcher = HMMMapMatcher(processed_trip.graph, hmm_cfg)   # hmm_cfg = config "hmm" section (+ tuned params)
res = matcher.decode(cands, prob, x, y, t)               # DecodeResult
res.obs_arc        # ndarray (n_obs,), canonical arc per observation, -1 = no candidate
res.route          # list[int], gap-filled arc sequence
res.route_raw, res.segment_starts_obs, res.n_unfilled_gaps, res.timings
classifier_only(cands, prob, n_obs) -> ndarray          # baseline
evaluate_prediction(obs_arc, route, ref_arc, labeled, ref_route, graph) -> dict
```

Tuned decoder parameters: `results/hmm/best_hmm_params.json` → `{"params": {...}}`
(merged over the config `hmm` section by both `evaluate_hmm.py` and the API).

## 5. Person 4: HTTP API (Laptop 1 ↔ Laptop 2)

`GET /health` → `{"status","model","model_path","hmm_params_source","trip_dir","n_trips"}`

`GET /trips` → `{"partition": str, "trips": [{"trip_id","n_observations","duration_s"}]}`

`POST /predict` body `{"trip_id": "00000042"}` → `PredictResponse` (see `api/schemas.py`):
`trip_id, model_name, n_observations, n_matched_observations, n_hmm_segments,
n_unfilled_gaps, predicted_route_arc_ids, gps [[lat,lon,t]], predicted_route
[[[lat,lon]...]], classifier_only_segments, network_segments, snapped_points,
evaluation {available, hmm{metrics}, classifier_only{metrics}, note}, timings_s`.

Errors: 404 unknown trip, 422 malformed id / malformed trip files.

## 6. Leakage rules enforced in code

* `load_trip_input` never opens `.route`; `PathSnapPipeline.predict` accepts a `TripInput` only.
* Scoring (`PathSnapPipeline.evaluate`) runs after prediction and returns numbers only.
* `build_partition_tables(cfg, "eval_test")` raises `PermissionError`.
* Scalers live inside sklearn Pipelines and are fitted on training folds only.
* Model selection uses CV on train; decoder tuning uses out-of-fold probabilities on train.
  Student-test is used for reporting/comparison; eval_test only by the running API.

## 7. Label caveats (say this in the report)

Observation labels are *derived* by aligning GPS points to the reference route
(`labels.py`). They inherit the reference route's limitations (13 physically
disconnected consecutive arcs in 10 training routes, tracks that extend
beyond the supplied network, e.g. trip 00000054) and the alignment's own
approximations. Observations > `labels.max_label_distance_m` from their
aligned arc are left unlabelled.

"""End-to-end inference pipeline used by the API and the report maps (owner: Person 4).

    TripInput (track + nodes + arcs, NO route)
      -> prepare_trip (Person 1: projection, graph, candidates, features)
      -> ModelBundle.predict_proba (Person 2)
      -> classifier-only argmax + HMMMapMatcher.decode (Person 3)
      -> PredictionResult (arc ids + display geometry)

Scoring is a separate step, :meth:`PathSnapPipeline.evaluate`, which is the
only place that touches the reference route and runs strictly AFTER the
prediction has been produced.  Its output is a dict of numbers only.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import load_config, models_dir, results_dir
from .dataset import ProcessedTrip, attach_labels, prepare_trip
from .graph import RoadGraph
from .hmm.decoder import HMMMapMatcher, classifier_only, collapse
from .hmm.route_metrics import evaluate_prediction
from .io import TripInput
from .models.artifact import load_bundle

log = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    trip_id: str
    n_obs: int
    hmm_obs_arc: np.ndarray
    clf_obs_arc: np.ndarray
    hmm_route: list[int]
    clf_route: list[int]
    n_segments: int
    n_unfilled_gaps: int
    timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------- geometry
def route_polylines(route: list[int], g: RoadGraph) -> list[list[list[float]]]:
    """Arc sequence -> list of [lat, lon] polylines, split where arcs are not connected."""
    lines: list[list[list[float]]] = []
    cur: list[list[float]] = []
    prev = None
    for a in route:
        f, t = int(g.arc_from[a]), int(g.arc_to[a])
        if prev is None or int(g.arc_to[prev]) != f:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[float(g.node_lat[f]), float(g.node_lon[f])]]
        cur.append([float(g.node_lat[t]), float(g.node_lon[t])])
        prev = a
    if len(cur) > 1:
        lines.append(cur)
    return lines


def arc_segments(arcs, g: RoadGraph, limit: int | None = None) -> list[list[list[float]]]:
    arcs = list(dict.fromkeys(int(a) for a in arcs if a >= 0))[:limit]
    return [[[float(g.node_lat[g.arc_from[a]]), float(g.node_lon[g.arc_from[a]])],
             [float(g.node_lat[g.arc_to[a]]), float(g.node_lon[g.arc_to[a]])]] for a in arcs]


def snapped_points(pt: ProcessedTrip, obs_arc: np.ndarray) -> list[list[float]]:
    """Projection of each observation onto its decoded arc, as [lat, lon]."""
    c = pt.cands
    key = c["obs_idx"].to_numpy(np.int64) * (pt.graph.n_arcs + 1) + c["arc_id"].to_numpy(np.int64)
    want = np.arange(len(obs_arc)) * (pt.graph.n_arcs + 1) + obs_arc
    sel = np.isin(key, want[obs_arc >= 0])
    lon, lat = pt.graph.proj.inverse(c["foot_x"].to_numpy()[sel], c["foot_y"].to_numpy()[sel])
    return [[float(a), float(b)] for a, b in zip(lat, lon)]


# ---------------------------------------------------------------- pipeline
class PathSnapPipeline:
    def __init__(self, cfg: dict | None = None, model_path: str | Path | None = None,
                 hmm_params_path: str | Path | None = None):
        self.cfg = cfg or load_config()
        mp = model_path or os.environ.get("PATHSNAP_MODEL_PATH") or (models_dir(self.cfg) / "final_model.joblib")
        self.bundle = load_bundle(mp)
        self.model_path = str(mp)
        self.hmm_cfg = dict(self.cfg["hmm"])
        hp = Path(hmm_params_path) if hmm_params_path else results_dir(self.cfg, "hmm") / "best_hmm_params.json"
        self.hmm_params_source = "configs/pathsnap.yaml"
        if hp.exists():
            self.hmm_cfg.update(json.loads(hp.read_text())["params"])
            self.hmm_params_source = str(hp)
        log.info("pipeline ready: model=%s (%s), hmm params from %s", self.bundle.model_name, mp,
                 self.hmm_params_source)

    def predict(self, trip: TripInput) -> tuple[PredictionResult, ProcessedTrip]:
        if not isinstance(trip, TripInput):
            raise TypeError("predict() accepts a TripInput (track, nodes, arcs) only")
        t0 = time.perf_counter()
        pt = prepare_trip(trip, self.cfg)
        t1 = time.perf_counter()
        prob = self.bundle.predict_proba(pt.cands)
        t2 = time.perf_counter()
        clf = classifier_only(pt.cands, prob, len(pt.x))
        dec = HMMMapMatcher(pt.graph, self.hmm_cfg).decode(pt.cands, prob, pt.x, pt.y, pt.t)
        t3 = time.perf_counter()
        res = PredictionResult(trip.trip_id, trip.n_obs, dec.obs_arc, clf, dec.route, collapse(clf),
                               dec.n_segments, dec.n_unfilled_gaps,
                               {"preprocess_s": t1 - t0, "classifier_s": t2 - t1, "hmm_s": t3 - t2,
                                "total_s": t3 - t0, **{f"prep_{k}": v for k, v in pt.timings.items()}})
        return res, pt

    def evaluate(self, pt: ProcessedTrip, res: PredictionResult, reference_route: np.ndarray) -> dict:
        """Server-side scoring.  Returns numbers only - never the reference itself."""
        attach_labels(pt, reference_route, self.cfg)
        ref, lab = pt.obs.ref_arc.to_numpy(), pt.obs.labeled.to_numpy()
        return {"hmm": evaluate_prediction(res.hmm_obs_arc, res.hmm_route, ref, lab, reference_route, pt.graph),
                "classifier_only": evaluate_prediction(res.clf_obs_arc, res.clf_route, ref, lab,
                                                       reference_route, pt.graph)}

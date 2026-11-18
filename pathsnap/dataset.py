"""Trip processing and partition-level feature tables (owner: Person 1).

Two entry points with deliberately different data access:

* :func:`prepare_trip` - inference path.  Takes a :class:`TripInput` (no
  route) and returns candidates + features.  Used by training, HMM evaluation
  and the API alike, so preprocessing is identical everywhere.
* :func:`attach_labels` - adds labels from the reference route.  Called only
  for ``train`` / ``student_test`` tables and for server-side scoring.

Cached tables live in ``data/processed`` (git-ignored) and are rebuilt
automatically when the relevant config sections change.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .config import partition_dir, processed_dir
from .features import build_features
from .geo import LocalProjection
from .graph import RoadGraph
from .io import TripInput, list_trips, load_reference_route, load_trip_input
from .labels import align_route

log = logging.getLogger(__name__)

FORBIDDEN_DEV_PARTITIONS = {"eval_test"}


@dataclass
class ProcessedTrip:
    trip: TripInput
    graph: RoadGraph
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    cands: pd.DataFrame            # candidate rows + FEATURE_COLUMNS
    obs: pd.DataFrame              # one row per observation
    timings: dict = field(default_factory=dict)

    @property
    def trip_id(self) -> str:
        return self.trip.trip_id


def prepare_trip(trip: TripInput, cfg: dict) -> ProcessedTrip:
    """Projection, graph, candidates and features for one trip (no labels)."""
    t0 = time.perf_counter()
    proj = LocalProjection.from_points(trip.track[:, 0], trip.track[:, 1])
    graph = RoadGraph(trip.nodes, trip.arcs, proj)
    x, y = proj.forward(trip.track[:, 0], trip.track[:, 1])
    t = trip.track[:, 2].astype(float)
    t1 = time.perf_counter()
    cc = cfg["candidates"]
    cands = generate_candidates(x, y, graph, cc["radius_m"], cc["max_candidates"], cc["sample_spacing_m"])
    t2 = time.perf_counter()
    fc = cfg["features"]
    cands = build_features(cands, x, y, t, graph, fc["heading_half_window"],
                           fc["min_heading_displacement_m"], fc["context_half_window"])
    cands.insert(0, "trip_id", trip.trip_id)
    t3 = time.perf_counter()
    n_c = np.bincount(cands["obs_idx"].to_numpy(np.int64), minlength=len(x)) if len(cands) else np.zeros(len(x), int)
    obs = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": np.arange(len(x)), "t": t,
                        "lon": trip.track[:, 0], "lat": trip.track[:, 1], "x": x, "y": y,
                        "n_cands": n_c})
    return ProcessedTrip(trip, graph, x, y, t, cands, obs,
                         {"graph_s": t1 - t0, "candidates_s": t2 - t1, "features_s": t3 - t2})


def attach_labels(pt: ProcessedTrip, route: np.ndarray, cfg: dict) -> ProcessedTrip:
    """Add reference labels (training / student-test / server-side scoring only)."""
    al = align_route(pt.x, pt.y, pt.graph, route, cfg["labels"]["max_label_distance_m"])
    pt.obs["ref_arc"] = al.ref_arc
    pt.obs["ref_dist"] = al.ref_dist
    pt.obs["route_pos"] = al.route_pos
    pt.obs["labeled"] = al.labeled
    if len(pt.cands):
        o = pt.cands["obs_idx"].to_numpy(np.int64)
        pt.cands["label"] = (pt.cands["arc_id"].to_numpy() == al.ref_arc[o]).astype(np.int8)
        pt.cands["obs_labeled"] = al.labeled[o]
        hit = np.zeros(len(pt.x), dtype=bool)
        hit[o[pt.cands["label"].to_numpy() == 1]] = True
    else:
        pt.cands["label"] = pd.Series(dtype=np.int8)
        pt.cands["obs_labeled"] = pd.Series(dtype=bool)
        hit = np.zeros(len(pt.x), dtype=bool)
    pt.obs["ref_in_cands"] = hit
    return pt


def process_trip_dir(trip_dir: str | Path, cfg: dict, with_labels: bool = True) -> ProcessedTrip:
    trip = load_trip_input(trip_dir, cfg.get("cleaning", {}).get("drop_duplicate_timestamps", True))
    pt = prepare_trip(trip, cfg)
    if with_labels:
        route = load_reference_route(trip_dir, n_arcs=len(trip.arcs))
        if route is None:
            raise FileNotFoundError(f"{trip.trip_id}: no .route file; cannot build labels")
        attach_labels(pt, route, cfg)
    return pt


# ------------------------------------------------------------ partition tables
def _cfg_hash(cfg: dict) -> str:
    rel = {k: cfg.get(k) for k in ("cleaning", "labels", "candidates", "features")}
    return hashlib.sha1(json.dumps(rel, sort_keys=True).encode()).hexdigest()[:12]


def _process_for_table(trip_dir: str, cfg: dict):
    pt = process_trip_dir(trip_dir, cfg, with_labels=True)
    pt.obs["n_raw_rows"] = pt.trip.raw_track_rows
    return pt.cands, pt.obs, {"trip_id": pt.trip_id, **pt.timings}


def build_partition_tables(cfg: dict, partition: str, use_cache: bool = True, n_jobs: int = 1,
                           trip_ids: list[str] | None = None):
    """Concatenated candidate and observation tables for a development partition."""
    pdir = partition_dir(cfg, partition)
    # block by logical name AND by folder, so an alias or a direct folder name cannot bypass the guard
    if partition in FORBIDDEN_DEV_PARTITIONS or pdir.resolve() == partition_dir(cfg, "eval_test").resolve():
        raise PermissionError("the evaluator partition must not be used to build development tables")
    ids = trip_ids or list_trips(pdir)
    if not ids:
        raise FileNotFoundError(f"No trips found in {pdir}")
    h = _cfg_hash(cfg)
    tag = partition if trip_ids is None else f"{partition}_{hashlib.sha1(','.join(ids).encode()).hexdigest()[:8]}"
    cdir = processed_dir(cfg)
    cpath, opath, mpath = cdir / f"{tag}_cands.pkl", cdir / f"{tag}_obs.pkl", cdir / f"{tag}_meta.json"
    if use_cache and cpath.exists() and opath.exists() and mpath.exists():
        meta = json.loads(mpath.read_text())
        if meta.get("hash") == h and meta.get("trip_ids") == ids:
            log.info("loading cached %s tables (%s)", partition, h)
            return pd.read_pickle(cpath), pd.read_pickle(opath)

    t0 = time.perf_counter()
    dirs = [str(pdir / tid) for tid in ids]
    if n_jobs == 1:
        parts = [_process_for_table(d, cfg) for d in dirs]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs)(delayed(_process_for_table)(d, cfg) for d in dirs)
    cands = pd.concat([p[0] for p in parts], ignore_index=True)
    obs = pd.concat([p[1] for p in parts], ignore_index=True)
    cands.to_pickle(cpath)
    obs.to_pickle(opath)
    mpath.write_text(json.dumps({"hash": h, "trip_ids": ids, "build_s": time.perf_counter() - t0,
                                 "timings": [p[2] for p in parts]}, indent=1))
    log.info("built %s tables: %d candidate rows, %d observations in %.1fs",
             partition, len(cands), len(obs), time.perf_counter() - t0)
    return cands, obs

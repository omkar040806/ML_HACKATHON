"""Feature engineering for (observation, candidate arc) pairs (owner: Person 1).

Every feature uses only GPS observations and road-network geometry, never the
reference route, so the same code runs unchanged at inference time.

Feature groups and motivation
-----------------------------
* **Geometric proximity** - ``dist_m``, ``rank``, ``dist_delta_min``,
  ``dist_ratio_min``, ``n_cands``: GPS noise is roughly isotropic, so closer
  arcs are more likely; the *relative* terms make the feature comparable
  between dense city centres and sparse highways.
* **Direction** - ``heading_cos``, ``heading_diff_deg``, ``heading_valid``:
  every two-way street is two arcs with identical geometry and opposite
  bearing.  Only the travel direction can separate them.
* **Temporal context** - ``ctx_dist_mean``, ``ctx_dist_max``,
  ``ctx_delta_min``: distance of the neighbouring observations (i-w .. i+w)
  to the same arc.  A single noisy fix may be closest to a side street, but
  its neighbours usually are not.
* **Arc / topology** - ``log_arc_len``, ``proj_t``, ``proj_at_end``,
  ``two_way``, ``from_degree``, ``to_degree``: projections that clamp to an
  arc end-point and junction-heavy areas are typical failure cases.
* **Sampling** - ``speed_mps``, ``log_dt_prev``, ``gap_flag``: gaps and low
  speed make heading and context less reliable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff_deg, bearing_deg, point_segment_distance
from .graph import RoadGraph

FEATURE_COLUMNS = [
    "dist_m", "rank", "n_cands", "dist_delta_min", "dist_ratio_min",
    "heading_cos", "heading_diff_deg", "heading_valid", "speed_mps",
    "ctx_dist_mean", "ctx_dist_max", "ctx_delta_min",
    "log_arc_len", "proj_t", "proj_at_end", "two_way", "from_degree", "to_degree",
    "log_dt_prev", "gap_flag",
]


def track_kinematics(x: np.ndarray, y: np.ndarray, t: np.ndarray, half_window: int = 2,
                     min_displacement_m: float = 3.0) -> pd.DataFrame:
    """Per-observation heading, speed and sampling-gap descriptors."""
    n = len(x)
    i = np.arange(n)
    i0 = np.clip(i - half_window, 0, n - 1)
    i1 = np.clip(i + half_window, 0, n - 1)
    dx, dy = x[i1] - x[i0], y[i1] - y[i0]
    disp = np.hypot(dx, dy)
    dt = t[i1] - t[i0]
    speed = np.where(dt > 0, disp / np.where(dt > 0, dt, 1), 0.0)
    dt_prev = np.concatenate([[0.0], np.diff(t)])
    dt_next = np.concatenate([np.diff(t), [0.0]])
    return pd.DataFrame({
        "heading": bearing_deg(dx, dy),
        "heading_valid": (disp >= min_displacement_m).astype(np.int8),
        "speed_mps": np.clip(speed, 0, 80),
        "dt_prev": dt_prev,
        "dt_next": dt_next,
    })


def build_features(cands: pd.DataFrame, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                   graph: RoadGraph, half_window: int = 2, min_displacement_m: float = 3.0,
                   context_half_window: int = 3) -> pd.DataFrame:
    """Return ``cands`` with :data:`FEATURE_COLUMNS` (and ``arc_from``/``arc_to``) appended."""
    out = cands.copy()
    if len(out) == 0:
        for c in FEATURE_COLUMNS + ["arc_from", "arc_to"]:
            out[c] = pd.Series(dtype=float)
        return out
    obs = out["obs_idx"].to_numpy(np.int64)
    arc = out["arc_id"].to_numpy(np.int64)
    dist = out["dist_m"].to_numpy(float)
    kin = track_kinematics(x, y, t, half_window, min_displacement_m)

    min_d = out.groupby("obs_idx")["dist_m"].transform("min").to_numpy()
    out["dist_delta_min"] = dist - min_d
    out["dist_ratio_min"] = (min_d + 1.0) / (dist + 1.0)

    hv = kin["heading_valid"].to_numpy()[obs].astype(bool) & (graph.arc_len[arc] > 0)
    diff = angle_diff_deg(kin["heading"].to_numpy()[obs], graph.arc_bearing[arc])
    out["heading_cos"] = np.where(hv, np.cos(np.radians(diff)), 0.0)
    out["heading_diff_deg"] = np.where(hv, diff, 90.0)
    out["heading_valid"] = hv.astype(np.int8)
    out["speed_mps"] = kin["speed_mps"].to_numpy()[obs]

    n = len(x)
    ctx = []
    ax, ay, bx, by = graph.ax[arc], graph.ay[arc], graph.bx[arc], graph.by[arc]
    for k in range(-context_half_window, context_half_window + 1):
        if k == 0:
            continue
        j = np.clip(obs + k, 0, n - 1)
        d, *_ = point_segment_distance(x[j], y[j], ax, ay, bx, by)
        ctx.append(d)
    ctx = np.vstack(ctx) if ctx else dist[None]
    out["ctx_dist_mean"] = ctx.mean(axis=0)
    out["ctx_dist_max"] = ctx.max(axis=0)
    out["ctx_delta_min"] = out["ctx_dist_mean"] - out.groupby("obs_idx")["ctx_dist_mean"].transform("min")

    out["log_arc_len"] = np.log1p(graph.arc_len[arc])
    pt = out["proj_t"].to_numpy()
    out["proj_at_end"] = ((pt <= 0.0) | (pt >= 1.0)).astype(np.int8)
    out["two_way"] = (graph.reverse_arc[arc] >= 0).astype(np.int8)
    fn, tn = graph.arc_from[arc], graph.arc_to[arc]
    out["from_degree"] = graph.out_degree[fn] + graph.in_degree[fn]
    out["to_degree"] = graph.out_degree[tn] + graph.in_degree[tn]
    dtp = kin["dt_prev"].to_numpy()[obs]
    dtn = kin["dt_next"].to_numpy()[obs]
    out["log_dt_prev"] = np.log1p(dtp)
    out["gap_flag"] = ((dtp > 5) | (dtn > 5)).astype(np.int8)
    out["arc_from"] = fn
    out["arc_to"] = tn
    return out

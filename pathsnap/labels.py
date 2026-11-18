"""Observation-level label construction (owner: Person 1).

Why this is needed
------------------
The ``.route`` file is an ordered list of arc ids.  It is NOT aligned to the
GPS rows (525 route arcs vs 1 315 GPS points in trip 00000001, for example),
so there is no ready-made "correct arc" per GPS observation.

Method: monotone alignment by dynamic programming
-------------------------------------------------
Each observation ``i`` is assigned a position ``j(i)`` in the route such that
``j`` is non-decreasing in time (the vehicle cannot go back along the route)
and the total point-to-segment distance ``sum_i d(i, route[j(i)])`` is
minimal.  Recurrence (``C`` = cost, ``d`` = distance in metres)::

    C[0, j] = d[0, j]
    C[i, j] = d[i, j] + min_{k <= j} C[i-1, k]

The inner ``min`` is a running prefix minimum, so the whole DP is
``O(n_obs * route_len)`` and vectorised per observation.  Skipping route
arcs is allowed (needed for GPS gaps and for very short arcs).

Observations whose aligned distance exceeds ``max_label_distance_m`` are
marked *unlabelled*: they are still predicted, but excluded from training and
from label-based metrics.  The reference route is treated as reference
information, not error-free truth (see docs/INTERFACES.md, "Label caveats").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import point_segment_distance
from .graph import RoadGraph


@dataclass
class Alignment:
    route_pos: np.ndarray   # (n_obs,) index into the route sequence
    ref_arc: np.ndarray     # (n_obs,) canonical reference arc id
    ref_dist: np.ndarray    # (n_obs,) distance obs -> reference arc [m]
    labeled: np.ndarray     # (n_obs,) bool

    @property
    def labeled_frac(self) -> float:
        return float(self.labeled.mean()) if len(self.labeled) else 0.0


def obs_to_route_distances(x: np.ndarray, y: np.ndarray, graph: RoadGraph, route: np.ndarray,
                           chunk: int = 2048) -> np.ndarray:
    """(n_obs, len(route)) matrix of point-to-arc distances (float32)."""
    ax, ay, bx, by = graph.ax[route], graph.ay[route], graph.bx[route], graph.by[route]
    out = np.empty((len(x), len(route)), dtype=np.float32)
    for s in range(0, len(x), chunk):
        e = min(s + chunk, len(x))
        d, *_ = point_segment_distance(x[s:e, None], y[s:e, None], ax[None], ay[None], bx[None], by[None])
        out[s:e] = d
    return out


def monotone_align(dist: np.ndarray) -> np.ndarray:
    """Solve the monotone assignment DP; returns route position per observation."""
    n, m = dist.shape
    if m == 0:
        raise ValueError("empty route")
    back = np.empty((n, m), dtype=np.int32)
    prev = dist[0].astype(np.float64)
    ar = np.arange(m, dtype=np.int32)
    for i in range(1, n):
        pm = np.minimum.accumulate(prev)
        is_min = prev <= pm
        arg = np.maximum.accumulate(np.where(is_min, ar, 0))
        back[i] = arg
        prev = dist[i] + pm
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos


def align_route(x: np.ndarray, y: np.ndarray, graph: RoadGraph, route: np.ndarray,
                max_label_distance_m: float) -> Alignment:
    route_c = graph.canonical[np.asarray(route, dtype=np.int64)]
    dist = obs_to_route_distances(x, y, graph, route_c)
    pos = monotone_align(dist)
    ref_dist = dist[np.arange(len(x)), pos].astype(float)
    return Alignment(route_pos=pos, ref_arc=route_c[pos], ref_dist=ref_dist,
                     labeled=ref_dist <= max_label_distance_m)


def route_continuity_breaks(route: np.ndarray, graph: RoadGraph) -> int:
    """Number of consecutive reference arcs that are not physically connected."""
    r = np.asarray(route, dtype=np.int64)
    if len(r) < 2:
        return 0
    return int((graph.arc_to[r[:-1]] != graph.arc_from[r[1:]]).sum())

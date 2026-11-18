"""Candidate road-segment generation (owner: Person 1).

For every GPS observation we return up to ``max_candidates`` canonical arcs
whose point-to-segment distance is at most ``radius_m``.

Spatial index: each arc near the trajectory is densified into points every
``sample_spacing_m`` metres and the points go into a KD-tree.  If a segment
lies within ``r`` of an observation then one of its sample points lies
within ``r + spacing / 2``, so querying with that radius and then computing
the exact point-to-segment distance gives an exact radius search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import point_segment_distance
from .graph import RoadGraph

CANDIDATE_COLUMNS = ["obs_idx", "arc_id", "dist_m", "proj_t", "foot_x", "foot_y", "rank", "n_cands"]


def arcs_near_bbox(graph: RoadGraph, x: np.ndarray, y: np.ndarray, margin: float) -> np.ndarray:
    """Canonical arc ids whose bounding box intersects the track bbox + margin."""
    xmin, xmax = x.min() - margin, x.max() + margin
    ymin, ymax = y.min() - margin, y.max() + margin
    ok = (np.minimum(graph.ax, graph.bx) <= xmax) & (np.maximum(graph.ax, graph.bx) >= xmin) & \
         (np.minimum(graph.ay, graph.by) <= ymax) & (np.maximum(graph.ay, graph.by) >= ymin)
    return np.nonzero(ok & graph.is_canonical)[0]


def _densify(graph: RoadGraph, arc_ids: np.ndarray, spacing: float):
    n_s = np.maximum(1, np.ceil(graph.arc_len[arc_ids] / spacing).astype(np.int64)) + 1
    rep = np.repeat(arc_ids, n_s)
    starts = np.cumsum(n_s) - n_s
    k = np.arange(n_s.sum()) - np.repeat(starts, n_s)
    t = k / np.repeat(n_s - 1, n_s)
    px = graph.ax[rep] + t * (graph.bx[rep] - graph.ax[rep])
    py = graph.ay[rep] + t * (graph.by[rep] - graph.ay[rep])
    return np.column_stack([px, py]), rep


def generate_candidates(x: np.ndarray, y: np.ndarray, graph: RoadGraph, radius_m: float = 50.0,
                        max_candidates: int = 8, sample_spacing_m: float = 10.0) -> pd.DataFrame:
    """Candidate table with columns :data:`CANDIDATE_COLUMNS`, sorted by (obs_idx, dist)."""
    if radius_m <= 0 or max_candidates <= 0 or sample_spacing_m <= 0:
        raise ValueError("radius_m, max_candidates and sample_spacing_m must be positive")
    empty = pd.DataFrame({c: pd.Series(dtype=float) for c in CANDIDATE_COLUMNS})
    arc_ids = arcs_near_bbox(graph, x, y, radius_m)
    if len(arc_ids) == 0:
        return empty
    pts, sample_arc = _densify(graph, arc_ids, sample_spacing_m)
    tree = cKDTree(pts)
    hits = tree.query_ball_point(np.column_stack([x, y]), r=radius_m + sample_spacing_m / 2 + 1e-6)
    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if lens.sum() == 0:
        return empty
    obs = np.repeat(np.arange(len(x)), lens)
    samp = np.fromiter((s for h in hits for s in h), dtype=np.int64, count=int(lens.sum()))
    arc = sample_arc[samp]
    key = np.unique(obs * graph.n_arcs + arc)
    obs, arc = key // graph.n_arcs, key % graph.n_arcs

    d, t, fx, fy = point_segment_distance(x[obs], y[obs], graph.ax[arc], graph.ay[arc],
                                          graph.bx[arc], graph.by[arc])
    keep = d <= radius_m
    obs, arc, d, t, fx, fy = obs[keep], arc[keep], d[keep], t[keep], fx[keep], fy[keep]
    order = np.lexsort((arc, d, obs))   # by obs, then distance, then arc id (deterministic)
    obs, arc, d, t, fx, fy = obs[order], arc[order], d[order], t[order], fx[order], fy[order]

    first = np.concatenate([[True], obs[1:] != obs[:-1]])
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(obs)), 0))
    rank = np.arange(len(obs)) - grp_start
    keep = rank < max_candidates
    obs, arc, d, t, fx, fy, rank = (v[keep] for v in (obs, arc, d, t, fx, fy, rank))
    n_c = np.bincount(obs, minlength=len(x))[obs]
    return pd.DataFrame({"obs_idx": obs, "arc_id": arc, "dist_m": d, "proj_t": t,
                         "foot_x": fx, "foot_y": fy, "rank": rank, "n_cands": n_c})

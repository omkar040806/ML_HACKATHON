"""HMM map matcher: candidates + probabilities -> per-observation arcs and a route (owner: Person 3).

Inputs (from Person 1 / Person 2):
  * ``cands``: candidate table with at least ``obs_idx, arc_id, dist_m, proj_t``
  * ``prob``:  P(candidate is reference) per row from ``ModelBundle.predict_proba``
  * projected GPS ``x, y`` and timestamps ``t``

Processing:
  1. HMM steps = observations that have >= 1 candidate (observations with no
     candidate within the radius are not forced onto a road; they get -1).
  2. A step whose time gap to the previous step exceeds ``gap_break_s`` gets
     no transition -> Viterbi starts a new segment there.
  3. Viterbi decodes each segment.
  4. Route reconstruction: repeated arcs are collapsed and every pair of
     consecutive arcs that is not directly connected is joined with the
     directed shortest path (if one exists within ``fill_gap_max_m``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..graph import RoadGraph
from .emission import log_emission
from .transition import TransitionModel
from .viterbi import viterbi


@dataclass
class DecodeResult:
    obs_arc: np.ndarray                 # (n_obs,) decoded canonical arc id, -1 = no candidate
    route_raw: list[int]                # decoded arcs, consecutive repeats collapsed
    route: list[int]                    # route_raw with shortest-path gap filling
    segment_starts_obs: list[int]       # observation index where each HMM segment starts
    n_unfilled_gaps: int = 0
    timings: dict = field(default_factory=dict)

    @property
    def n_segments(self) -> int:
        return len(self.segment_starts_obs)


def collapse(seq) -> list[int]:
    out: list[int] = []
    for a in seq:
        if a < 0:
            continue
        if not out or out[-1] != a:
            out.append(int(a))
    return out


def fill_route(route_raw: list[int], graph: RoadGraph, fill_max_m: float) -> tuple[list[int], int]:
    """Insert shortest paths between disconnected consecutive arcs."""
    if not route_raw:
        return [], 0
    out = [route_raw[0]]
    unfilled = 0
    for b in route_raw[1:]:
        a = out[-1]
        if graph.arc_to[a] != graph.arc_from[b]:
            path = graph.shortest_path_arcs(int(graph.arc_to[a]), int(graph.arc_from[b]), fill_max_m)
            if path is None:
                unfilled += 1
            else:
                out.extend(int(p) for p in path)
        if out[-1] != b:
            out.append(int(b))
    return out, unfilled


def classifier_only(cands: pd.DataFrame, prob: np.ndarray, n_obs: int) -> np.ndarray:
    """Independent per-observation argmax (no sequence model)."""
    obs_arc = np.full(n_obs, -1, dtype=np.int64)
    if len(cands) == 0:
        return obs_arc
    o = cands["obs_idx"].to_numpy(np.int64)
    order = np.lexsort((-np.asarray(prob), o))
    o_s = o[order]
    first = np.concatenate([[True], o_s[1:] != o_s[:-1]])
    obs_arc[o_s[first]] = cands["arc_id"].to_numpy(np.int64)[order][first]
    return obs_arc


class HMMMapMatcher:
    def __init__(self, graph: RoadGraph, hmm_cfg: dict):
        self.g = graph
        self.cfg = hmm_cfg
        self.trans = TransitionModel(graph, hmm_cfg["beta_m"], hmm_cfg["max_speed_mps"],
                                     hmm_cfg["route_search_slack_m"], hmm_cfg["route_search_factor"],
                                     hmm_cfg.get("backward_tolerance_m", 10.0))

    def decode(self, cands: pd.DataFrame, prob: np.ndarray | None, x: np.ndarray, y: np.ndarray,
               t: np.ndarray, emission_mode: str | None = None) -> DecodeResult:
        t0 = time.perf_counter()
        n_obs = len(x)
        c = self.cfg
        mode = emission_mode or c["emission"]
        obs_arc = np.full(n_obs, -1, dtype=np.int64)
        if len(cands) == 0:
            return DecodeResult(obs_arc, [], [], [], 0, {"decode_s": 0.0})
        o = cands["obs_idx"].to_numpy(np.int64)
        em_all = log_emission(cands["dist_m"].to_numpy(float), prob, mode, c["sigma_m"], c["min_prob"],
                              c.get("emission_weight", 1.0))
        arcs_all = cands["arc_id"].to_numpy(np.int64)
        tp_all = cands["proj_t"].to_numpy(float)
        bounds = np.flatnonzero(np.concatenate([[True], o[1:] != o[:-1], [True]]))
        step_obs = o[bounds[:-1]]
        sl = [slice(bounds[k], bounds[k + 1]) for k in range(len(step_obs))]
        emissions = [em_all[s] for s in sl]

        def trans(s: int):
            i_prev, i_cur = step_obs[s - 1], step_obs[s]
            dt = float(t[i_cur] - t[i_prev])
            if dt > c["gap_break_s"]:
                return None
            gc = float(np.hypot(x[i_cur] - x[i_prev], y[i_cur] - y[i_prev]))
            rd = self.trans.route_distances(arcs_all[sl[s - 1]], tp_all[sl[s - 1]], arcs_all[sl[s]],
                                            tp_all[sl[s]], gc, dt)
            return self.trans.log_prob(rd, gc)

        path, starts = viterbi(emissions, trans)
        t1 = time.perf_counter()
        for k, j in enumerate(path):
            obs_arc[step_obs[k]] = arcs_all[sl[k]][j]
        route_raw = collapse(obs_arc)
        route, unfilled = fill_route(route_raw, self.g, c["fill_gap_max_m"])
        t2 = time.perf_counter()
        return DecodeResult(obs_arc, route_raw, route, [int(step_obs[s]) for s in starts], unfilled,
                            {"viterbi_s": t1 - t0, "route_fill_s": t2 - t1, "decode_s": t2 - t0})

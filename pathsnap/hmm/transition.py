"""Transition scores (owner: Person 3).

Following Newson & Krumm (2009): a transition from candidate i (arc a,
projection fraction t_a) at step s-1 to candidate j (arc b, fraction t_b)
at step s is scored by how well the *network* distance between the two
projected positions agrees with the straight-line distance between the two
GPS fixes:

    log P(j | i) = -| d_route(i, j) - d_gc | / beta

d_route:
  * same arc, moving forward (or backward by less than the jitter
    tolerance):          |t_b - t_a| * len(a)   (backward jitter -> 0)
  * otherwise:           (1 - t_a) len(a) + shortest_path(to(a), from(b)) + t_b len(b)
    The shortest path respects arc direction, so jumping to an unconnected
    or opposite-direction arc costs a detour or is impossible.

Unreachable pairs (no path within the search limit) get -inf.  The limit is
``min(factor * d_gc + slack, max_speed * dt + slack)`` so a vehicle cannot
"teleport" further than physically possible in the elapsed time.
"""
from __future__ import annotations

import math

import numpy as np

from ..graph import RoadGraph


class TransitionModel:
    def __init__(self, graph: RoadGraph, beta_m: float = 5.0, max_speed_mps: float = 70.0,
                 slack_m: float = 150.0, factor: float = 3.0, backward_tolerance_m: float = 10.0):
        if beta_m <= 0:
            raise ValueError("beta_m must be positive")
        self.g = graph
        self.beta = beta_m
        self.max_speed = max_speed_mps
        self.slack = slack_m
        self.factor = factor
        self.back_tol = backward_tolerance_m

    def search_limit(self, gc: float, dt: float) -> float:
        return min(self.factor * gc + self.slack, self.max_speed * max(dt, 1.0) + self.slack)

    def route_distances(self, arcs_p: np.ndarray, t_p: np.ndarray, arcs_c: np.ndarray, t_c: np.ndarray,
                        gc: float, dt: float) -> np.ndarray:
        g = self.g
        limit = self.search_limit(gc, dt)
        len_p = g.arc_len[arcs_p]
        len_c = g.arc_len[arcs_c]
        rem_p = (1.0 - t_p) * len_p           # distance to the end of the previous arc
        off_c = t_c * len_c                   # distance from the start of the current arc
        from_c = g.arc_from[arcs_c].tolist()
        out = np.full((len(arcs_p), len(arcs_c)), np.inf)
        for i, a in enumerate(arcs_p.tolist()):
            dist, _ = g.bounded_dijkstra(int(g.arc_to[a]), limit)
            base = rem_p[i]
            row = out[i]
            for j, b in enumerate(arcs_c.tolist()):
                if a == b:
                    delta = (t_c[j] - t_p[i]) * len_p[i]
                    if delta >= -self.back_tol:
                        row[j] = max(delta, 0.0)
                        continue
                sp = dist.get(from_c[j], math.inf)
                if sp < math.inf:
                    d = base + sp + off_c[j]
                    if d <= limit + len_c[j]:
                        row[j] = d
        return out

    def log_prob(self, route_d: np.ndarray, gc: float) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            lp = -np.abs(route_d - gc) / self.beta
        lp[~np.isfinite(route_d)] = -np.inf
        return lp

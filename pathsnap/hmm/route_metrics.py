"""Route-level evaluation (owner: Person 3; also used server-side by the API).

All comparisons use canonical arc ids (see pathsnap/graph.py).

* ``pointwise_accuracy``: fraction of labelled observations whose predicted
  arc equals the aligned reference arc; observations without a prediction
  count as wrong.
* ``direction_agnostic_accuracy``: same, but the reverse arc also counts as
  correct (diagnoses direction errors on two-way streets).
* ``continuity``: fraction of consecutive distinct arcs that are physically
  connected (to_node(a) == from_node(b)).
* ``route_set_metrics``: precision / recall / F1 between the predicted and
  reference arc sets, both by count and weighted by arc length, plus the
  Newson & Krumm *route mismatch fraction*
  (length wrongly added + length wrongly omitted) / reference length.
* ``edit_similarity``: 1 - Levenshtein(pred_route, ref_route) / max(len),
  an order-aware sequence-level score.
"""
from __future__ import annotations

import numpy as np

from ..graph import RoadGraph


def pointwise_accuracy(pred: np.ndarray, ref: np.ndarray, labeled: np.ndarray) -> float:
    lab = np.asarray(labeled, bool)
    if not lab.any():
        return float("nan")
    return float((np.asarray(pred)[lab] == np.asarray(ref)[lab]).mean())


def direction_agnostic_accuracy(pred: np.ndarray, ref: np.ndarray, labeled: np.ndarray, graph: RoadGraph) -> float:
    lab = np.asarray(labeled, bool)
    if not lab.any():
        return float("nan")
    p, r = np.asarray(pred)[lab], np.asarray(ref)[lab]
    rev = graph.reverse_arc[r]
    return float(((p == r) | ((p == rev) & (rev >= 0))).mean())


def continuity(route, graph: RoadGraph) -> float:
    r = [a for i, a in enumerate(route) if i == 0 or a != route[i - 1]]
    if len(r) < 2:
        return 1.0 if r else float("nan")
    r = np.asarray(r, dtype=np.int64)
    return float((graph.arc_to[r[:-1]] == graph.arc_from[r[1:]]).mean())


def route_set_metrics(pred_route, ref_route, graph: RoadGraph) -> dict:
    P = set(int(a) for a in pred_route)
    R = set(int(a) for a in graph.canonical[np.asarray(ref_route, dtype=np.int64)])
    inter = P & R
    lp = graph.arc_len[list(P)].sum() if P else 0.0
    lr = graph.arc_len[list(R)].sum() if R else 0.0
    li = graph.arc_len[list(inter)].sum() if inter else 0.0
    prec = len(inter) / len(P) if P else 0.0
    rec = len(inter) / len(R) if R else 0.0
    lprec = li / lp if lp > 0 else 0.0
    lrec = li / lr if lr > 0 else 0.0
    f = lambda a, b: 2 * a * b / (a + b) if a + b > 0 else 0.0  # noqa: E731
    return {"route_precision": prec, "route_recall": rec, "route_f1": f(prec, rec),
            "route_len_precision": float(lprec), "route_len_recall": float(lrec),
            "route_len_f1": float(f(lprec, lrec)),
            "route_mismatch_fraction": float(((lp - li) + (lr - li)) / lr) if lr > 0 else float("nan")}


def edit_similarity(a, b) -> float:
    a = np.asarray(list(a), dtype=np.int64)
    b = np.asarray(list(b), dtype=np.int64)
    n, m = len(a), len(b)
    if max(n, m) == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    prev = np.arange(m + 1, dtype=np.int64)
    ar = np.arange(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        sub = prev[:-1] + (b != a[i - 1])
        dele = prev[1:] + 1
        tmp = np.empty(m + 1, dtype=np.int64)
        tmp[0] = i
        tmp[1:] = np.minimum(sub, dele)
        cur = np.minimum.accumulate(tmp - ar) + ar   # insertions
        prev = cur
    return float(1 - prev[-1] / max(n, m))


def evaluate_prediction(obs_arc: np.ndarray, route, ref_arc: np.ndarray, labeled: np.ndarray,
                        ref_route, graph: RoadGraph) -> dict:
    ref_c = graph.canonical[np.asarray(ref_route, dtype=np.int64)]
    out = {"pointwise_accuracy": pointwise_accuracy(obs_arc, ref_arc, labeled),
           "direction_agnostic_accuracy": direction_agnostic_accuracy(obs_arc, ref_arc, labeled, graph),
           "route_continuity": continuity(list(route), graph),
           "edit_similarity": edit_similarity(route, ref_c),
           "n_labeled_obs": int(np.asarray(labeled).sum()),
           "predicted_route_arcs": int(len(route)), "reference_route_arcs": int(len(ref_c))}
    out.update(route_set_metrics(route, ref_route, graph))
    return out

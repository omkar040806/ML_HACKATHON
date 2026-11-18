"""Classification and observation-level metrics (owner: Person 2).

Definitions (also used in the report)
-------------------------------------
* Candidate-level metrics (accuracy, precision, recall, F1, macro-F1, log
  loss, confusion matrix) are computed on candidate rows of *labelled*
  observations with threshold ``threshold`` on P(candidate is reference).
* **Observation-level Top-1 accuracy**: for every labelled observation pick
  the candidate with the highest probability; correct if it equals the
  reference arc.  Observations with no candidates or whose reference arc is
  not among the candidates count as **errors** (denominator = all labelled
  observations), so the number is an honest end-to-end figure.
* ``top1_given_candidate``: same but only over observations whose reference
  arc is in the candidate set (isolates the classifier from candidate misses).
* **Candidate miss rate**: fraction of labelled observations whose reference
  arc is absent from the candidate set (model independent).
* **Route continuity** (classifier-only): order the argmax arcs by time,
  collapse repeats, and report the fraction of consecutive arc pairs that are
  physically connected (``to_node(a) == from_node(b)``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)


def candidate_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion": cm.tolist(),
    }


def argmax_per_obs(cands: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """One row per (trip_id, obs_idx): the arc with highest probability."""
    df = cands[["trip_id", "obs_idx", "arc_id", "arc_from", "arc_to"]].copy()
    if "label" in cands.columns:
        df["label"] = cands["label"].to_numpy()
    df["prob"] = prob
    df = df.sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False], kind="mergesort")
    return df.drop_duplicates(["trip_id", "obs_idx"], keep="first").reset_index(drop=True)


def sequence_continuity(arc_seq, arc_from_seq, arc_to_seq) -> tuple[float, int, int]:
    """(fraction connected, n_connected, n_pairs) after collapsing repeated arcs."""
    a = np.asarray(arc_seq)
    if len(a) == 0:
        return float("nan"), 0, 0
    keep = np.concatenate([[True], a[1:] != a[:-1]])
    fr, to = np.asarray(arc_from_seq)[keep], np.asarray(arc_to_seq)[keep]
    if len(fr) < 2:
        return 1.0, 0, 0
    ok = to[:-1] == fr[1:]
    return float(ok.mean()), int(ok.sum()), int(len(ok))


def top1_and_continuity(cands: pd.DataFrame, obs: pd.DataFrame, prob: np.ndarray) -> dict:
    """Observation-level metrics; ``obs`` defines which observations are evaluated."""
    best = argmax_per_obs(cands, prob)
    lab = obs[obs["labeled"]][["trip_id", "obs_idx", "ref_arc", "ref_in_cands"]]
    m = lab.merge(best[["trip_id", "obs_idx", "arc_id"]], on=["trip_id", "obs_idx"], how="left")
    correct = (m["arc_id"].to_numpy() == m["ref_arc"].to_numpy())
    inc = m["ref_in_cands"].to_numpy(bool)
    n_conn = n_pairs = 0
    per_trip = []
    for tid, g in best.groupby("trip_id", sort=False):
        frac, c, n = sequence_continuity(g.arc_id.to_numpy(), g.arc_from.to_numpy(), g.arc_to.to_numpy())
        n_conn += c
        n_pairs += n
        per_trip.append(frac)
    return {
        "top1_accuracy": float(correct.mean()) if len(m) else float("nan"),
        "top1_given_candidate": float(correct[inc].mean()) if inc.any() else float("nan"),
        "candidate_miss_rate": float(1 - inc.mean()) if len(m) else float("nan"),
        "route_continuity": float(n_conn / n_pairs) if n_pairs else float("nan"),
        "route_continuity_trip_mean": float(np.nanmean(per_trip)) if per_trip else float("nan"),
        "n_labeled_obs": int(len(m)),
    }


def evaluate_all(cands: pd.DataFrame, obs: pd.DataFrame, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Candidate-level metrics on labelled rows + observation-level metrics."""
    mask = cands["obs_labeled"].to_numpy(bool)
    out = candidate_metrics(cands["label"].to_numpy()[mask], prob[mask], threshold)
    out.update(top1_and_continuity(cands, obs, prob))
    return out

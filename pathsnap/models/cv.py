"""Training helpers and trip-level group cross-validation (owner: Person 2).

* Rows of the same trip never appear in both a training and a validation
  fold (``GroupKFold`` on ``trip_id``).
* ``obs_stride``: 1 Hz tracks are highly redundant; fitting on every k-th
  observation keeps trips balanced and training fast.  Candidate groups are
  kept intact (all candidates of a chosen observation are used).
* Slow models (KNN, GNB, MLP) are fitted on a random subsample of whole
  observations capped at ``slow_model_max_rows`` rows; the cap is recorded
  with the results.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..features import FEATURE_COLUMNS
from .metrics import evaluate_all
from .zoo import build_model, get_spec


def stride_mask(df: pd.DataFrame, stride: int) -> np.ndarray:
    return (df["obs_idx"].to_numpy() % max(1, stride)) == 0


def training_rows(cands: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Labelled candidate rows on the observation stride."""
    return cands[cands["obs_labeled"].to_numpy(bool) & stride_mask(cands, stride)]


def subsample_obs(rows: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    keys = rows[["trip_id", "obs_idx"]].drop_duplicates()
    frac = max_rows / len(rows)
    keep = keys.sample(frac=frac, random_state=seed)
    return rows.merge(keep, on=["trip_id", "obs_idx"], how="inner")


def fit(name: str, params: dict | None, rows: pd.DataFrame, seed: int, n_jobs: int = -1,
        slow_cap: int | None = None):
    """Fit model ``name`` on candidate rows; returns (model, params, info)."""
    spec = get_spec(name)
    if spec.slow and slow_cap:
        rows = subsample_obs(rows, slow_cap, seed)
    model, p = build_model(name, params, seed, n_jobs)
    X = rows[FEATURE_COLUMNS].to_numpy(np.float32)
    y = rows["label"].to_numpy(np.int8)
    if len(np.unique(y)) < 2:
        raise ValueError("training rows contain a single class")
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, p, {"fit_s": time.perf_counter() - t0, "fit_rows": int(len(rows)),
                      "fit_pos_rate": float(y.mean())}


def predict_proba(model, cands: pd.DataFrame) -> tuple[np.ndarray, float]:
    if len(cands) == 0:
        return np.zeros(0), 0.0
    t0 = time.perf_counter()
    p = model.predict_proba(cands[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
    return p, time.perf_counter() - t0


def group_folds(cands: pd.DataFrame, n_splits: int, seed: int):
    trips = np.array(sorted(cands["trip_id"].unique()))
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(gkf.split(trips, groups=trips)):
        yield k, set(trips[tr]), set(trips[va])


def group_cv(name: str, params: dict | None, cands: pd.DataFrame, obs: pd.DataFrame, n_splits: int,
             stride: int, seed: int, threshold: float = 0.5, n_jobs: int = -1,
             slow_cap: int | None = None, on_fold=None) -> pd.DataFrame:
    """Per-fold metrics.  Validation uses the same observation stride as training.

    ``on_fold(k, metrics)`` may raise (e.g. optuna pruning) to stop early.
    """
    rows = []
    for k, tr_trips, va_trips in group_folds(cands, n_splits, seed):
        tr = training_rows(cands[cands.trip_id.isin(tr_trips)], stride)
        va_c = cands[cands.trip_id.isin(va_trips) & stride_mask(cands, stride)]
        va_o = obs[obs.trip_id.isin(va_trips) & stride_mask(obs, stride)]
        model, p, info = fit(name, params, tr, seed, n_jobs, slow_cap)
        prob, pred_s = predict_proba(model, va_c)
        m = evaluate_all(va_c, va_o, prob, threshold)
        m.update(info)
        # ms per 1000 observations = (pred_s * 1000) / (n_obs / 1000)
        m.update({"fold": k, "predict_s": pred_s, "val_trips": len(va_trips),
                  "predict_ms_per_1k_obs": 1e6 * pred_s / max(1, len(va_o))})
        rows.append(m)
        if on_fold is not None:
            on_fold(k, m)
    return pd.DataFrame(rows)

"""Emission scores (owner: Person 3).

Hidden state at step t = one candidate arc of observation t (the candidate
table row, so the candidate <-> road-segment correspondence is preserved).

Three emission models (log space, additive constants dropped because Viterbi
only compares states):

* ``classifier``: log max(p, min_prob) where p = P(candidate is reference)
  from the exported model bundle.
* ``gaussian``: -0.5 (d / sigma)^2 - the classic Newson & Krumm (2009)
  distance-only emission; used as a baseline that ignores the classifier.
* ``blend``: w * classifier + (1 - w) * gaussian.
"""
from __future__ import annotations

import numpy as np

VALID_MODES = ("classifier", "gaussian", "blend")


def log_emission(dist_m: np.ndarray, prob: np.ndarray | None, mode: str = "classifier", sigma_m: float = 8.0,
                 min_prob: float = 1e-6, weight: float = 1.0) -> np.ndarray:
    if mode not in VALID_MODES:
        raise ValueError(f"emission mode must be one of {VALID_MODES}")
    gauss = -0.5 * (np.asarray(dist_m, float) / sigma_m) ** 2
    if mode == "gaussian":
        return gauss
    if prob is None:
        raise ValueError("classifier probabilities required for emission mode %r" % mode)
    clf = np.log(np.clip(np.asarray(prob, float), min_prob, 1.0))
    if mode == "classifier":
        return clf
    return weight * clf + (1.0 - weight) * gauss

"""Log-space Viterbi with segment breaks (owner: Person 3).

Recurrence for step s (states j) given step s-1 (states i)::

    delta_s(j) = e_s(j) + max_i [ delta_{s-1}(i) + T_s(i, j) ]
    psi_s(j)   = argmax_i [ delta_{s-1}(i) + T_s(i, j) ]

Break handling: if the transition matrix for a step is ``None`` (forced
break, e.g. a long GPS gap) or every entry of ``delta_s`` is -inf (no
candidate of step s is reachable from any surviving state), the current
segment is closed - back-tracked from its best final state - and a new
segment starts at step s with ``delta_s = e_s``.  The decoder therefore
always returns a state for every step and never crashes on disconnected
road segments.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def viterbi(emissions: Sequence[np.ndarray],
            transitions: Sequence[np.ndarray | None] | Callable[[int], np.ndarray | None]):
    """Return ``(state_index_per_step, segment_start_steps)``.

    ``transitions[s]`` (or ``transitions(s)``) is the (K_{s-1}, K_s) log
    transition matrix into step s; index 0 is ignored.
    """
    T = len(emissions)
    if T == 0:
        return [], []
    get_t = transitions if callable(transitions) else (lambda s: transitions[s])
    path = [0] * T
    starts = [0]
    backs: list[np.ndarray | None] = [None] * T
    delta = np.asarray(emissions[0], float)

    def close(seg_start: int, seg_end: int, last_delta: np.ndarray) -> None:
        j = int(np.argmax(last_delta))
        path[seg_end] = j
        for s in range(seg_end, seg_start, -1):
            j = int(backs[s][j])
            path[s - 1] = j

    for s in range(1, T):
        e = np.asarray(emissions[s], float)
        tm = get_t(s)
        new = None
        if tm is not None:
            scores = delta[:, None] + tm
            arg = np.argmax(scores, axis=0)
            best = scores[arg, np.arange(scores.shape[1])]
            new = best + e
            if not np.isfinite(new).any():
                new = None
        if new is None:
            close(starts[-1], s - 1, delta)
            starts.append(s)
            delta = e
            continue
        backs[s] = arg
        delta = new
    close(starts[-1], T - 1, delta)
    return path, starts

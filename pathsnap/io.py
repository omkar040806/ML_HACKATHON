"""Raw trip loading and validation (owner: Person 1).

Leakage rule enforced here: :func:`load_trip_input` reads ONLY the
``.track``, ``.nodes`` and ``.arcs`` files.  The reference ``.route`` file is
read exclusively by :func:`load_reference_route`, which must only be called
by label construction (training / student-test) and by server-side evaluation.

File formats (verified against every trip in train/ and student_test/):

* ``<id>.track``  whitespace separated: longitude, latitude, timestamp [s]
* ``<id>.nodes``  whitespace separated: longitude, latitude; row index = node_id
* ``<id>.arcs``   two ints: from_node_id, to_node_id; row index = arc_id
* ``<id>.route``  one int per line: chronological reference arc_id sequence
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TRIP_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class TripFormatError(ValueError):
    """Raised when a trip folder is missing files or has malformed content."""


@dataclass
class TripInput:
    """Everything the prediction pipeline is allowed to see for one trip."""

    trip_id: str
    track: np.ndarray            # (n_obs, 3) lon, lat, t  (cleaned, time-sorted)
    nodes: np.ndarray            # (n_nodes, 2) lon, lat
    arcs: np.ndarray             # (n_arcs, 2) int64 from_node, to_node
    raw_track_rows: int = 0      # rows in the file before cleaning
    kept_rows: np.ndarray | None = None  # indices of raw rows kept after cleaning
    issues: list[str] = field(default_factory=list)

    @property
    def n_obs(self) -> int:
        return int(self.track.shape[0])


def list_trips(partition_path: str | Path) -> list[str]:
    """Trip ids (sub-folder names) that contain a ``.track`` file."""
    root = Path(partition_path)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / f"{d.name}.track").exists():
            out.append(d.name)
    return out


def trip_path(partition_path: str | Path, trip_id: str) -> Path:
    """Safe join; rejects ids that could escape the partition folder."""
    if not TRIP_ID_RE.match(trip_id or ""):
        raise TripFormatError(f"Invalid trip id: {trip_id!r}")
    return Path(partition_path) / trip_id


def _load_matrix(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists():
        raise TripFormatError(f"Missing file: {path.name}")
    try:
        arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as exc:  # non-numeric content
        raise TripFormatError(f"Could not parse {path.name}: {exc}") from exc
    if arr.size == 0:
        raise TripFormatError(f"Empty file: {path.name}")
    if arr.shape[1] != ncols:
        raise TripFormatError(f"{path.name}: expected {ncols} columns, found {arr.shape[1]}")
    return arr


def clean_track(track: np.ndarray, drop_duplicate_timestamps: bool = True):
    """Drop non-finite / implausible rows, sort by time, drop duplicate timestamps.

    Returns ``(clean_track, kept_row_indices, issues)``.
    """
    issues: list[str] = []
    idx = np.arange(len(track))
    finite = np.isfinite(track).all(axis=1)
    if (~finite).any():
        issues.append(f"dropped {int((~finite).sum())} non-finite rows")
    plaus = (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    bad = finite & ~plaus
    if bad.any():
        issues.append(f"dropped {int(bad.sum())} rows with implausible coordinates")
    keep = finite & plaus
    track, idx = track[keep], idx[keep]
    order = np.argsort(track[:, 2], kind="stable")
    if (order != np.arange(len(order))).any():
        issues.append("track was not time-sorted; sorted by timestamp")
    track, idx = track[order], idx[order]
    if drop_duplicate_timestamps and len(track) > 1:
        dup = np.concatenate([[False], np.diff(track[:, 2]) == 0])
        if dup.any():
            issues.append(f"dropped {int(dup.sum())} duplicate-timestamp rows")
        track, idx = track[~dup], idx[~dup]
    return track, idx, issues


def load_trip_input(trip_dir: str | Path, drop_duplicate_timestamps: bool = True) -> TripInput:
    """Load track + road network for one trip.  Never touches ``.route``."""
    d = Path(trip_dir)
    tid = d.name
    track_raw = _load_matrix(d / f"{tid}.track", 3, float)
    nodes = _load_matrix(d / f"{tid}.nodes", 2, float)
    arcs = _load_matrix(d / f"{tid}.arcs", 2, np.int64)

    if (arcs < 0).any() or arcs.max() >= len(nodes):
        raise TripFormatError(f"{tid}: arc references a node id outside 0..{len(nodes) - 1}")
    if not np.isfinite(nodes).all():
        raise TripFormatError(f"{tid}: non-finite node coordinates")

    track, kept, issues = clean_track(track_raw, drop_duplicate_timestamps)
    if len(track) < 2:
        raise TripFormatError(f"{tid}: fewer than 2 valid GPS observations")
    for msg in issues:
        log.info("trip %s: %s", tid, msg)
    return TripInput(trip_id=tid, track=track, nodes=nodes, arcs=arcs,
                     raw_track_rows=len(track_raw), kept_rows=kept, issues=issues)


def load_reference_route(trip_dir: str | Path, n_arcs: int | None = None) -> np.ndarray | None:
    """Reference arc sequence, or ``None`` if the trip has no ``.route`` file.

    EVALUATION / LABELLING ONLY - never pass the result to the predictor.
    """
    d = Path(trip_dir)
    path = d / f"{d.name}.route"
    if not path.exists():
        return None
    route = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if n_arcs is not None and route.size and ((route < 0).any() or route.max() >= n_arcs):
        raise TripFormatError(f"{d.name}: route references an arc id outside 0..{n_arcs - 1}")
    return route

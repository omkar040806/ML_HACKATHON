"""Shared synthetic fixtures (owner: Person 1; used by every member's tests).

A 6 x 6 grid city with 100 m blocks and two-way streets.  The vehicle drives
east along row 2, turns north at column 4 and continues to row 5.  GPS points
are sampled every 1 s at ~10 m/s with 3 m Gaussian noise.  Because the
geometry is known exactly, tests can assert correct answers without the real
dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LON0, LAT0 = 10.0, 50.0
BLOCK_M = 100.0
N = 6


def _m_to_lonlat(x, y):
    kx = 6_371_008.8 * np.cos(np.radians(LAT0)) * np.pi / 180
    ky = 6_371_008.8 * np.pi / 180
    return LON0 + np.asarray(x) / kx, LAT0 + np.asarray(y) / ky


def make_grid():
    nodes_xy = np.array([(c * BLOCK_M, r * BLOCK_M) for r in range(N) for c in range(N)], float)
    nid = lambda r, c: r * N + c  # noqa: E731
    arcs = []
    for r in range(N):
        for c in range(N):
            if c + 1 < N:
                arcs += [(nid(r, c), nid(r, c + 1)), (nid(r, c + 1), nid(r, c))]
            if r + 1 < N:
                arcs += [(nid(r, c), nid(r + 1, c)), (nid(r + 1, c), nid(r, c))]
    return nodes_xy, np.array(arcs, dtype=np.int64), nid


def arc_index(arcs, a, b):
    return int(np.nonzero((arcs[:, 0] == a) & (arcs[:, 1] == b))[0][0])


def make_synthetic_trip(root: Path, trip_id: str = "00000001", seed: int = 0, noise_m: float = 3.0,
                        with_route: bool = True, gap: bool = False):
    rng = np.random.default_rng(seed)
    nodes_xy, arcs, nid = make_grid()
    path_nodes = [nid(2, c) for c in range(0, 5)] + [nid(r, 4) for r in range(3, 6)]
    route = [arc_index(arcs, a, b) for a, b in zip(path_nodes[:-1], path_nodes[1:])]
    pts, true_arc = [], []
    for a in route:
        ax, ay = nodes_xy[arcs[a, 0]]
        bx, by = nodes_xy[arcs[a, 1]]
        for s in np.arange(0, BLOCK_M, 10.0):
            f = s / BLOCK_M
            pts.append((ax + f * (bx - ax), ay + f * (by - ay)))
            true_arc.append(a)
    pts = np.array(pts) + rng.normal(0, noise_m, (len(pts), 2))
    t = np.arange(1, len(pts) + 1, dtype=float)
    if gap:
        t[len(t) // 2:] += 120.0
    lon, lat = _m_to_lonlat(pts[:, 0], pts[:, 1])
    nlon, nlat = _m_to_lonlat(nodes_xy[:, 0], nodes_xy[:, 1])
    d = Path(root) / trip_id
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / f"{trip_id}.track", np.column_stack([lon, lat, t]), fmt="%.7f\t%.7f\t%.6f")
    np.savetxt(d / f"{trip_id}.nodes", np.column_stack([nlon, nlat]), fmt="%.7f\t%.7f")
    np.savetxt(d / f"{trip_id}.arcs", arcs, fmt="%d\t%d")
    if with_route:
        np.savetxt(d / f"{trip_id}.route", np.array(route), fmt="%d")
    return {"dir": d, "route": np.array(route), "true_arc": np.array(true_arc), "arcs": arcs,
            "nodes_xy": nodes_xy}


@pytest.fixture
def synthetic_trip(tmp_path):
    return make_synthetic_trip(tmp_path / "part")


@pytest.fixture
def synthetic_partition(tmp_path):
    """A partition folder with 6 synthetic trips (different noise seeds)."""
    root = tmp_path / "part"
    infos = [make_synthetic_trip(root, f"{i:08d}", seed=i, noise_m=3.0 + i) for i in range(6)]
    return root, infos


@pytest.fixture
def cfg():
    from pathsnap.config import load_config
    return load_config()

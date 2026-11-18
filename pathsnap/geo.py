"""Geodesy helpers (owner: Person 1).

Projection choice
-----------------
Trips are located all over the world (longitudes -100 .. 137), so a single
global projected CRS is not usable.  Each trip is projected with a *local
equirectangular* projection centred on the mean of its GPS points:

    x = R * (lon - lon0) * cos(lat0),   y = R * (lat - lat0)

Within a few tens of kilometres of the centre the distance error is well
below 1 % (``tests/test_data_layer.py::test_projection_matches_haversine``
checks it against the haversine formula),
which is far smaller than GPS noise.  It is cheap, needs no extra dependency
and works for every trip.  The EDA script reports each trip's extent so that
the assumption can be verified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class LocalProjection:
    lon0: float
    lat0: float

    @classmethod
    def from_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    @property
    def _kx(self) -> float:
        return EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180.0

    @property
    def _ky(self) -> float:
        return EARTH_RADIUS_M * np.pi / 180.0

    def forward(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        dlon = (lon - self.lon0 + 180.0) % 360.0 - 180.0   # safe across the antimeridian
        return dlon * self._kx, (lat - self.lat0) * self._ky

    def inverse(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x / self._kx + self.lon0, y / self._ky + self.lat0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, 90 = east) of a planar vector."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Absolute smallest difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def point_segment_distance(px, py, ax, ay, bx, by):
    """Vectorised distance from points P to segments AB.

    Returns ``(distance, t, foot_x, foot_y)`` where ``t`` in [0, 1] is the
    clamped position of the perpendicular foot along AB.  Zero-length
    segments are handled (t = 0).
    """
    px, py, ax, ay, bx, by = (np.asarray(v, dtype=float) for v in (px, py, ax, ay, bx, by))
    vx, vy = bx - ax, by - ay
    seg2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * vx + (py - ay) * vy) / seg2
    t = np.where(seg2 > 0, t, 0.0)
    t = np.clip(t, 0.0, 1.0)
    fx, fy = ax + t * vx, ay + t * vy
    return np.hypot(px - fx, py - fy), t, fx, fy

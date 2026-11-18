"""Geometry helpers.

Trips cover a few km, so we project lon/lat into a local equirectangular frame
(metres, origin at the trip's GPS centroid). Over <20 km the distortion of this
projection is well under 0.1 %, far below GPS noise, and it lets us use cheap
planar point-to-segment distances instead of geodesic ones.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius, metres


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * EARTH_R

    @classmethod
    def for_track(cls, track: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(track[:, 0])), float(np.mean(track[:, 1])))

    def forward(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float)
        return np.column_stack([(lonlat[:, 0] - self.lon0) * self.kx,
                                (lonlat[:, 1] - self.lat0) * self.ky])

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised planar projection of points p onto segments a->b (all (n,2)).
    Returns (distance, t) where t in [0,1] is the clipped position along the segment."""
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.linalg.norm(p - proj, axis=1), t


def bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass-style bearing in radians (0 = north, clockwise) from planar deltas."""
    return np.arctan2(dx, dy)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute smallest difference between two angles, in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

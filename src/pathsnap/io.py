"""Raw data loading for PathSnap trips.

Each trip folder <root>/<trip_id>/ holds four header-less whitespace files:
  <id>.track  lon lat t        (GPS observations, ~1 Hz)
  <id>.nodes  lon lat          (road nodes, row index = node_id)
  <id>.arcs   from to          (directed road segments, row index = arc_id)
  <id>.route  arc_id           (reference route, chronological; optional)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (V, 2) lon, lat
    arcs: np.ndarray             # (E, 2) from_node, to_node
    route: np.ndarray | None = None   # (M,) arc ids, None when not supplied
    meta: dict = field(default_factory=dict)


def _read(path: str, dtype=float, ncols: int | None = None) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    if ncols is not None and df.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, found {df.shape[1]}")
    return df.to_numpy().astype(dtype)


def list_trips(root: str) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_trip(root: str, trip_id: str, with_route: bool = True) -> Trip:
    base = os.path.join(root, trip_id, trip_id)
    track = _read(base + ".track", float, 3)
    nodes = _read(base + ".nodes", float, 2)
    arcs = _read(base + ".arcs", int, 2)
    route = None
    if with_route and os.path.exists(base + ".route"):
        route = _read(base + ".route", int, 1).reshape(-1)
    return Trip(trip_id, track, nodes, arcs, route)


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop non-finite rows, implausible coordinates and duplicate / non-increasing
    timestamps. Returns the cleaned track and a report of what was removed."""
    rep = {"raw": len(track)}
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    rep["invalid"] = int((~ok).sum())
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    rep["reordered"] = int((order != np.arange(len(order))).sum())
    track = track[order]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    rep["duplicate_t"] = int((~keep).sum())
    track = track[keep]
    rep["clean"] = len(track)
    return track, rep

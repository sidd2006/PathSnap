"""Turn the reference route (a sequence of arc ids) into a per-GPS-point label.

The .route file lists arcs in travel order but has no timing, and there are far
fewer route arcs than GPS points. We therefore align the two sequences with a
monotone dynamic programme: point i is assigned route position j_i with
j_0 <= j_1 <= ... and the total point-to-arc distance is minimised. This is a
labelling step used ONLY for training and for server-side scoring.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment


def align_route(track_xy: np.ndarray, route: np.ndarray, net) -> tuple[np.ndarray, np.ndarray]:
    """Returns (label_arc_ids, label_dist) with one entry per GPS point.
    Route arcs outside the pruned network get a large distance (never chosen unless forced)."""
    N, M = len(track_xy), len(route)
    D = np.full((N, M), 1e6)
    for j, arc in enumerate(route):
        li = net.local.get(int(arc))
        if li is None:
            continue
        d, _ = point_segment(track_xy, np.repeat(net.a[li:li + 1], N, 0), np.repeat(net.b[li:li + 1], N, 0))
        D[:, j] = d

    # cost[i, j] = D[i, j] + min_{j' <= j} cost[i-1, j']  (prefix-min keeps it O(N*M))
    cost = np.empty_like(D)
    back = np.empty((N, M), dtype=np.int32)
    cost[0] = D[0]
    for i in range(1, N):
        prev = cost[i - 1]
        run = np.minimum.accumulate(prev)
        cost[i] = D[i] + run
        back[i] = _running_argmin(prev, run)
    j = int(np.argmin(cost[-1]))
    path = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = j
        if i:
            j = int(back[i, j])
    return route[path], D[np.arange(N), path]


def _running_argmin(x: np.ndarray, run: np.ndarray) -> np.ndarray:
    # index where the running minimum was attained
    is_new = np.r_[True, x[1:] < run[:-1]]
    idx = np.where(is_new, np.arange(len(x)), 0)
    return np.maximum.accumulate(idx).astype(np.int32)

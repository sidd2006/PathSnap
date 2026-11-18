"""Road-network representation used by candidate generation and the HMM.

The raw networks are large (~90k nodes / ~170k arcs per trip) but a trip only
touches a small corridor, so we keep the arcs whose bounding box lies within
`margin` metres of the GPS bounding box. Arc ids are kept as the ORIGINAL ids so
predictions remain comparable with the .route file.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 track_xy: np.ndarray, margin: float = 400.0, sample_step: float = 8.0):
        self.proj = proj
        node_xy = proj.forward(nodes_lonlat)
        lo, hi = track_xy.min(0) - margin, track_xy.max(0) + margin
        a, b = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
        amin, amax = np.minimum(a, b), np.maximum(a, b)
        keep = np.all((amax >= lo) & (amin <= hi), axis=1)

        self.arc_ids = np.flatnonzero(keep)            # local index -> original arc id
        self.local = {int(g): i for i, g in enumerate(self.arc_ids)}
        self.frm = arcs[keep, 0]
        self.to = arcs[keep, 1]
        self.a = a[keep]
        self.b = b[keep]
        d = self.b - self.a
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing(d[:, 0], d[:, 1])
        self.node_xy = node_xy
        self.nodes_lonlat = nodes_lonlat

        # out-arcs per node (only kept arcs), for Dijkstra and degree features
        self.out_arcs: dict[int, list[int]] = {}
        for i, f in enumerate(self.frm):
            self.out_arcs.setdefault(int(f), []).append(i)
        in_deg: dict[int, int] = {}
        for t in self.to:
            in_deg[int(t)] = in_deg.get(int(t), 0) + 1
        self.out_deg_to = np.array([len(self.out_arcs.get(int(t), ())) for t in self.to])
        self.in_deg_from = np.array([in_deg.get(int(f), 0) for f in self.frm])
        # does the reverse arc (to->from) exist? i.e. two-way street
        pairs = set(zip(self.frm.tolist(), self.to.tolist()))
        self.has_twin = np.array([(t, f) in pairs for f, t in zip(self.frm.tolist(), self.to.tolist())])

        # spatial index: sample every arc every `sample_step` m (incl. endpoints)
        n_s = np.maximum(1, np.ceil(self.length / sample_step).astype(int))
        idx = np.repeat(np.arange(len(self.length)), n_s + 1)
        frac = np.concatenate([np.linspace(0, 1, k + 1) for k in n_s]) if len(n_s) else np.zeros(0)
        pts = self.a[idx] + (self.b[idx] - self.a[idx]) * frac[:, None]
        self._sample_arc = idx
        self._tree = cKDTree(pts) if len(pts) else None
        self.sample_step = sample_step

    def __len__(self):
        return len(self.arc_ids)

    def arcs_near(self, xy: np.ndarray, radius: float) -> list[np.ndarray]:
        """Local arc indices with a sample point within radius + step/2 of each query point."""
        if self._tree is None:
            return [np.zeros(0, int) for _ in xy]
        hits = self._tree.query_ball_point(xy, r=radius + self.sample_step / 2)
        return [np.unique(self._sample_arc[h]) if len(h) else np.zeros(0, int) for h in hits]

    def shortest_from(self, src_node: int, cutoff: float) -> dict[int, float]:
        """Dijkstra over kept arcs from a node, stopping beyond `cutoff` metres."""
        dist = {src_node: 0.0}
        heap = [(0.0, src_node)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for e in self.out_arcs.get(u, ()):
                v = int(self.to[e])
                nd = d + self.length[e]
                if nd < dist.get(v, np.inf) and nd <= cutoff:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def arc_lonlat(self, local_idx: int) -> list[list[float]]:
        """[[lat, lon], [lat, lon]] of an arc, for map drawing."""
        f, t = int(self.frm[local_idx]), int(self.to[local_idx])
        return [[self.nodes_lonlat[f, 1], self.nodes_lonlat[f, 0]],
                [self.nodes_lonlat[t, 1], self.nodes_lonlat[t, 0]]]

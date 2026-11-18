"""Candidate generation and feature engineering.

For every GPS observation we gather the road arcs within `radius` metres
(widening to `fallback_radius` if none are found), keep the `k` nearest, and
describe each (observation, candidate arc) pair with geometric, kinematic,
contextual and topological features. The classifier then scores how likely a
candidate is the reference arc for that observation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import LocalProjection, angle_diff, bearing, point_segment
from .io import Trip, clean_track
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "gauss_prior",
    "head_diff", "head_cos", "speed", "moving", "dt_prev", "dt_next",
    "arc_len", "t_proj", "at_end", "n_cands",
    "ctx_dist_mean", "ctx_dist_max", "ctx_head_diff",
    "out_deg_to", "in_deg_from", "has_twin",
]

LABEL_MAX_DIST = 60.0  # metres; aligned reference arc further than this = unreliable label
CTX = 3        # +-3 observations for contextual distance
HEAD_WIN = 2   # +-2 observations for heading / speed estimate


def kinematics(xy: np.ndarray, t: np.ndarray, w: int = HEAD_WIN):
    """Centred-difference heading (rad) and speed (m/s) at every observation."""
    n = len(xy)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[hi] - xy[lo]
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    return bearing(d[:, 0], d[:, 1]), np.hypot(d[:, 0], d[:, 1]) / dt


class TripContext:
    """Everything derived from one trip's raw files that inference needs."""

    def __init__(self, trip: Trip, radius: float = 50.0, fallback_radius: float = 120.0, k: int = 12):
        self.trip = trip
        self.track, self.clean_report = clean_track(trip.track)
        self.proj = LocalProjection.for_track(self.track)
        self.xy = self.proj.forward(self.track[:, :2])
        self.t = self.track[:, 2]
        self.net = RoadNetwork(trip.nodes, trip.arcs, self.proj, self.xy)
        self.radius, self.fallback_radius, self.k = radius, fallback_radius, k

    def candidates(self) -> pd.DataFrame:
        net, xy, t = self.net, self.xy, self.t
        n = len(xy)
        near = net.arcs_near(xy, self.radius)
        miss = [i for i, c in enumerate(near) if len(c) == 0]
        if miss:
            wide = net.arcs_near(xy[miss], self.fallback_radius)
            for i, c in zip(miss, wide):
                near[i] = c

        obs = np.concatenate([np.full(len(c), i) for i, c in enumerate(near)]).astype(int)
        arc = np.concatenate(near).astype(int) if obs.size else np.zeros(0, int)
        d, tp = point_segment(xy[obs], net.a[arc], net.b[arc])
        df = pd.DataFrame({"obs": obs, "arc_local": arc, "dist": d, "t_proj": tp})
        df = df.sort_values(["obs", "dist"]).groupby("obs", sort=False).head(self.k).reset_index(drop=True)
        obs, arc, d = df["obs"].to_numpy(), df["arc_local"].to_numpy(), df["dist"].to_numpy()

        g = df.groupby("obs")["dist"]
        dmin = g.transform("min").to_numpy()
        df["dist_rank"] = g.rank(method="first").to_numpy() - 1
        df["dist_minus_min"] = d - dmin
        df["dist_ratio"] = d / (dmin + 1.0)
        w = np.exp(-0.5 * (d / 10.0) ** 2)
        df["gauss_prior"] = w / (pd.Series(w).groupby(obs).transform("sum").to_numpy() + 1e-12)

        head, speed = kinematics(xy, t)
        hd = angle_diff(head[obs], net.bearing[arc])
        df["head_diff"] = hd
        df["head_cos"] = np.cos(hd)
        df["speed"] = speed[obs]
        df["moving"] = (speed[obs] > 1.5).astype(float)
        dtp = np.r_[np.nan, np.diff(t)]
        dtn = np.r_[np.diff(t), np.nan]
        df["dt_prev"] = np.nan_to_num(dtp, nan=1.0)[obs]
        df["dt_next"] = np.nan_to_num(dtn, nan=1.0)[obs]

        df["arc_len"] = net.length[arc]
        df["at_end"] = ((df["t_proj"] <= 0) | (df["t_proj"] >= 1)).astype(float)
        df["n_cands"] = df.groupby("obs")["obs"].transform("size")

        # context: how well does this arc explain the neighbouring observations?
        ctx = np.zeros((len(df), 2 * CTX + 1))
        for j, off in enumerate(range(-CTX, CTX + 1)):
            o2 = np.clip(obs + off, 0, n - 1)
            ctx[:, j], _ = point_segment(xy[o2], net.a[arc], net.b[arc])
        df["ctx_dist_mean"] = ctx.mean(1)
        df["ctx_dist_max"] = ctx.max(1)
        lo, hi = np.clip(obs - 5, 0, n - 1), np.clip(obs + 5, 0, n - 1)
        disp = xy[hi] - xy[lo]
        df["ctx_head_diff"] = angle_diff(bearing(disp[:, 0], disp[:, 1]), net.bearing[arc])

        df["out_deg_to"] = net.out_deg_to[arc]
        df["in_deg_from"] = net.in_deg_from[arc]
        df["has_twin"] = net.has_twin[arc].astype(float)
        df["arc_id"] = net.arc_ids[arc]
        df["trip_id"] = self.trip.trip_id
        return df


def build_trip_table(trip: Trip, **kw) -> tuple[pd.DataFrame, TripContext, dict]:
    """Candidate table with a `label` column (1 = reference arc). Training-side only."""
    from .labels import align_route

    ctx = TripContext(trip, **kw)
    df = ctx.candidates()
    lab_arc, lab_dist = align_route(ctx.xy, trip.route, ctx.net)
    df["true_arc"] = lab_arc[df["obs"].to_numpy()]
    df["label"] = (df["arc_id"] == df["true_arc"]).astype(int)
    # reference routes are not error-free: some trips sit >100 m from their route.
    # Observations whose aligned reference arc is far away get flagged so training can skip them.
    df["label_dist"] = lab_dist[df["obs"].to_numpy()]
    covered = df.groupby("obs")["label"].max()
    stats = {
        "trip_id": trip.trip_id, "n_obs": len(ctx.xy), "n_rows": len(df),
        "candidate_miss_rate": float(1 - covered.reindex(range(len(ctx.xy)), fill_value=0).mean()),
        "label_dist_median": float(np.median(lab_dist)),
        "label_dist_p95": float(np.percentile(lab_dist, 95)),
        "unreliable_label_frac": float(np.mean(lab_dist > LABEL_MAX_DIST)),
        **{f"clean_{k}": v for k, v in ctx.clean_report.items()},
    }
    return df, ctx, stats

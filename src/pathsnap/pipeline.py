"""End-to-end inference for one trip: raw files -> candidates -> classifier -> HMM.

This module never reads the .route file; scoring against the reference route is
done separately (see evaluate_trip) so labels cannot leak into predictions.
"""
from __future__ import annotations

import time

import numpy as np

from .features import FEATURES, TripContext
from .hmm import HMMConfig, fill_missing, viterbi
from .io import Trip


def predict_trip(trip: Trip, model, hmm_cfg: HMMConfig, cand_cfg: dict | None = None) -> dict:
    t0 = time.perf_counter()
    ctx = TripContext(trip, **(cand_cfg or {}))
    cands = ctx.candidates()
    t1 = time.perf_counter()
    prob = model.predict_proba(cands[FEATURES].to_numpy())[:, 1]
    t2 = time.perf_counter()

    n = len(ctx.xy)
    # classifier-only: arg-max candidate per observation
    clf = np.full(n, -1, dtype=np.int64)
    best = cands.assign(p=prob).groupby("obs")["p"].idxmax()
    clf[best.index.to_numpy()] = cands.loc[best.to_numpy(), "arc_id"].to_numpy()

    rows, diag = viterbi(cands, prob, ctx.xy, ctx.t, ctx.net, hmm_cfg)
    hmm = np.full(n, -1, dtype=np.int64)
    ok = rows >= 0
    hmm[ok] = cands["arc_id"].to_numpy()[rows[ok]]
    t3 = time.perf_counter()

    return {
        "ctx": ctx, "cands": cands, "prob": prob,
        "pred_classifier": fill_missing(clf), "pred_hmm": fill_missing(hmm),
        "diag": diag,
        "timing_s": {"candidates": t1 - t0, "classifier": t2 - t1, "hmm": t3 - t2, "total": t3 - t0},
    }


def evaluate_trip(out: dict, route: np.ndarray) -> dict:
    """Server-side scoring against the reference route (never shown to the model/UI)."""
    from .labels import align_route
    from .metrics import pointwise_accuracy, route_continuity, route_set_scores

    ctx = out["ctx"]
    true, _ = align_route(ctx.xy, route, ctx.net)
    arcs = ctx.trip.arcs
    res = {}
    for k in ("classifier", "hmm"):
        p = out[f"pred_{k}"]
        res[k] = {"accuracy": pointwise_accuracy(p, true),
                  "route_continuity": route_continuity(p, arcs),
                  **route_set_scores(p, route)}
    return res

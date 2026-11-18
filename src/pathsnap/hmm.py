"""HMM map-matching with Viterbi decoding.

Hidden states   : at observation i, each candidate arc (with its projection point).
Emissions       : the classifier's candidate probability, renormalised over the
                  observation's candidates:  log e = w_e * log(p / sum p).
Transitions     : Newson & Krumm style - compare the network distance between the
                  two projection points with the straight-line GPS displacement:
                  log a = -|d_route - d_gps| / beta_t, beta_t = beta * sqrt(max(dt,1)).
                  Pairs with no path within the search cutoff are invalid (-inf).
Breaks          : if every transition into step i is invalid (disconnected
                  candidates, or a long GPS gap), the chain is restarted at i from
                  emissions alone, so one bad step cannot destroy the whole route.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

NEG = -1e18


@dataclass
class HMMConfig:
    beta: float = 6.0            # metres; tolerance on |route - gps| distance
    emission_weight: float = 1.0
    backward_penalty: float = 5.0  # extra metres charged for moving backwards on an arc
    cutoff_factor: float = 2.0     # Dijkstra cutoff = cutoff_factor * gps_dist + cutoff_extra
    cutoff_extra: float = 150.0
    eps: float = 1e-4

    def to_dict(self):
        return asdict(self)


def _route_dist(net, u, tu, v, tv, sp_cache, cutoff, backward_penalty):
    if u == v:
        delta = (tv - tu) * net.length[u]
        return delta if delta >= 0 else -delta + backward_penalty
    src = int(net.to[u])
    dist = sp_cache.get(src)
    if dist is None or dist[1] < cutoff:
        dist = (net.shortest_from(src, cutoff), cutoff)
        sp_cache[src] = dist
    sp = dist[0].get(int(net.frm[v]))
    if sp is None or sp > cutoff:
        return None
    return (1 - tu) * net.length[u] + sp + tv * net.length[v]


def viterbi(cands: pd.DataFrame, prob: np.ndarray, xy: np.ndarray, t: np.ndarray, net,
            cfg: HMMConfig = HMMConfig()) -> tuple[np.ndarray, dict]:
    """Decode one trip. `cands` must have columns obs, arc_local, t_proj, sorted by obs.
    Returns the chosen row index (into cands) for every observation that has candidates
    (-1 for observations with none) plus decoding diagnostics."""
    n_obs = len(xy)
    cands = cands.reset_index(drop=True)
    groups = cands.groupby("obs").indices
    p = np.asarray(prob, float) + cfg.eps
    obs_with = sorted(groups)
    pa = cands["arc_local"].to_numpy()
    pt = cands["t_proj"].to_numpy()

    chosen = np.full(n_obs, -1, dtype=np.int64)
    breaks = 0
    sp_cache: dict = {}
    prev_rows = prev_score = None
    prev_obs = None
    back: dict = {}       # obs -> backpointers into the previous step (None = chain start)
    bp_prev: dict = {}    # obs -> (previous obs, its candidate rows)
    seg_start = []        # observations where a chain (re)starts

    def emis(rows):
        q = p[rows] / p[rows].sum()
        return cfg.emission_weight * np.log(q)

    def backtrack(last_obs, last_rows, last_score):
        j = int(np.argmax(last_score))
        o = last_obs
        rows = last_rows
        while True:
            chosen[o] = rows[j]
            bp = back.get(o)
            if bp is None:
                break
            j = int(bp[j])
            o, rows = bp_prev[o]
            if o is None:
                break

    for o in obs_with:
        rows = groups[o]
        e = emis(rows)
        if prev_rows is None:
            score = e
            back[o] = None
            bp_prev[o] = (None, None)
            seg_start.append(o)
        else:
            gps = float(np.hypot(*(xy[o] - xy[prev_obs])))
            dt = max(float(t[o] - t[prev_obs]), 1.0)
            beta = cfg.beta * np.sqrt(dt)
            cutoff = cfg.cutoff_factor * gps + cfg.cutoff_extra * np.sqrt(dt)
            A = np.full((len(prev_rows), len(rows)), NEG)
            for a, ru in enumerate(prev_rows):
                for b, rv in enumerate(rows):
                    r = _route_dist(net, pa[ru], pt[ru], pa[rv], pt[rv], sp_cache, cutoff, cfg.backward_penalty)
                    if r is not None:
                        A[a, b] = -abs(r - gps) / beta
            tot = prev_score[:, None] + A
            best = tot.argmax(0)
            bs = tot[best, np.arange(len(rows))]
            if np.all(bs <= NEG / 2):
                # disconnected step: close the current chain and restart here
                backtrack(prev_obs, prev_rows, prev_score)
                breaks += 1
                score = e
                back[o] = None
                bp_prev[o] = (None, None)
                seg_start.append(o)
            else:
                score = bs + e
                back[o] = best
                bp_prev[o] = (prev_obs, prev_rows)
        # keep scores bounded
        score = score - score.max()
        prev_rows, prev_score, prev_obs = rows, score, o
        if len(sp_cache) > 5000:
            sp_cache.clear()

    if prev_rows is not None:
        backtrack(prev_obs, prev_rows, prev_score)
    return chosen, {"breaks": breaks, "segments": len(seg_start), "obs_without_candidates": n_obs - len(obs_with)}


def fill_missing(pred_arc: np.ndarray) -> np.ndarray:
    """Observations without candidates inherit the previous (else next) prediction."""
    s = pd.Series(np.where(pred_arc < 0, np.nan, pred_arc))
    return s.ffill().bfill().fillna(-1).astype(np.int64).to_numpy()

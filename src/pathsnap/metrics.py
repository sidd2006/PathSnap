"""Evaluation metrics for candidate classification and route reconstruction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score


def candidate_metrics(y: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_per_obs(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, obs). Returns trip_id, obs, arc_id, label."""
    tmp = df[["trip_id", "obs", "arc_id", "label"]].copy()
    tmp["p"] = prob
    idx = tmp.groupby(["trip_id", "obs"])["p"].idxmax()
    return tmp.loc[idx].reset_index(drop=True)


def top1_accuracy(df: pd.DataFrame, prob: np.ndarray, n_obs_total: int | None = None) -> float:
    """Observation-level Top-1 accuracy. Observations with no candidate count as errors
    when n_obs_total is given."""
    top = top1_per_obs(df, prob)
    denom = n_obs_total if n_obs_total else len(top)
    return float(top["label"].sum() / denom)


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive prediction changes that move to a connected arc
    (head of previous == tail of next). Staying on the same arc counts as continuous."""
    p = np.asarray(pred_arcs)
    p = p[p >= 0]
    if len(p) < 2:
        return 1.0
    a, b = p[:-1], p[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def route_set_scores(pred_arcs: np.ndarray, route: np.ndarray) -> dict:
    """Sequence-level view: compare the set of traversed arcs with the reference route."""
    P, R = set(int(x) for x in pred_arcs if x >= 0), set(int(x) for x in route)
    inter = len(P & R)
    prec = inter / len(P) if P else 0.0
    rec = inter / len(R) if R else 0.0
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def pointwise_accuracy(pred_arcs: np.ndarray, true_arcs: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred_arcs) == np.asarray(true_arcs)))

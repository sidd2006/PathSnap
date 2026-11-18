"""HMM / Viterbi decoder: parameter tuning and classifier-only vs classifier+HMM comparison.

1. Decoder tuning (training data only): a LightGBM with the tuned parameters is fit on
   68 training trips; the decoder grid (beta x emission_weight) is scored on the other
   12 held-out training trips, so probabilities are out-of-sample.
2. Final comparison on the 10 student-test trips with the final model.

Outputs
  reports/hmm_tuning.csv, reports/hmm_vs_classifier.csv, reports/hmm_summary.csv
  reports/figures/hmm_vs_classifier.png, reports/maps/pred_<trip>.html
  artifacts/hmm_config.json
"""
import itertools
import json
import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402
from pathsnap.features import FEATURES, LABEL_MAX_DIST  # noqa: E402
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.models import SEED  # noqa: E402
from pathsnap.pipeline import evaluate_trip, predict_trip  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

# Viterbi's arg-max is unchanged when every log-score is multiplied by a constant, and the
# score of a path is  w * sum(log e) - sum(|d_route - d_gps|) / beta.  Dividing by w shows that
# only the product beta * w matters, so we fix w = 1 and search beta over a wide log range
# (a first 5x3 grid over beta and w confirmed it: equal products gave identical accuracy).
BETAS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
EMIS_W = [1.0]


def tune():
    meta = json.load(open(os.path.join(config.ARTIFACTS, "final_model_meta.json")))
    tr = pd.read_pickle(os.path.join(config.CACHE, "features_train.pkl"))
    tr = tr[tr.label_dist <= LABEL_MAX_DIST]
    stats = pd.read_csv(os.path.join(config.REPORTS, "candidate_stats_train.csv"), dtype={"trip_id": str})
    # held-out tuning trips: reliable labels, moderate length (keeps the grid affordable)
    ok = stats[(stats.unreliable_label_frac < 0.02) & (stats.n_obs < 3000)]
    hold = ok.sample(12, random_state=SEED).trip_id.tolist()
    fit = tr[~tr.trip_id.isin(hold)]
    m = LGBMClassifier(**meta["params"], subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy())

    rows = []
    trips = [load_trip(config.TRAIN_DIR, t) for t in hold]
    for beta, w in itertools.product(BETAS, EMIS_W):
        cfg = HMMConfig(beta=beta, emission_weight=w)
        accs, conts = [], []
        for trip in trips:
            out = predict_trip(trip, m, cfg, config.CAND_CFG)
            ev = evaluate_trip(out, trip.route)
            accs.append(ev["hmm"]["accuracy"]); conts.append(ev["hmm"]["route_continuity"])
        rows.append({"beta": beta, "emission_weight": w, "accuracy": np.mean(accs), "continuity": np.mean(conts)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(config.REPORTS, "hmm_tuning.csv"), index=False)
    best = res.sort_values(["accuracy", "continuity"], ascending=False).iloc[0]
    cfg = HMMConfig(beta=float(best.beta), emission_weight=float(best.emission_weight))
    with open(os.path.join(config.ARTIFACTS, "hmm_config.json"), "w") as f:
        json.dump({**cfg.to_dict(), "tuned_on": f"12 held-out training trips: {hold}"}, f, indent=2)
    return cfg


def compare(cfg):
    model = joblib.load(os.path.join(config.ARTIFACTS, "final_model.joblib"))
    rows = []
    for tid in list_trips(config.STUDENT_TEST_DIR):
        trip = load_trip(config.STUDENT_TEST_DIR, tid)
        out = predict_trip(trip, model, cfg, config.CAND_CFG)
        ev = evaluate_trip(out, trip.route)
        rows.append({"trip_id": tid, "n_obs": len(out["ctx"].xy),
                     **{f"clf_{k}": v for k, v in ev["classifier"].items()},
                     **{f"hmm_{k}": v for k, v in ev["hmm"].items()},
                     "hmm_breaks": out["diag"]["breaks"], "time_total_s": out["timing_s"]["total"],
                     "time_hmm_s": out["timing_s"]["hmm"]})
        print(rows[-1], flush=True)
        m = trip_map(trip, out["ctx"].track, {
            "Reference route": (trip.route, "#2ca02c", 7),
            "Classifier only (top-1)": (out["pred_classifier"], "#ff7f0e", 3),
            "Classifier + HMM (Viterbi)": (out["pred_hmm"], "#d62728", 4),
        }, corridor_arcs=out["cands"].arc_id.to_numpy(),
            title=f"Trip {tid} | HMM acc {ev['hmm']['accuracy']:.3f} vs classifier {ev['classifier']['accuracy']:.3f}")
        m.save(os.path.join(config.MAPS, f"pred_{tid}.html"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.REPORTS, "hmm_vs_classifier.csv"), index=False)
    w = df.n_obs / df.n_obs.sum()
    summ = pd.DataFrame({
        "method": ["Classifier only", "Classifier + HMM"],
        "pointwise_accuracy_obs_weighted": [(df.clf_accuracy * w).sum(), (df.hmm_accuracy * w).sum()],
        "pointwise_accuracy_trip_mean": [df.clf_accuracy.mean(), df.hmm_accuracy.mean()],
        "route_continuity": [df.clf_route_continuity.mean(), df.hmm_route_continuity.mean()],
        "route_precision": [df.clf_route_precision.mean(), df.hmm_route_precision.mean()],
        "route_recall": [df.clf_route_recall.mean(), df.hmm_route_recall.mean()],
        "route_f1": [df.clf_route_f1.mean(), df.hmm_route_f1.mean()],
        "trips_with_acc_ge_95": [(df.clf_accuracy >= 0.95).sum(), (df.hmm_accuracy >= 0.95).sum()],
    })
    summ.to_csv(os.path.join(config.REPORTS, "hmm_summary.csv"), index=False)
    print(summ.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(df))
    for a, k, t in ((ax[0], "accuracy", "Pointwise accuracy"), (ax[1], "route_continuity", "Route continuity")):
        a.bar(x - 0.2, df[f"clf_{k}"], 0.4, label="classifier only")
        a.bar(x + 0.2, df[f"hmm_{k}"], 0.4, label="classifier + HMM")
        a.set_xticks(x, df.trip_id, rotation=45); a.set_ylim(0.7, 1.0); a.set_title(f"{t} per student-test trip"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURES, "hmm_vs_classifier.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    cfg_path = os.path.join(config.ARTIFACTS, "hmm_config.json")
    if "--skip-tuning" in sys.argv and os.path.exists(cfg_path):
        d = json.load(open(cfg_path)); d.pop("tuned_on", None); cfg = HMMConfig(**d)
    else:
        cfg = tune()
    compare(cfg)

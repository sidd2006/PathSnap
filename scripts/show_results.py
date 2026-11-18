"""Print the key PathSnap results in the terminal (read from reports/, nothing recomputed).

Usage:  python scripts/show_results.py [eda|models|bayes|hmm|trips|eval|all]
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402

R = config.REPORTS
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def title(t):
    print("\n" + "=" * len(t) + f"\n{t}\n" + "=" * len(t))


def eda():
    title("Dataset summary (reports/eda/summary_by_partition.csv)")
    s = pd.read_csv(os.path.join(R, "eda", "summary_by_partition.csv"), index_col=0)
    print(s[["trips", "obs_total", "obs_mean", "length_km_mean", "dt_frac_1s", "dt_max", "nodes_mean", "arcs_mean"]].round(3).T)


def models():
    title("Model comparison: 5-fold GroupKFold CV + student test (reports/model_comparison.csv)")
    c = pd.read_csv(os.path.join(R, "model_comparison.csv")).sort_values("test_top1", ascending=False)
    print(c[["model", "family", "cv_macro_f1", "cv_macro_f1_std", "cv_top1", "test_top1", "test_macro_f1",
             "test_log_loss", "route_continuity", "train_time_s"]].round(4).to_string(index=False))


def bayes():
    title("Bayesian optimisation of LightGBM (reports/bayes_opt_search_space.json, tuned_vs_untuned.csv)")
    sp = json.load(open(os.path.join(R, "bayes_opt_search_space.json")))
    print(f"sampler={sp['sampler']}  trials={sp['n_trials']}  objective={sp['objective']}  best={sp['best_value']:.4f}")
    print("best params:", {k: round(v, 4) if isinstance(v, float) else v for k, v in sp["best_params"].items()})
    print(pd.read_csv(os.path.join(R, "tuned_vs_untuned.csv")).round(4).to_string(index=False))


def hmm():
    title("HMM decoder tuning on 12 held-out training trips (reports/hmm_tuning.csv)")
    print(pd.read_csv(os.path.join(R, "hmm_tuning.csv")).round(4).to_string(index=False))
    title("Classifier only vs classifier + HMM, student test (reports/hmm_summary.csv)")
    print(pd.read_csv(os.path.join(R, "hmm_summary.csv")).round(4).T.to_string(header=False))


def trips():
    title("Per-trip results, student test (reports/hmm_vs_classifier.csv)")
    h = pd.read_csv(os.path.join(R, "hmm_vs_classifier.csv"), dtype={"trip_id": str})
    print(h[["trip_id", "n_obs", "clf_accuracy", "hmm_accuracy", "clf_route_continuity", "hmm_route_continuity",
             "hmm_route_f1", "hmm_breaks", "time_total_s"]].round(4).to_string(index=False))


def evaluator():
    d = os.path.join(R, "evaluator_test")
    if not os.path.exists(os.path.join(d, "summary.json")):
        print("\n(no evaluator run yet: start the API on the evaluator trips and run scripts/07_evaluator_run.py)")
        return
    s = json.load(open(os.path.join(d, "summary.json")))
    title("FINAL: 10 evaluator-test trips through the API (reports/evaluator_test/)")
    print(f"run {s['run_time']}  model {s['model']}  model sha {s['final_model_sha256_16']}  decoder sha {s['hmm_config_sha256_16']}")
    print(f"accuracy  classifier {s['clf_accuracy_obs_weighted']:.4f}  ->  HMM {s['hmm_accuracy_obs_weighted']:.4f}   (observation-weighted)")
    print(f"continuity classifier {s['clf_route_continuity_mean']:.4f}  ->  HMM {s['hmm_route_continuity_mean']:.4f}")
    print(f"route F1  classifier {s['clf_route_f1_mean']:.4f}  ->  HMM {s['hmm_route_f1_mean']:.4f}")
    print(f"trips >= 95% accuracy: {s['trips_hmm_acc_ge_95']}/10   HMM better on {s['trips_where_hmm_beats_classifier']}/10 trips")
    e = pd.read_csv(os.path.join(d, "results.csv"), dtype={"trip_id": str})
    print(e[["trip_id", "n_obs", "clf_accuracy", "hmm_accuracy", "hmm_route_continuity", "hmm_route_f1", "server_time_s"]]
          .round(4).to_string(index=False))


SECTIONS = {"eda": eda, "models": models, "bayes": bayes, "hmm": hmm, "trips": trips, "eval": evaluator}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in SECTIONS.items():
        if which in ("all", name):
            fn()

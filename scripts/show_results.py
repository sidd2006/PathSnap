"""Print the key PathSnap results in the terminal (read from reports/, nothing recomputed).

Usage:  python scripts/show_results.py [eda|models|bayes|hmm|trips|all]
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


SECTIONS = {"eda": eda, "models": models, "bayes": bayes, "hmm": hmm, "trips": trips}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in SECTIONS.items():
        if which in ("all", name):
            fn()

"""Bayesian hyper-parameter optimisation (Optuna TPE) of the selected model: LightGBM.

Objective  : mean observation-level Top-1 accuracy over 3-fold GroupKFold on the
             80 training trips (the quantity the decoder actually consumes: is the
             reference arc ranked first?). Macro F1 and log loss are logged per trial.
Budget     : N_TRIALS trials; each trial fits on a fixed 50 % sample of training
             observations to keep the search affordable. The best configuration is
             then refit on ALL training rows and compared with the untuned model
             using the same 5-fold CV and the same student-test partition.

Outputs
  reports/bayes_opt_trials.csv, reports/bayes_opt_search_space.json
  reports/tuned_vs_untuned.csv, reports/figures/bayes_opt_history.png
  artifacts/final_model.joblib, artifacts/final_model_meta.json
"""
import json
import os
import sys
import time

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402
from pathsnap.features import FEATURES, LABEL_MAX_DIST  # noqa: E402
from pathsnap.metrics import candidate_metrics, top1_accuracy  # noqa: E402
from pathsnap.models import MODEL_SPECS, SEED  # noqa: E402

N_TRIALS = int(os.environ.get("N_TRIALS", 30))
SEARCH_SPACE = {
    "n_estimators": ["int", 100, 900],
    "learning_rate": ["float_log", 0.01, 0.2],
    "num_leaves": ["int", 15, 255],
    "min_child_samples": ["int", 10, 300],
    "subsample": ["float", 0.5, 1.0],
    "colsample_bytree": ["float", 0.5, 1.0],
    "reg_lambda": ["float_log", 1e-3, 10.0],
    "reg_alpha": ["float_log", 1e-3, 10.0],
}


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "int":
            p[k] = trial.suggest_int(k, lo, hi)
        elif kind == "float":
            p[k] = trial.suggest_float(k, lo, hi)
        else:
            p[k] = trial.suggest_float(k, lo, hi, log=True)
    return p


def make(params):
    return LGBMClassifier(**params, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1)


def cv_eval(factory, df, n_splits):
    out = []
    for a, b in GroupKFold(n_splits=n_splits).split(df, groups=df.trip_id):
        m = factory()
        m.fit(df.iloc[a][FEATURES].to_numpy(), df.iloc[a].label.to_numpy())
        p = m.predict_proba(df.iloc[b][FEATURES].to_numpy())[:, 1]
        out.append({**candidate_metrics(df.iloc[b].label.to_numpy(), p), "top1": top1_accuracy(df.iloc[b], p)})
    return pd.DataFrame(out)


def full_eval(name, factory, tr, te, n_obs_test):
    cv = cv_eval(factory, tr, 5)
    m = factory()
    t0 = time.perf_counter(); m.fit(tr[FEATURES].to_numpy(), tr.label.to_numpy()); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict_proba(te[FEATURES].to_numpy())[:, 1]; inf = time.perf_counter() - t0
    cm = candidate_metrics(te.label.to_numpy(), p)
    return m, {"model": name, "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
               "cv_top1": cv.top1.mean(), "test_top1": top1_accuracy(te, p, n_obs_test),
               "test_macro_f1": cm["macro_f1"], "test_log_loss": cm["log_loss"],
               "train_time_s": fit_s, "inference_ms_per_1k_rows": inf / len(te) * 1e6}


def main():
    tr = pd.read_pickle(os.path.join(config.CACHE, "features_train.pkl"))
    tr = tr[tr.label_dist <= LABEL_MAX_DIST].reset_index(drop=True)
    te = pd.read_pickle(os.path.join(config.CACHE, "features_student_test.pkl"))
    n_obs_test = int(pd.read_csv(os.path.join(config.REPORTS, "candidate_stats_student_test.csv")).n_obs.sum())

    keys = tr[["trip_id", "obs"]].drop_duplicates().sample(frac=0.5, random_state=SEED)
    sub = tr.merge(keys, on=["trip_id", "obs"])

    def objective(trial):
        params = suggest(trial)
        t0 = time.perf_counter()
        cv = cv_eval(lambda: make(params), sub, 3)
        trial.set_user_attr("cv_macro_f1", float(cv.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(cv.log_loss.mean()))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(cv.top1.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                study_name="pathsnap_lgbm")
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(config.REPORTS, "bayes_opt_trials.csv"), index=False)
    with open(os.path.join(config.REPORTS, "bayes_opt_search_space.json"), "w") as f:
        json.dump({"sampler": "TPE (Optuna)", "n_trials": N_TRIALS, "objective": "mean 3-fold GroupKFold Top-1",
                   "search_space": SEARCH_SPACE, "search_seconds": search_s,
                   "best_value": study.best_value, "best_params": study.best_params}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = trials["value"].to_numpy()
    ax.plot(vals, "o", alpha=0.5, label="trial")
    ax.plot(np.maximum.accumulate(vals), "-", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 (3-fold, 50% sample)"); ax.legend()
    ax.set_title("Bayesian optimisation history (LightGBM, TPE)")
    fig.tight_layout(); fig.savefig(os.path.join(config.FIGURES, "bayes_opt_history.png"), dpi=120); plt.close(fig)

    best = study.best_params
    untuned_factory = MODEL_SPECS["LightGBM"][1]
    _, r_un = full_eval("LightGBM (untuned)", untuned_factory, tr, te, n_obs_test)
    m_best, r_tu = full_eval("LightGBM (tuned)", lambda: make(best), tr, te, n_obs_test)
    cmp = pd.DataFrame([r_un, r_tu])
    cmp.to_csv(os.path.join(config.REPORTS, "tuned_vs_untuned.csv"), index=False)
    print(cmp.to_string())

    joblib.dump(m_best, os.path.join(config.ARTIFACTS, "final_model.joblib"))
    with open(os.path.join(config.ARTIFACTS, "final_model_meta.json"), "w") as f:
        json.dump({"model": "LightGBM (Bayesian-tuned)", "params": best, "features": FEATURES,
                   "candidate_config": config.CAND_CFG, "label_max_dist": LABEL_MAX_DIST, "seed": SEED,
                   "metrics": r_tu}, f, indent=2)


if __name__ == "__main__":
    main()

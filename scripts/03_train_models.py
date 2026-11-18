"""Train and compare the candidate classifiers.

Protocol
  * 5-fold GroupKFold on the 80 training trips (groups = trip) -> CV metrics
  * refit on all 80 training trips -> evaluate on the 10 student-test trips
  * training rows whose aligned reference arc is > LABEL_MAX_DIST away are skipped
    (unreliable labels); evaluation on student_test uses every observation.

Outputs
  reports/model_comparison.csv, reports/cv_folds.csv, reports/experiment_log.jsonl
  reports/figures/cmp_*.png, cm_<model>.png, mlp_training_curve.png
  artifacts/models/<model>.joblib
"""
import json
import os
import sys
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402
from pathsnap.features import FEATURES, LABEL_MAX_DIST  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_continuity, top1_accuracy, top1_per_obs  # noqa: E402
from pathsnap.models import MODEL_SPECS, SEED  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402

MODELS_DIR = os.path.join(config.ARTIFACTS, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_tables():
    tr = pd.read_pickle(os.path.join(config.CACHE, "features_train.pkl"))
    te = pd.read_pickle(os.path.join(config.CACHE, "features_student_test.pkl"))
    tr = tr[tr.label_dist <= LABEL_MAX_DIST].reset_index(drop=True)
    return tr, te


def cap_rows(df, cap, rng):
    """Subsample whole observations (not single rows) so candidate groups stay intact."""
    if cap is None or len(df) <= cap:
        return df
    keys = df[["trip_id", "obs"]].drop_duplicates()
    frac = cap / len(df)
    keep = keys.sample(frac=frac, random_state=rng)
    return df.merge(keep, on=["trip_id", "obs"])


def classifier_continuity(te, prob):
    top = top1_per_obs(te, prob)
    vals = []
    for tid, g in top.groupby("trip_id"):
        arcs = load_trip(config.STUDENT_TEST_DIR, tid, with_route=False).arcs
        vals.append(route_continuity(g.sort_values("obs").arc_id.to_numpy(), arcs))
    return float(np.mean(vals))


def run_model(name, tr, te, n_obs_test, log):
    family, factory, cfg_str, cap = MODEL_SPECS[name]
    gkf = GroupKFold(n_splits=5)
    folds = []
    for k, (a, b) in enumerate(gkf.split(tr, groups=tr.trip_id)):
        fit_df = cap_rows(tr.iloc[a], cap, SEED + k)
        val = tr.iloc[b]
        m = factory()
        t0 = time.perf_counter()
        m.fit(fit_df[FEATURES].to_numpy(), fit_df.label.to_numpy())
        fit_s = time.perf_counter() - t0
        p = m.predict_proba(val[FEATURES].to_numpy())[:, 1]
        r = {"model": name, "fold": k, "fit_s": fit_s, **candidate_metrics(val.label.to_numpy(), p),
             "top1": top1_accuracy(val, p)}
        folds.append(r)
        print(f"  {name} fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1']:.4f} ({fit_s:.1f}s)", flush=True)
    cv = pd.DataFrame(folds)

    fit_df = cap_rows(tr, cap, SEED)
    m = factory()
    t0 = time.perf_counter()
    m.fit(fit_df[FEATURES].to_numpy(), fit_df.label.to_numpy())
    train_s = time.perf_counter() - t0
    X = te[FEATURES].to_numpy()
    t0 = time.perf_counter()
    p = m.predict_proba(X)[:, 1]
    inf_s = time.perf_counter() - t0
    cm = candidate_metrics(te.label.to_numpy(), p)
    ptr = m.predict_proba(fit_df[FEATURES].to_numpy())[:, 1]
    row = {
        "model": name, "family": family, "config": cfg_str,
        "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
        "cv_top1": cv.top1.mean(), "cv_log_loss": cv.log_loss.mean(),
        "train_macro_f1": candidate_metrics(fit_df.label.to_numpy(), ptr)["macro_f1"],
        "test_top1": top1_accuracy(te, p, n_obs_test),
        "test_macro_f1": cm["macro_f1"], "test_precision_macro": cm["precision_macro"],
        "test_recall_macro": cm["recall_macro"], "test_accuracy": cm["accuracy"],
        "test_log_loss": cm["log_loss"], "route_continuity": classifier_continuity(te, p),
        "train_time_s": train_s, "inference_ms_per_1k_rows": inf_s / len(X) * 1e6,
        "train_rows": len(fit_df),
    }
    joblib.dump(m, os.path.join(MODELS_DIR, f"{name}.joblib"))

    cmx = confusion_matrix(te.label, (p >= 0.5).astype(int))
    disp = ConfusionMatrixDisplay(cmx, display_labels=["not ref arc", "ref arc"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"{name} - student_test candidates"); plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES, f"cm_{name}.png"), dpi=110); plt.close()

    if name == "MLP":
        mlp = m[-1]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(mlp.loss_curve_, label="training log-loss", color="tab:blue")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
        ax2 = ax1.twinx()
        ax2.plot(mlp.validation_scores_, label="validation accuracy (10% hold-out)", color="tab:orange")
        ax2.set_ylabel("validation accuracy")
        fig.legend(loc="upper center"); fig.tight_layout()
        fig.savefig(os.path.join(config.FIGURES, "mlp_training_curve.png"), dpi=120); plt.close(fig)

    log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, **row}) + "\n")
    return row, cv


def comparison_plots(cmp, cv):
    F = config.FIGURES
    order = cmp.sort_values("cv_macro_f1").model
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [cv[cv.model == m].macro_f1 for m in order]
    ax.boxplot(data, tick_labels=list(order), orientation="horizontal")
    ax.set_xlabel("Macro F1 (5-fold GroupKFold, train)"); ax.set_title("Cross-validation Macro F1 by model")
    fig.tight_layout(); fig.savefig(os.path.join(F, "cmp_cv_macro_f1.png"), dpi=120); plt.close(fig)

    c = cmp.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(c))
    ax.barh(y - 0.2, c.test_top1, 0.4, label="Top-1 accuracy")
    ax.barh(y + 0.2, c.test_macro_f1, 0.4, label="Macro F1")
    ax.set_yticks(y, c.index); ax.set_xlim(0.5, 1.0); ax.legend(); ax.set_title("Student-test performance")
    fig.tight_layout(); fig.savefig(os.path.join(F, "cmp_student_test.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(c.index, c.test_log_loss); ax.set_xscale("log"); ax.set_xlabel("log loss (student_test, log scale)")
    ax.set_title("Log loss by model (lower is better)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "cmp_log_loss.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(c.inference_ms_per_1k_rows, c.test_top1)
    for n, r in c.iterrows():
        ax.annotate(n, (r.inference_ms_per_1k_rows, r.test_top1), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("inference time (ms per 1k candidate rows)"); ax.set_ylabel("student-test Top-1")
    ax.set_title("Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(os.path.join(F, "cmp_speed_vs_accuracy.png"), dpi=120); plt.close(fig)


def main():
    only = sys.argv[1:] or list(MODEL_SPECS)
    tr, te = load_tables()
    n_obs_test = int(pd.read_csv(os.path.join(config.REPORTS, "candidate_stats_student_test.csv")).n_obs.sum())
    print(f"train rows {len(tr)}  test rows {len(te)}  test obs {n_obs_test}")
    rows, cvs = [], []
    with open(os.path.join(config.REPORTS, "experiment_log.jsonl"), "a") as log:
        for name in only:
            print(f"== {name}", flush=True)
            r, cv = run_model(name, tr, te, n_obs_test, log)
            rows.append(r); cvs.append(cv)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    cmp = pd.DataFrame(rows); cv = pd.concat(cvs)
    cmp_path, cv_path = os.path.join(config.REPORTS, "model_comparison.csv"), os.path.join(config.REPORTS, "cv_folds.csv")
    if sys.argv[1:] and os.path.exists(cmp_path):   # partial rerun: merge with previous results
        old = pd.read_csv(cmp_path); cmp = pd.concat([old[~old.model.isin(only)], cmp])
        oldcv = pd.read_csv(cv_path); cv = pd.concat([oldcv[~oldcv.model.isin(only)], cv])
    cmp.to_csv(cmp_path, index=False); cv.to_csv(cv_path, index=False)
    comparison_plots(cmp, cv)
    print(cmp[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_log_loss", "route_continuity"]].to_string())


if __name__ == "__main__":
    main()

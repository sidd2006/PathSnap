# Team and work split

| Member | GitHub | Area owned | Files |
|---|---|---|---|
| Shryes | [@Shryes19](https://github.com/Shryes19) | Data loading, cleaning, EDA, EDA visualisations (incl. Folium), results summary | `src/pathsnap/io.py`, `src/pathsnap/geo.py`, `scripts/01_eda.py`, `scripts/06_results_summary.py`, `reports/eda/`, `reports/figures/eda_*`, `reports/RESULTS.md` |
| Shubham | [@Shubhambhat06](https://github.com/Shubhambhat06) | Road-network model, candidate generation, route-to-observation labelling, feature engineering | `src/pathsnap/network.py`, `src/pathsnap/labels.py`, `src/pathsnap/features.py`, `scripts/02_build_features.py`, `reports/candidate_stats_*` |
| Siddhant Anil | [@sirex750](https://github.com/sirex750) | Model zoo (7 models), cross-validation, metrics, model comparison, Bayesian optimisation | `src/pathsnap/models.py`, `src/pathsnap/metrics.py`, `scripts/03_train_models.py`, `scripts/04_bayes_opt.py`, `reports/model_comparison.csv`, `reports/bayes_opt_*`, `artifacts/final_model*` |
| Siddhant Khurdiya | [@sidd2006](https://github.com/sidd2006) | HMM/Viterbi decoder, inference pipeline, prediction maps, FastAPI backend (Laptop 1), Streamlit UI (Laptop 2), deployment, README | `src/pathsnap/hmm.py`, `src/pathsnap/pipeline.py`, `src/pathsnap/viz.py`, `scripts/05_hmm_eval.py`, `api/`, `ui/`, `README.md` |

## Branches

Each member's work lives on their own branch and was merged into `main` in pipeline order:

| Branch | Owner |
|---|---|
| `shryes/data-eda` | Shryes |
| `shubham/features` | Shubham |
| `siddhant-anil/models` | Siddhant Anil |
| `siddhant-k/hmm-api-ui` | Siddhant Khurdiya |

## What each person must be able to explain in the viva

**Shryes – data & EDA**
- File formats (`.track/.nodes/.arcs/.route`), row-index ids, that every trip has its own road network.
- Cleaning rules in `clean_track` and the missing-value table (`reports/eda/missing_values.csv`).
- Why we project to a local equirectangular frame (metres) instead of using degrees.
- Sampling intervals, the large-gap trips, network size/connectivity figures, the EDA Folium map.

**Shubham – candidates & features**
- Corridor pruning and the KD-tree over sampled arc points (`RoadNetwork.arcs_near`).
- Candidate radius 50 m → fallback 120 m, top-12; candidate miss rate per partition.
- The monotone DP that aligns `.route` arcs to GPS points (`labels.align_route`), and why labels > 60 m away are dropped from training (trips 33 and 54).
- Every feature in `FEATURES` and why it helps (distance, heading vs arc bearing for direction, context window, topology).

**Siddhant Anil – models & tuning**
- The 7 models, their families and configs (`models.py`), GroupKFold by trip (no trip leakage).
- Metric definitions: candidate macro-F1/log-loss vs observation Top-1, route continuity.
- Why LightGBM was selected (CV Top-1/F1, log loss, stability across folds, speed).
- Optuna TPE search space, objective (3-fold CV Top-1), trial count, tuned vs untuned table.

**Siddhant Khurdiya – HMM, API, UI**
- Hidden states = candidate arcs; emissions = renormalised classifier probabilities; transitions = |network distance − GPS distance| / β·√Δt.
- Dijkstra with cutoff, invalid transitions, chain restart on disconnected steps / gaps, filling observations without candidates.
- Decoder tuning on held-out training trips; classifier-only vs HMM results.
- API endpoints, why routes are read only after prediction and never returned, UI ↔ API flow on two laptops.

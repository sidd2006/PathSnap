# PathSnap – Map-matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (PES University, RR campus).
PathSnap matches noisy ~1 Hz GPS tracks to a road network: a classifier scores
candidate road arcs for every GPS fix, and an HMM with Viterbi decoding turns those
scores into a connected route. A FastAPI service (Laptop 1) runs the pipeline and a
Streamlit UI (Laptop 2) lets the evaluator pick a trip and view the result.

Team and ownership: see [TEAM.md](TEAM.md).

## Repository layout

```
src/pathsnap/        core library
  io.py              raw file loading + GPS cleaning
  geo.py             local metric projection, point-to-segment geometry
  network.py         per-trip road network, spatial index, Dijkstra
  labels.py          monotone DP aligning the reference route to GPS fixes (training/scoring only)
  features.py        candidate generation + feature engineering
  models.py          the 7 candidate classifiers compared
  metrics.py         macro-F1, log loss, Top-1, route continuity, route-set scores
  hmm.py             HMM / Viterbi decoder
  pipeline.py        raw trip -> candidates -> classifier -> HMM (+ separate server-side scoring)
  viz.py             Folium maps of matched trips
scripts/
  01_eda.py          dataset inventory, missing values, gaps, network connectivity, plots, EDA Folium map
  02_build_features.py  candidate/feature tables for train + student_test
  03_train_models.py    5-fold GroupKFold + student-test evaluation of all models
  04_bayes_opt.py       Optuna (TPE) tuning of the selected model, tuned vs untuned
  05_hmm_eval.py        decoder tuning, classifier-only vs classifier+HMM, prediction maps
api/main.py          FastAPI backend (Laptop 1)
ui/app.py            Streamlit evaluator UI (Laptop 2)
artifacts/           final_model.joblib, final_model_meta.json, hmm_config.json
reports/             CSV results, experiment log, figures/, maps/ (interactive Folium HTML), eda/
```

## Setup

Python 3.11 recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Put the organiser data under `data/` (not committed):

```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/student_test/<trip_id>/...
```

## Reproducing the experiments

```bash
python scripts/01_eda.py              # EDA tables, figures, Folium map
python scripts/02_build_features.py   # candidate tables -> cache/
python scripts/03_train_models.py     # 7 models, CV + student-test -> reports/model_comparison.csv
python scripts/04_bayes_opt.py        # Bayesian optimisation -> artifacts/final_model.joblib
python scripts/05_hmm_eval.py         # HMM tuning + comparison -> artifacts/hmm_config.json, reports/maps/
```

Seeds are fixed (`SEED = 42`). Every run appends to `reports/experiment_log.jsonl`.

## Deployment (two laptops)

**Laptop 1 – model + API**

```bash
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_trips      # folder with the 10 evaluation trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`GET /trips` lists trip ids, `POST /predict {"trip_id": "..."}` runs the whole pipeline and
returns the predicted route, nearby road geometry and server-side scores. Each request is
logged in the terminal (trip, accuracy, continuity, time) – show this window to the evaluator.
Allow port 8000 through the Windows firewall and find the LAN IP with `ipconfig`.

**Laptop 2 – UI**

```bash
set PATHSNAP_API=http://<laptop-1-ip>:8000
streamlit run ui/app.py
```

## Leakage safeguards

- Trips never cross partitions; CV uses `GroupKFold` grouped by trip.
- All learned choices (features, model, hyper-parameters, decoder β / emission weight) use the
  80 training trips and the 10 student-test trips only.
- `pipeline.predict_trip` never reads `.route`. The API loads the reference route only after
  prediction, uses it only to compute the score, and never sends it to the UI.
- No trip ids or predictions are hard-coded anywhere.

## Results

See `reports/RESULTS.md` (generated from the CSVs in `reports/`).

## AI-assistance disclosure

Parts of this codebase were written with the help of an AI coding assistant (Claude);
commits containing AI-assisted code carry a `Co-Authored-By: Claude` trailer. All experiments
were actually run and every number in `reports/` comes from those runs. Each team member has
reviewed the part listed for them in TEAM.md and is responsible for explaining it.

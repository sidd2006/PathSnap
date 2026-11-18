# PathSnap team handoff and crash course

**UE24CS352A Machine Learning · PES University · 18 November 2026**

PathSnap reconstructs a driven route from noisy GPS observations and a directed road network. It scores nearby road segments with LightGBM, then uses a Hidden Markov Model (HMM) and Viterbi decoding to select a plausible sequence. On the organiser’s 10 evaluator trips, pointwise accuracy improved from **91.25% to 95.89%** after decoding. The evaluator procedure, reports, and README update were [pushed to main as commit 9ad8bf3](https://github.com/sidd2006/PathSnap/commit/9ad8bf3f1e5cc2d95948ae03679ae198959b20d3).

## 1. What was done, from start to finish

| Stage | Work completed | Where to look in the repository |
|---|---|---|
| Project setup | Created the Python package layout, configuration, dependencies, Git exclusions, and team ownership guide. | `README.md`, `TEAM.md`, `requirements.txt`, `src/pathsnap/config.py` |
| Data and exploration | Loaded raw trip files, removed invalid fixes and duplicate timestamps, sorted fixes by time, and projected lat/lon to local metres. Checked sampling, gaps, road connectivity, and reference quality. Produced plots and an interactive EDA map. | `scripts/01_eda.py`, `reports/eda/`, `io.py`, `geo.py` |
| Candidates, labels, and features | Pruned the graph near the trace, indexed sampled road geometry with a KD-tree, generated candidates, aligned reference arcs to observations, and built 21 features per observation–arc pair. | `scripts/02_build_features.py`, `network.py`, `labels.py`, `features.py` |
| Model comparison | Compared seven classifiers using five-fold GroupKFold grouped by trip, then refit and measured performance on the student-test split. Recorded fold results, metrics, training times, and plots. | `scripts/03_train_models.py`, `reports/model_comparison.csv`, `reports/figures/` |
| LightGBM tuning | Used Optuna TPE for 30 trials over eight hyperparameters. The search objective was mean Top-1 over three grouped CV folds on training data. Refit the selected model and saved the model and metadata. | `scripts/04_bayes_opt.py`, `reports/bayes_opt_trials.csv`, `reports/tuned_vs_untuned.csv`, `artifacts/final_model.joblib` |
| Sequence decoding | Implemented HMM/Viterbi with bounded Dijkstra transitions, handling for missing candidates and disconnected steps, and separate scoring. Tuned the decoder on 12 held-out training trips with a classifier fit without those trips. The selected β was 1.0. | `scripts/05_hmm_eval.py`, `hmm.py`, `pipeline.py`, `artifacts/hmm_config.json` |
| Demo and documentation | Built the FastAPI backend and Streamlit front end with a Folium map, added an API smoke test, published student-test comparison maps, and added a report generator and terminal stats command. | `api/main.py`, `ui/app.py`, `scripts/api_smoke_test.py`, `scripts/06_results_summary.py`, `scripts/show_results.py`, `reports/maps/` |
| Windows setup | Cloned the repository over Windows OpenSSH after the HTTPS attempt stalled on authentication. Created a project-local Python 3.13.2 virtual environment, installed dependencies, passed `pip check`, opened the stats terminal, and exercised inference on a synthetic GPS trace. The README originally specified Python 3.11. | Local `repo/.venv`, `PathSnap-Setup.md`, launchers, and `installed-requirements.txt` |
| Final evaluator run | Validated and extracted the supplied ZIP into Git-ignored `data/evaluator_test`. Scored all 10 trips with the existing model and decoder. The evaluator script records per-trip results, an aggregate summary, input and artifact hashes, and package versions. | `scripts/evaluate_dataset.py`, `reports/evaluator_test/` |
| Evaluator demo and push | Started the local API and UI with evaluator data, checked all 10 trips were listed, matched one live API prediction to its saved accuracy, and opened an evaluator results terminal. Pushed the reproducible script, reports, and README update to `main` with Git author `sirex750 <119781835+SiRex750@users.noreply.github.com>`. Raw evaluator files stayed on the laptop. | Commit `9ad8bf3`, local launchers and reports |

## 2. What the experiments show

The training partition had **80 trips and 208,417 observations**. The team used **10 student-test trips and 19,160 observations** for development comparisons. The final evaluator run used **10 trips and 19,674 cleaned observations**. All comparisons below use the tuned classifier for both branches of the pipeline.

The team also ran all 10 evaluator trips through the deployed API. The API and offline runs agree on pointwise accuracy, route continuity, and route F1. The API run took 12.97 seconds of total server time and added prediction maps, a per-trip CSV, and a JSON summary. Both runs are recorded in `reports/evaluator_test/`.

| Metric | Student test, classifier | Student test, + HMM | Evaluator, classifier | Evaluator, + HMM |
|---|---:|---:|---:|---:|
| Pointwise accuracy, weighted by observations | 90.02% | 94.31% | 91.25% | **95.89%** |
| Pointwise accuracy, average across trips | 91.61% | 94.93% | 93.57% | **95.75%** |
| Route continuity, average across trips | 94.42% | 97.89% | 93.40% | **96.43%** |
| Route F1, average across trips | 90.37% | 93.47% | 87.71% | **90.91%** |
| Trips with at least 95% pointwise accuracy | 3/10 | 6/10 | 5/10 | **7/10** |

The evaluator HMM improved weighted pointwise accuracy by **4.65 percentage points** and improved it on nine of ten trips. Trip `00000015` rose from **83.10% to 94.88%** across 6,854 points and contributed heavily to the weighted improvement. Trip `00000017` fell from **93.91% to 93.09%**, while its route continuity improved from **85.28% to 86.99%**. Trip `00000030` had the lowest final pointwise accuracy at **88.69%**. The per-trip report shows roughly **0.27–1.64 seconds** for candidate generation, classification, and decoding in the saved run. This timing excludes file loading, reference scoring, the HTTP round trip, and rendering.

| Evaluator trip | GPS observations | LightGBM accuracy | + HMM accuracy | + HMM route F1 |
|---|---:|---:|---:|---:|
| 00000015 | 6,854 | 83.10% | 94.88% | 88.67% |
| 00000017 | 1,346 | 93.91% | 93.09% | 84.86% |
| 00000023 | 458 | 94.32% | 95.85% | 88.14% |
| 00000030 | 654 | 83.49% | 88.69% | 80.94% |
| 00000049 | 2,642 | 99.28% | 99.62% | 98.01% |
| 00000059 | 1,031 | 96.41% | 96.61% | 86.14% |
| 00000067 | 916 | 96.40% | 96.51% | 94.19% |
| 00000073 | 1,144 | 98.60% | 98.69% | 97.64% |
| 00000087 | 1,399 | 96.00% | 98.43% | 96.21% |
| 00000094 | 3,230 | 94.21% | 95.14% | 94.30% |

Sources in the repository include `reports/RESULTS.md`, `reports/hmm_summary.csv`, and `reports/evaluator_test/{summary.csv,per_trip.csv,provenance.json}`.

## 3. Crash course for the team

**The 30-second pitch.** “GPS can drift onto a parallel road or jump near an intersection. PathSnap gives LightGBM a short list of plausible road segments for each GPS fix. An HMM then combines those scores with how the roads connect and how far the vehicle appears to move. That raises accuracy from 91.25% to 95.89% on the organiser’s evaluator trips.”

### Follow one GPS fix through the system

1. **Read and clean.** A `.track` row contains longitude, latitude, and timestamp. A `.nodes` row holds one intersection or road geometry vertex. Each `.arcs` row specifies the from-node and to-node of a directed road segment. A `.route` file lists the reference arc IDs in travel order. Road node and arc IDs refer to row indices within that trip’s own network.
2. **Find plausible roads.** Project coordinates to metres. Prune the network to a corridor and use a KD-tree over sampled road geometry for nearby lookup. Try 50 m, widen to 120 m if no candidates turn up, and retain up to 12 candidates. The reference arc was present for about 99.8% of student-test fixes. Candidate retrieval sets a ceiling for a classifier that must choose among those candidates.
3. **Describe each candidate.** Features capture distance and distance rank, agreement between GPS heading and road direction, speed and time gaps, distance to adjacent fixes, position along the arc, arc length, and network topology. Distance helps choose between nearby roads. Heading helps distinguish the two directions of a divided street. Context can expose a segment that looks plausible at one fix but poorly explains its neighbors.
4. **Score roads.** The classifier learns a binary task for each GPS-fix/candidate pair. During development, the team assigned targets by aligning the ordered reference arcs to GPS observations with a monotone dynamic program. This preserves travel order and minimizes total point-to-segment distance. Labels more than 60 m from their aligned road were filtered out of training. The classifier-only prediction takes the highest-scoring arc at each fix.
5. **Decode the trip.** HMM states are candidate arcs. Emissions come from normalized classifier scores. A transition scores how well shortest road-network travel distance agrees with GPS displacement, with a larger tolerance as the time gap grows. The decoder uses bounded Dijkstra searches to find feasible paths. Viterbi uses dynamic programming to recover the highest-scoring sequence. Where no feasible transition exists it can restart the chain, so output continuity is measured rather than assumed.
6. **Score and display.** The API loads the reference route after inference, aligns it to the observations, and calculates metrics. The UI receives predictions and metrics for display, plus GPS and road geometry. The API keeps the reference route server-side.

### Know the seven models

This table uses the original student-test model comparison. The evaluator comparison uses the tuned LightGBM model and its HMM decoder.

| Classifier | How it makes a prediction | Student-test Top-1 |
|---|---|---:|
| LightGBM | Adds trees in sequence to improve the ensemble’s predictions. Learns interactions between features like distance and heading. | 90.01% |
| Random Forest | Averages 150 trees built with randomized samples and feature choices. Strong results from averaging many different trees. | 89.48% |
| MLP | Learns nonlinear feature combinations through two hidden layers with 64 and 32 units. Uses scaled inputs and early stopping. | 88.83% |
| Decision Tree | Applies a sequence of feature thresholds to reach a leaf with a candidate score. The comparison limits depth to 12. | 88.80% |
| Logistic Regression | Weighs scaled features, adds them up, and maps the result to a probability. Provides a simple regularized baseline. | 88.52% |
| KNN | Predicts from the 25 closest labelled training examples in feature space, weighted by distance. The comparison used a training sample to manage prediction cost. | 87.94% |
| Gaussian Naive Bayes | Uses a Gaussian distribution per feature and class, assuming independence given the class. Related features such as distance and distance rank challenge that assumption. | 85.72% |

LightGBM led the student-test comparison in Top-1 and macro F1. Random Forest was close in grouped CV and had slightly better untuned student-test log loss, but took about **99 s** to fit versus **23 s** for LightGBM in that comparison. Tuning LightGBM raised student-test Top-1 only slightly (**90.01% to 90.02%**) and reduced log loss from **0.0696 to 0.0662**. The decoder uses those probabilities, so log loss matters. Optuna’s TPE sampler uses previous trials to propose promising hyperparameter settings. Viterbi is the decoding algorithm used by the HMM.

### Read the results correctly

| Term | Meaning in PathSnap |
|---|---|
| GroupKFold | Keeps all observations and candidates from a given trip together in the same fold so highly related samples cannot cross the fit/validation boundary. |
| Top-1 / pointwise accuracy | The share of GPS fixes assigned to the same arc as the aligned reference label. Observation weighting gives longer trips more influence. |
| Candidate macro F1 | The average of the F1 scores for positive and negative candidate labels, with equal weight for the two classes. |
| Log loss | How accurate the classifier’s probabilities are, with a larger penalty for confident errors. Lower is better. |
| Route continuity | The share of adjacent valid road predictions that stay on the same arc or move directly to a connected arc. |
| Route precision, recall, and F1 | Precision asks what fraction of predicted distinct arcs belong to the reference. Recall asks what fraction of reference arcs were predicted. F1 combines precision and recall. The set-based score ignores ordering. |

### Viva prep by owner

| Teammate | Lead on | Question to practice |
|---|---|---|
| Shryes | Ingestion, cleaning, projection, EDA, and summary | **Why project to metres?** Distance in degrees varies with latitude and is awkward for metre-based thresholds. A local metric frame makes the geometry and candidate radii consistent at the scale of these trips. |
| Shubham Bhat | Road graph, candidates, reference alignment, and features | **Why widen the candidate search?** A tight radius keeps plausible comparisons small in normal conditions. The fallback recovers candidates when fixes drift farther from roads. |
| Siddhant Anil | Classifiers, validation, metrics, and Optuna | **Why group by trip?** Nearby fixes share GPS patterns and the same road network. Random candidate-level splits would let related examples appear on both sides. |
| Siddhant Khurdiya | HMM, Viterbi, inference, API, UI, deployment | **Can the HMM hurt?** Yes. A plausible sequence can lead away from the reference on some fixes. Trip `00000017` illustrates better continuity alongside slightly worse pointwise accuracy. |

Everyone should be able to explain which data informed the model and decoder, what the evaluator ZIP was used for, and why **95.89% pointwise accuracy**, **96.43% continuity**, and **90.91% route F1** measure different things.

## 4. Demo and reproduction

**Demo sequence (about three minutes).** Open the evaluator results terminal and show the aggregate and per-trip scores. Open the UI and select `00000015`, enable the classifier-only overlay, then run route reconstruction. Point to the two accuracy values and toggle between the routes. Use `00000017` as a backup discussion example for trade-offs. Keep the saved report and the project’s existing student-test maps available for a demo without the live servers.

On the prepared Windows laptop, use `Start-API.cmd` and then `Start-UI.cmd`. The repository copies are in `scripts/windows/`. Start only the services that are stopped. The API launcher picks up the evaluator directory when present. The app is at [localhost:8501](http://127.0.0.1:8501) and the API docs at [localhost:8000/docs](http://127.0.0.1:8000/docs). A second laptop requires a LAN-reachable API address rather than localhost.

Run these from the repository root after setting up the project environment:

```powershell
.\.venv\Scripts\python.exe scripts\show_results.py all
.\.venv\Scripts\python.exe scripts\evaluate_dataset.py --data data\evaluator_test --output data\evaluator_rerun
.\.venv\Scripts\python.exe scripts\api_smoke_test.py http://127.0.0.1:8000
```

The first command displays the original experiments. The second writes a fresh copy of the evaluation under the ignored data folder, so the committed report is preserved. The third sends predictions for all trips exposed by the running API. Full regeneration of the original training experiments requires the organiser’s train and student-test data. Those folders are empty on this checkout. The existing trained model and supplied evaluator data are enough to reproduce the evaluator scores.

## 5. Deliverables and remaining decisions

| Status | Item | Detail |
|---|---|---|
| Done | Source, trained model, decoder and evaluation | On GitHub `main`. Includes original experimental reports, evaluator results, and a repeatable evaluator script. |
| Done | Laptop setup and demo files | Virtual environment, local evaluator data, stats and server launchers, and the handoff in this file. |
| Checked | Local demo | All ten trips appeared in the UI and one live prediction matched the batch report. The setup uses one laptop on localhost. |
| Confirm with organiser | Submission format and deadline | The repository and accompanying archive message do not establish whether a portal receipt, signed contribution statement, PDF, slides, video, or particular naming convention is required. No official submission was verified. |
| Conditional | Two-laptop rehearsal | If presenting on two machines, start the backend with LAN access, point the evaluator UI at its LAN address, and test `/health`, `/trips`, and one prediction from the second laptop. |
| Conditional | Training-data reproducibility | Restore the original train and student-test folders if the organiser requires full retraining. The evaluator-only run is already reproducible on this laptop. |

For a submission package, the available materials are the repo link, evaluator summary and per-trip CSVs, and this handoff. Confirm the required format before packaging or submitting them. Keep the raw archive in the organiser-approved location.

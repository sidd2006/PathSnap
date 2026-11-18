"""Final evaluation run: send every evaluator trip through the running API.

This is the same path the evaluator uses in the UI: GET /trips, then POST /predict per
trip. Nothing is loaded locally: predictions and scores come from the API response
(scoring against the reference route happens inside the API, server-side). The model
and decoder config are frozen; their checksums are recorded so the run is traceable.

Usage (API already running with PATHSNAP_EVAL_DIR pointing at the evaluator trips):
  python scripts/07_evaluator_run.py [http://<laptop-1-ip>:8000]

Outputs: reports/evaluator_test/{results.csv, summary.json, run_log.txt, maps/pred_<trip>.html}
Maps show GPS, nearby road network and the predicted route only (no reference route).
"""
import hashlib
import json
import os
import sys
import time

import folium
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402

API = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
OUT = os.path.join(config.REPORTS, "evaluator_test")
MAPS = os.path.join(OUT, "maps")
os.makedirs(MAPS, exist_ok=True)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def save_map(res, path):
    gps = res["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14, control_scale=True)
    g = folium.FeatureGroup("Road network (candidate arcs)")
    for line in res["road_network"]:
        folium.PolyLine(line, color="#8c8c8c", weight=1.5, opacity=0.6).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Classifier only", show=False)
    for line in res["classifier_route"]:
        folium.PolyLine(line, color="#ff7f0e", weight=3, opacity=0.8).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Predicted route (HMM)")
    for line in res["predicted_route"]:
        folium.PolyLine(line, color="#d62728", weight=5, opacity=0.85).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("GPS observations")
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color="#1f4fff", fill=True).add_to(g)
    g.add_to(m)
    folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    acc = res["metrics"]["hmm"]["accuracy"] if res.get("metrics") else float("nan")
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
        f'border:1px solid #999;border-radius:6px;font:12px sans-serif"><b>Evaluator trip {res["trip_id"]}</b>'
        f'<br>HMM accuracy {acc:.3f}<br><span style="color:#d62728">&#9644;</span> predicted route (HMM)'
        f'<br><span style="color:#ff7f0e">&#9644;</span> classifier only<br><span style="color:#1f4fff">&#9679;</span> GPS</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    m.save(path)


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    health = requests.get(f"{API}/health", timeout=10).json()
    log(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  API {API}  health={json.dumps(health)}")
    trips = requests.get(f"{API}/trips", timeout=10).json()["trips"]
    log(f"GET /trips -> {len(trips)} trips: {', '.join(trips)}")
    rows = []
    for tid in trips:
        t0 = time.time()
        r = requests.post(f"{API}/predict", json={"trip_id": tid}, timeout=900)
        r.raise_for_status()
        res = r.json()
        assert not ({"route", "reference_route", "true_arcs"} & set(res)), "reference route leaked to client"
        m = res["metrics"]
        row = {"trip_id": tid, "n_obs": res["n_observations"],
               "hmm_accuracy": m["hmm"]["accuracy"], "clf_accuracy": m["classifier"]["accuracy"],
               "hmm_route_continuity": m["hmm"]["route_continuity"], "clf_route_continuity": m["classifier"]["route_continuity"],
               "hmm_route_precision": m["hmm"]["route_precision"], "hmm_route_recall": m["hmm"]["route_recall"],
               "hmm_route_f1": m["hmm"]["route_f1"], "clf_route_f1": m["classifier"]["route_f1"],
               "hmm_breaks": res["decoder"]["breaks"], "obs_without_candidates": res["decoder"]["obs_without_candidates"],
               "server_time_s": res["server_time_s"], "roundtrip_s": round(time.time() - t0, 3)}
        rows.append(row)
        save_map(res, os.path.join(MAPS, f"pred_{tid}.html"))
        log(f"POST /predict {tid}: obs={row['n_obs']:5d}  HMM acc={row['hmm_accuracy']:.4f}  "
            f"classifier acc={row['clf_accuracy']:.4f}  continuity={row['hmm_route_continuity']:.4f}  "
            f"route F1={row['hmm_route_f1']:.4f}  server {row['server_time_s']:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    w = df.n_obs / df.n_obs.sum()
    summary = {
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"), "api": API, "n_trips": len(df), "n_obs": int(df.n_obs.sum()),
        "model": health["model"], "hmm_config": health["hmm"],
        "final_model_sha256_16": sha256(os.path.join(config.ARTIFACTS, "final_model.joblib")),
        "hmm_config_sha256_16": sha256(os.path.join(config.ARTIFACTS, "hmm_config.json")),
        "hmm_accuracy_obs_weighted": float((df.hmm_accuracy * w).sum()),
        "clf_accuracy_obs_weighted": float((df.clf_accuracy * w).sum()),
        "hmm_accuracy_trip_mean": float(df.hmm_accuracy.mean()),
        "clf_accuracy_trip_mean": float(df.clf_accuracy.mean()),
        "hmm_route_continuity_mean": float(df.hmm_route_continuity.mean()),
        "clf_route_continuity_mean": float(df.clf_route_continuity.mean()),
        "hmm_route_f1_mean": float(df.hmm_route_f1.mean()),
        "clf_route_f1_mean": float(df.clf_route_f1.mean()),
        "trips_hmm_acc_ge_95": int((df.hmm_accuracy >= 0.95).sum()),
        "trips_where_hmm_beats_classifier": int((df.hmm_accuracy > df.clf_accuracy).sum()),
        "server_time_s_total": float(df.server_time_s.sum()),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("SUMMARY " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()
                                 if k not in ("hmm_config",)}))
    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

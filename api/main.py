"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            liveness + loaded model info
  GET  /trips             ids of the trips available for evaluation (no labels)
  POST /predict           {"trip_id": "..."} -> full pipeline + server-side scoring

The evaluation folder is set with PATHSNAP_EVAL_DIR (default: data/student_test).
Reference routes are read only AFTER prediction, only for scoring, and are never
returned to the client.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import sys
import time

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.pipeline import evaluate_trip, predict_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", config.STUDENT_TEST_DIR)
MODEL = joblib.load(os.path.join(config.ARTIFACTS, "final_model.joblib"))
META = json.load(open(os.path.join(config.ARTIFACTS, "final_model_meta.json")))
_hmm = json.load(open(os.path.join(config.ARTIFACTS, "hmm_config.json")))
_hmm.pop("tuned_on", None)
HMM_CFG = HMMConfig(**_hmm)
CAND_CFG = META.get("candidate_config", config.CAND_CFG)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _lines(trip, arc_ids):
    out, prev = [], None
    for e in arc_ids:
        e = int(e)
        if e < 0 or e == prev:
            continue
        f, t = trip.arcs[e]
        out.append([[float(trip.nodes[f, 1]), float(trip.nodes[f, 0])], [float(trip.nodes[t, 1]), float(trip.nodes[t, 0])]])
        prev = e
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model"], "eval_dir": os.path.basename(EVAL_DIR), "hmm": HMM_CFG.to_dict()}


@app.get("/trips")
def trips():
    ids = list_trips(EVAL_DIR)
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.perf_counter()
    log.info("POST /predict trip=%s : running pipeline", req.trip_id)
    trip = load_trip(EVAL_DIR, req.trip_id, with_route=False)      # prediction never sees the route
    out = predict_trip(trip, MODEL, HMM_CFG, CAND_CFG)

    route = load_trip(EVAL_DIR, req.trip_id, with_route=True).route  # server-side scoring only
    scores = evaluate_trip(out, route) if route is not None else None
    ctx = out["ctx"]
    resp = {
        "trip_id": req.trip_id,
        "n_observations": int(len(ctx.xy)),
        "gps": [[float(lat), float(lon)] for lon, lat in ctx.track[:, :2]],
        "predicted_route": _lines(trip, out["pred_hmm"]),
        "classifier_route": _lines(trip, out["pred_classifier"]),
        "road_network": _lines(trip, sorted(set(out["cands"]["arc_id"].tolist()))),
        "predicted_arc_ids": [int(x) for x in out["pred_hmm"]],
        "metrics": scores,
        "decoder": out["diag"],
        "timing_s": {k: round(v, 3) for k, v in out["timing_s"].items()},
        "server_time_s": round(time.perf_counter() - t0, 3),
    }
    if scores:
        log.info("POST /predict trip=%s : obs=%d  HMM acc=%.4f  classifier acc=%.4f  continuity=%.4f  route F1=%.4f  (%.2fs)",
                 req.trip_id, resp["n_observations"], scores["hmm"]["accuracy"], scores["classifier"]["accuracy"],
                 scores["hmm"]["route_continuity"], scores["hmm"]["route_f1"], resp["server_time_s"])
    return resp

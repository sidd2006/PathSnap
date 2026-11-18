"""Call a running PathSnap API for every listed trip and print the scores.

Usage:  python scripts/api_smoke_test.py [http://<laptop-1-ip>:8000]
"""
import sys
import time

import requests

api = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
print("health:", requests.get(f"{api}/health", timeout=10).json())
trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
print(f"{len(trips)} trips")
for tid in trips:
    t0 = time.time()
    r = requests.post(f"{api}/predict", json={"trip_id": tid}, timeout=600)
    r.raise_for_status()
    j = r.json()
    m = j["metrics"]
    assert "route" not in j and "reference_route" not in j, "reference route must stay server-side"
    print(f"{tid}: obs={j['n_observations']:5d}  hmm_acc={m['hmm']['accuracy']:.4f}  "
          f"clf_acc={m['classifier']['accuracy']:.4f}  continuity={m['hmm']['route_continuity']:.4f}  "
          f"route_f1={m['hmm']['route_f1']:.4f}  {time.time() - t0:.1f}s")

"""PathSnap evaluator UI (runs on Laptop 2).

Lists the evaluation trips from the API, sends the selected trip id to
POST /predict and renders the returned route and scores. The UI never touches
model files or reference routes - everything comes from the API response.

Run:  streamlit run ui/app.py   (set PATHSNAP_API=http://<laptop-1-ip>:8000)
"""
import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - GPS map-matching")

api = st.sidebar.text_input("API URL", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")
try:
    health = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"Connected: {health['model']}")
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api}: {e}")
    st.stop()

trip_id = st.sidebar.radio(f"Evaluation trips ({len(trips)})", trips)
show_clf = st.sidebar.checkbox("Also show classifier-only route", value=False)
run = st.sidebar.button("Reconstruct route", type="primary")

if run:
    with st.spinner(f"Running pipeline for trip {trip_id} on the API..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    st.subheader(f"Trip {res['trip_id']}  ·  {res['n_observations']} GPS observations")
    m = res.get("metrics")
    if m:
        c = st.columns(5)
        c[0].metric("Accuracy (HMM)", f"{m['hmm']['accuracy']:.2%}",
                    f"{(m['hmm']['accuracy'] - m['classifier']['accuracy']) * 100:+.2f} pts vs classifier")
        c[1].metric("Route continuity", f"{m['hmm']['route_continuity']:.2%}")
        c[2].metric("Route F1", f"{m['hmm']['route_f1']:.2%}")
        c[3].metric("Classifier-only accuracy", f"{m['classifier']['accuracy']:.2%}")
        c[4].metric("API time", f"{res['server_time_s']:.2f}s")

    gps = res["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14, control_scale=True)
    net = folium.FeatureGroup("Road network (candidate arcs)")
    for line in res["road_network"]:
        folium.PolyLine(line, color="#8c8c8c", weight=1.5, opacity=0.6).add_to(net)
    net.add_to(fmap)
    if show_clf:
        g = folium.FeatureGroup("Classifier only")
        for line in res["classifier_route"]:
            folium.PolyLine(line, color="#ff7f0e", weight=3, opacity=0.8).add_to(g)
        g.add_to(fmap)
    g = folium.FeatureGroup("Predicted route (HMM)")
    for line in res["predicted_route"]:
        folium.PolyLine(line, color="#d62728", weight=5, opacity=0.85).add_to(g)
    g.add_to(fmap)
    g = folium.FeatureGroup("GPS observations")
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color="#1f4fff", fill=True).add_to(g)
    g.add_to(fmap)
    folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    components.html(fmap.get_root().render(), height=650)
    with st.expander("Raw API response details"):
        st.json({k: res[k] for k in ("metrics", "decoder", "timing_s")})
else:
    st.info("Pick a trip on the left and press **Reconstruct route**.")

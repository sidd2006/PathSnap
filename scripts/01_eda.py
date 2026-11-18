"""Exploratory data analysis of the raw PathSnap trips (train + student_test).

Outputs (reports/eda/ and reports/figures/)
  inventory.csv          per-trip file sizes, row counts, gaps, extents, network stats
  missing_values.csv     non-finite / invalid / duplicate-timestamp counts per trip
  data_quality_notes.md  short record of issues and assumptions
  fig_*.png              distributions, trajectories, road network, connectivity
  reports/maps/eda_trip_<id>.html   Folium map: GPS, road network corridor, reference route
"""
import os
import sys

import folium
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config  # noqa: E402
from pathsnap.geo import LocalProjection, haversine  # noqa: E402
from pathsnap.io import clean_track, list_trips, load_trip  # noqa: E402
from pathsnap.viz import base_map  # noqa: E402

EDA = os.path.join(config.REPORTS, "eda")
os.makedirs(EDA, exist_ok=True)


def trip_row(partition, root, tid):
    tr = load_trip(root, tid)
    base = os.path.join(root, tid, tid)
    raw = tr.track
    nonfinite = int((~np.isfinite(raw)).any(axis=1).sum())
    clean, rep = clean_track(raw)
    dt = np.diff(clean[:, 2])
    step = haversine(clean[:-1, 0], clean[:-1, 1], clean[1:, 0], clean[1:, 1])
    V, E = len(tr.nodes), len(tr.arcs)
    g = coo_matrix((np.ones(E), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(V, V))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(tr.arcs[:, 0], minlength=V) + np.bincount(tr.arcs[:, 1], minlength=V)
    route = tr.route
    cont = float(np.mean(tr.arcs[route[:-1], 1] == tr.arcs[route[1:], 0])) if len(route) > 1 else np.nan
    return {
        "partition": partition, "trip_id": tid,
        "track_kb": os.path.getsize(base + ".track") / 1024,
        "nodes_kb": os.path.getsize(base + ".nodes") / 1024,
        "arcs_kb": os.path.getsize(base + ".arcs") / 1024,
        "has_route": route is not None,
        "n_obs": len(raw), "n_obs_clean": len(clean),
        "nonfinite_rows": nonfinite, "invalid_rows": rep["invalid"],
        "duplicate_t": rep["duplicate_t"], "reordered": rep["reordered"],
        "duration_s": clean[-1, 2] - clean[0, 2],
        "dt_frac_1s": float(np.mean(dt == 1)), "dt_max": float(dt.max()),
        "n_gaps_gt5s": int((dt > 5).sum()),
        "length_km": float(step.sum() / 1000), "max_step_m": float(step.max()),
        "mean_speed_kmh": float(step.sum() / max(dt.sum(), 1) * 3.6),
        "lon_min": clean[:, 0].min(), "lon_max": clean[:, 0].max(),
        "lat_min": clean[:, 1].min(), "lat_max": clean[:, 1].max(),
        "n_nodes": V, "n_arcs": E, "isolated_nodes": int((deg == 0).sum()),
        "weak_components": n_weak, "largest_weak_frac": float(np.bincount(lab_w).max() / V),
        "strong_components": n_strong, "largest_strong_frac": float(np.bincount(lab_s).max() / V),
        "route_len_arcs": len(route), "route_unique_arcs": len(np.unique(route)),
        "route_continuity": cont,
    }


def folium_trip(root, tid, out_html):
    tr = load_trip(root, tid)
    track, _ = clean_track(tr.track)
    proj = LocalProjection.for_track(track)
    xy = proj.forward(track[:, :2])
    nxy = proj.forward(tr.nodes)
    lo, hi = xy.min(0) - 150, xy.max(0) + 150
    a, b = nxy[tr.arcs[:, 0]], nxy[tr.arcs[:, 1]]
    inside = np.all((a >= lo) & (a <= hi) & (b >= lo) & (b <= hi), axis=1)
    m = base_map([track[:, 1].mean(), track[:, 0].mean()])
    net = folium.FeatureGroup("Road network (corridor)")
    for f, t in tr.arcs[inside][:6000]:
        folium.PolyLine([[tr.nodes[f, 1], tr.nodes[f, 0]], [tr.nodes[t, 1], tr.nodes[t, 0]]],
                        color="#9e9e9e", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route")
    for e in tr.route:
        f, t = tr.arcs[e]
        folium.PolyLine([[tr.nodes[f, 1], tr.nodes[f, 0]], [tr.nodes[t, 1], tr.nodes[t, 0]]],
                        color="green", weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup("GPS observations")
    for lon, lat, ts in track:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, tooltip=f"t={ts:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker([track[0, 1], track[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([track[-1, 1], track[-1, 0]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track[:, 1].min(), track[:, 0].min()], [track[:, 1].max(), track[:, 0].max()]])
    m.save(out_html)


def plots(inv: pd.DataFrame):
    F = config.FIGURES
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, g in inv.groupby("partition"):
        ax[0].hist(g.n_obs, bins=20, alpha=0.6, label=p)
        ax[1].hist(g.length_km, bins=20, alpha=0.6, label=p)
        ax[2].hist(g.mean_speed_kmh, bins=20, alpha=0.6, label=p)
    for a, t in zip(ax, ["GPS observations per trip", "Trip length (km)", "Mean speed (km/h)"]):
        a.set_title(t); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(F, "eda_trip_distributions.png"), dpi=120); plt.close(fig)

    # sampling interval distribution (train)
    dts = []
    for tid in inv[inv.partition == "train"].trip_id:
        tr = load_trip(config.TRAIN_DIR, tid)
        dts.append(np.diff(clean_track(tr.track)[0][:, 2]))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(dts, 0, 15), bins=np.arange(0, 16) + 0.5, log=True)
    ax.set_xlabel("time between consecutive fixes (s, clipped at 15)"); ax.set_ylabel("count (log)")
    ax.set_title(f"Sampling intervals, train ({np.mean(dts == 1):.1%} exactly 1 s)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "eda_sampling_intervals.png"), dpi=120); plt.close(fig)

    # trajectories overview
    fig, ax = plt.subplots(figsize=(8, 8))
    for p, root, c in (("train", config.TRAIN_DIR, "tab:blue"), ("student_test", config.STUDENT_TEST_DIR, "tab:orange")):
        for tid in inv[inv.partition == p].trip_id:
            tr = load_trip(root, tid)
            ax.plot(tr.track[:, 0], tr.track[:, 1], lw=0.7, color=c, alpha=0.7)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("All GPS trajectories (blue=train, orange=student_test)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "eda_all_trajectories.png"), dpi=120); plt.close(fig)

    # one trip: road network + GPS + route
    tid = inv[inv.partition == "train"].trip_id.iloc[0]
    tr = load_trip(config.TRAIN_DIR, tid)
    fig, ax = plt.subplots(figsize=(8, 8))
    lo, hi = tr.track[:, :2].min(0) - 0.004, tr.track[:, :2].max(0) + 0.004
    a, b = tr.nodes[tr.arcs[:, 0]], tr.nodes[tr.arcs[:, 1]]
    ins = np.all((a >= lo) & (a <= hi), axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.stack([a[ins], b[ins]], 1), colors="#bbbbbb", linewidths=0.5))
    rs = np.stack([tr.nodes[tr.arcs[tr.route, 0]], tr.nodes[tr.arcs[tr.route, 1]]], 1)
    ax.add_collection(LineCollection(rs, colors="green", linewidths=2.5, label="reference route"))
    ax.scatter(tr.track[:, 0], tr.track[:, 1], s=3, c="blue", label="GPS")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.legend()
    ax.set_title(f"Trip {tid} (train): road network, GPS and reference route")
    fig.tight_layout(); fig.savefig(os.path.join(F, "eda_road_network_trip.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(inv.n_nodes, inv.n_arcs, c=(inv.partition == "train").map({True: "tab:blue", False: "tab:orange"}))
    ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs"); ax[0].set_title("Road-network size per trip")
    ax[1].hist(inv.largest_strong_frac, bins=20)
    ax[1].set_title("Share of nodes in largest strongly-connected component")
    fig.tight_layout(); fig.savefig(os.path.join(F, "eda_network_connectivity.png"), dpi=120); plt.close(fig)


def main():
    rows = []
    for p, root in (("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)):
        for tid in list_trips(root):
            rows.append(trip_row(p, root, tid))
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(EDA, "inventory.csv"), index=False)
    inv[["partition", "trip_id", "n_obs", "nonfinite_rows", "invalid_rows", "duplicate_t", "reordered"]] \
        .to_csv(os.path.join(EDA, "missing_values.csv"), index=False)
    summ = inv.groupby("partition").agg(
        trips=("trip_id", "size"), obs_total=("n_obs", "sum"), obs_mean=("n_obs", "mean"),
        length_km_mean=("length_km", "mean"), dt_frac_1s=("dt_frac_1s", "mean"), dt_max=("dt_max", "max"),
        gaps_gt5s=("n_gaps_gt5s", "sum"), nodes_mean=("n_nodes", "mean"), arcs_mean=("n_arcs", "mean"),
        isolated_nodes=("isolated_nodes", "sum"), route_continuity=("route_continuity", "mean"),
        nonfinite=("nonfinite_rows", "sum"), duplicate_t=("duplicate_t", "sum"))
    summ.to_csv(os.path.join(EDA, "summary_by_partition.csv"))
    print(summ.T.to_string())
    plots(inv)
    folium_trip(config.TRAIN_DIR, inv[inv.partition == "train"].trip_id.iloc[0],
                os.path.join(config.MAPS, f"eda_trip_{inv.trip_id.iloc[0]}.html"))
    anomalies = inv.sort_values("dt_max", ascending=False).head(5)[["partition", "trip_id", "dt_max", "max_step_m", "n_gaps_gt5s"]]
    with open(os.path.join(EDA, "data_quality_notes.md"), "w") as f:
        f.write("# Data-quality notes (auto-generated by scripts/01_eda.py)\n\n")
        f.write(f"- Trips: {summ.trips.to_dict()}\n")
        f.write(f"- Non-finite GPS rows: {int(inv.nonfinite_rows.sum())}; duplicate timestamps: {int(inv.duplicate_t.sum())}\n")
        f.write(f"- Share of 1 s sampling steps (mean over trips): {inv.dt_frac_1s.mean():.3f}\n")
        f.write(f"- Mean reference-route continuity (head of arc k == tail of arc k+1): {inv.route_continuity.mean():.3f}\n")
        f.write(f"- Isolated road nodes (total): {int(inv.isolated_nodes.sum())}\n")
        f.write("- Each trip ships its own road network; arc ids are only meaningful within a trip.\n")
        f.write("- Coordinates are lon/lat (WGS84); distances use a local equirectangular projection in metres.\n\n")
        f.write("## Trips with largest GPS gaps\n\n" + anomalies.to_markdown(index=False) + "\n")
        cs = os.path.join(config.REPORTS, "candidate_stats_train.csv")
        if os.path.exists(cs):  # written by 02_build_features.py
            c = pd.read_csv(cs, dtype={"trip_id": str})
            bad = c[c.label_dist_median > 30][["trip_id", "n_obs", "label_dist_median", "unreliable_label_frac"]]
            f.write("\n## Reference routes far from the GPS track (train)\n\n"
                    "Median distance from GPS fix to its aligned reference arc > 30 m. These look like "
                    "reference-route errors, not GPS noise; observations > 60 m from their reference arc "
                    "are excluded from training.\n\n" + bad.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()

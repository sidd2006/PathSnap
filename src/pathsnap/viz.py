"""Folium map of a matched trip: GPS, road corridor, reference route and predictions."""
from __future__ import annotations

import folium
import numpy as np


def _arc_lines(nodes, arcs, ids):
    out, prev = [], None
    for e in ids:
        if e < 0 or e == prev:
            continue
        f, t = arcs[e]
        out.append([[nodes[f, 1], nodes[f, 0]], [nodes[t, 1], nodes[t, 0]]])
        prev = e
    return out


def trip_map(trip, track, layers: dict, corridor_arcs=None, title: str = "") -> folium.Map:
    """layers: {name: (arc_id_sequence, colour)}; corridor_arcs: arc ids drawn grey."""
    m = folium.Map(location=[track[:, 1].mean(), track[:, 0].mean()], zoom_start=15, control_scale=True)
    if corridor_arcs is not None:
        g = folium.FeatureGroup("Road network (candidates)", show=True)
        for line in _arc_lines(trip.nodes, trip.arcs, np.unique(corridor_arcs)):
            folium.PolyLine(line, color="#8c8c8c", weight=1.5, opacity=0.6).add_to(g)
        g.add_to(m)
    for name, (ids, colour, weight) in layers.items():
        g = folium.FeatureGroup(name)
        for line in _arc_lines(trip.nodes, trip.arcs, ids):
            folium.PolyLine(line, color=colour, weight=weight, opacity=0.85, tooltip=name).add_to(g)
        g.add_to(m)
    g = folium.FeatureGroup("GPS observations")
    for lon, lat, ts in track:
        folium.CircleMarker([lat, lon], radius=2, color="#1f4fff", fill=True, fill_opacity=0.8,
                            tooltip=f"t={ts:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker([track[0, 1], track[0, 0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([track[-1, 1], track[-1, 0]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    legend = "".join(f'<div><span style="background:{c};display:inline-block;width:18px;height:4px;'
                     f'margin-right:6px;vertical-align:middle"></span>{n}</div>' for n, (_, c, _) in layers.items())
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
        f'border:1px solid #999;border-radius:6px;font:12px sans-serif"><b>{title}</b>{legend}'
        f'<div><span style="color:#1f4fff">&#9679;</span> GPS observation</div></div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track[:, 1].min(), track[:, 0].min()], [track[:, 1].max(), track[:, 0].max()]])
    return m

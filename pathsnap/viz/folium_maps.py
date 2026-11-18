"""Folium map builder shared by the UI (Laptop 2) and the report scripts (owner: Person 4).

Inputs are plain lists of [lat, lon] so that this module only needs
``folium`` - Laptop 2 does not need numpy/sklearn models to draw a map.

Layers (each toggleable in the layer control):
  * Road network near the track  - grey, thin
  * GPS observations             - blue dots
  * Classifier-only matched arcs - orange, dashed
  * HMM predicted route          - red, thick
  * Snapped positions            - small dark-red dots (off by default)
  * Reference route              - green; ONLY drawn by offline report scripts
                                   for student-test trips, never by the UI
"""
from __future__ import annotations

import folium
from branca.element import MacroElement, Template

COLORS = {"network": "#9e9e9e", "gps": "#1f77b4", "clf": "#ff7f0e", "hmm": "#d62728",
          "snap": "#7f0000", "reference": "#2ca02c"}


def _legend(items: list[tuple[str, str]], title: str) -> MacroElement:
    rows = "".join(f'<div><span style="background:{c};width:18px;height:4px;display:inline-block;'
                   f'margin-right:6px;vertical-align:middle"></span>{label}</div>' for label, c in items)
    html = f"""{{% macro html(this, kwargs) %}}
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;
                border:1px solid #888;border-radius:4px;font:12px sans-serif;">
      <b>{title}</b>{rows}</div>{{% endmacro %}}"""
    el = MacroElement()
    el._template = Template(html)
    return el


def build_result_map(gps: list, hmm_route: list | None = None, clf_segments: list | None = None,
                     network_segments: list | None = None, snapped: list | None = None,
                     reference_route: list | None = None, title: str = "PathSnap",
                     max_gps_points: int = 4000) -> folium.Map:
    """``gps``: [[lat, lon, (t)], ...]; routes: list of polylines; segments: list of 2-point lines."""
    if not gps:
        raise ValueError("gps must contain at least one point")
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    legend = []
    if network_segments:
        fg = folium.FeatureGroup(name=f"Road network near track ({len(network_segments)} arcs)")
        for seg in network_segments:
            folium.PolyLine(seg, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))
    if reference_route:
        fg = folium.FeatureGroup(name="Reference route (student-test only)")
        for line in reference_route:
            folium.PolyLine(line, color=COLORS["reference"], weight=8, opacity=0.5).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))
    if clf_segments:
        fg = folium.FeatureGroup(name="Classifier-only matched arcs", show=False)
        for seg in clf_segments:
            folium.PolyLine(seg, color=COLORS["clf"], weight=3, opacity=0.9, dash_array="6 6").add_to(fg)
        fg.add_to(m)
        legend.append(("Classifier-only arcs", COLORS["clf"]))
    if hmm_route:
        fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
        for line in hmm_route:
            folium.PolyLine(line, color=COLORS["hmm"], weight=4, opacity=0.9).add_to(fg)
        fg.add_to(m)
        legend.append(("Predicted route (HMM)", COLORS["hmm"]))
    step = max(1, len(gps) // max_gps_points)
    fg = folium.FeatureGroup(name=f"GPS observations ({len(gps)})")
    for p in gps[::step]:
        tip = f"t = {p[2]:.0f} s" if len(p) > 2 else None
        folium.CircleMarker(p[:2], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=1, tooltip=tip).add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))
    if snapped:
        fg = folium.FeatureGroup(name="Snapped positions", show=False)
        for p in snapped[::step]:
            folium.CircleMarker(p, radius=1.5, color=COLORS["snap"], fill=True).add_to(fg)
        fg.add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().add_child(_legend(legend, title))
    return m

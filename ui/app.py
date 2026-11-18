"""PathSnap evaluator UI - runs on Laptop 2 (owner: Person 4).

    streamlit run ui/app.py -- --api http://<LAPTOP1_IP>:8000
    (or set PATHSNAP_API_URL)

Flow: list trips from GET /trips -> evaluator picks one -> POST /predict with
the trip id -> show the predicted route on a Folium map and the metrics the
API computed server-side.  The UI never reads data files and never receives
the reference route.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from pathsnap.viz.folium_maps import build_result_map


def _cli_api() -> str | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None)
    args, _ = ap.parse_known_args()
    return args.api


DEFAULT_API = _cli_api() or os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
METRIC_LABELS = {
    "pointwise_accuracy": "Point-wise accuracy (per GPS observation)",
    "direction_agnostic_accuracy": "Accuracy ignoring travel direction",
    "route_continuity": "Route continuity (connected consecutive arcs)",
    "route_precision": "Route precision (arcs)",
    "route_recall": "Route recall (arcs)",
    "route_f1": "Route F1 (arcs)",
    "route_len_f1": "Route F1 (length-weighted)",
    "route_mismatch_fraction": "Route mismatch fraction (lower is better)",
    "edit_similarity": "Sequence edit similarity",
}

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - map-matching & route reconstruction")

with st.sidebar:
    st.header("Connection")
    api = st.text_input("API URL (Laptop 1)", value=st.session_state.get("api", DEFAULT_API)).rstrip("/")
    st.session_state["api"] = api
    timeout = st.number_input("Request timeout (s)", 10, 600, 180)
    try:
        h = requests.get(f"{api}/health", timeout=5).json()
        st.success(f"API online - model: {h['model']}")
        st.caption(f"trips served: {h['n_trips']}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API not reachable: {exc}")
        st.stop()


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(api_url: str):
    r = requests.get(f"{api_url}/trips", timeout=30)
    r.raise_for_status()
    return r.json()


try:
    tl = get_trips(api)
except Exception as exc:  # noqa: BLE001
    st.error(f"could not load trips: {exc}")
    st.stop()

trips = tl["trips"]
if not trips:
    st.warning("The API has no trips to serve.")
    st.stop()

st.subheader(f"1. Choose an evaluation trip ({len(trips)} available, partition: {tl['partition']})")
labels = {f"{t['trip_id']}  -  {t['n_observations']} GPS points, {t['duration_s'] / 60:.1f} min": t["trip_id"]
          for t in trips}
choice = st.radio("Trip", list(labels), horizontal=False, label_visibility="collapsed")
trip_id = labels[choice]

if st.button("2. Run map-matching on the API", type="primary"):
    with st.spinner(f"API is processing trip {trip_id} ..."):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=timeout)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except requests.HTTPError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"request failed: {exc}")

res = st.session_state.get("result")
if res:
    st.subheader(f"3. Result for trip {res['trip_id']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPS observations", res["n_observations"])
    c2.metric("Matched observations", res["n_matched_observations"])
    c3.metric("Predicted route arcs", len(res["predicted_route_arc_ids"]))
    c4.metric("API processing time", f"{res['timings_s']['request_total_s']:.2f} s")

    ev = res["evaluation"]
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Evaluation (computed on the API server)**")
        if ev["available"]:
            rows = [{"metric": METRIC_LABELS[k], "classifier + HMM": ev["hmm"][k],
                     "classifier only": ev["classifier_only"][k]} for k in METRIC_LABELS]
            st.dataframe(pd.DataFrame(rows).set_index("metric").style.format("{:.4f}"))
            st.caption(f"{ev['hmm']['n_labeled_obs']} labelled observations; {ev['note']}")
        else:
            st.info(ev["note"])
        st.markdown("**Run details**")
        st.json({"model": res["model_name"], "hmm_params": res["hmm_params_source"],
                 "hmm_segments": res["n_hmm_segments"], "unfilled_gaps": res["n_unfilled_gaps"],
                 "timings_s": res["timings_s"]}, expanded=False)
    with right:
        m = build_result_map(res["gps"], hmm_route=res["predicted_route"],
                             clf_segments=res["classifier_only_segments"],
                             network_segments=res["network_segments"], snapped=res["snapped_points"],
                             title=f"Trip {res['trip_id']}")
        html = m.get_root().render()          # built locally from our own API response
        if hasattr(st, "iframe"):              # Streamlit >= 1.5x
            st.iframe(html, height=650)
        else:                                  # older Streamlit
            components.html(html, height=650)

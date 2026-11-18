"""Report maps for student-test trips (owner: Person 4).

Draws GPS, nearby road network, classifier-only arcs, HMM route AND the
reference route.  The reference layer is allowed here because these are
student-test trips used offline for the report - this script is never run
on eval_test and the UI never shows a reference route.

Input:  results/hmm/predictions/<trip>.json   (from scripts/evaluate_hmm.py)
Output: results/maps/student_test_<trip>.html

Usage:  python scripts/make_maps.py [--trips 00000000 00000056]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pathsnap.config import load_config, partition_dir, results_dir
from pathsnap.dataset import prepare_trip
from pathsnap.io import load_reference_route, load_trip_input
from pathsnap.pipeline import arc_segments, route_polylines
from pathsnap.viz.folium_maps import build_result_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config()
    pred_dir = results_dir(cfg, "hmm", "predictions")
    out = results_dir(cfg, "maps")
    files = sorted(pred_dir.glob("*.json"))
    if args.trips:
        files = [f for f in files if f.stem in args.trips]
    if not files:
        raise SystemExit("no predictions found - run scripts/evaluate_hmm.py first")
    for f in files:
        p = json.loads(f.read_text())
        d = partition_dir(cfg, "student_test") / p["trip_id"]
        trip = load_trip_input(d)
        pt = prepare_trip(trip, cfg)
        g = pt.graph
        ref = g.canonical[load_reference_route(d, len(trip.arcs))]
        clf = np.array(p["classifier_only_obs_arc"])
        m = build_result_map(
            [[r[1], r[0], r[2]] for r in trip.track], hmm_route=route_polylines(p["hmm_route"], g),
            clf_segments=arc_segments(clf, g, cfg["api"]["max_network_segments"]),
            network_segments=arc_segments(pt.cands.arc_id.unique(), g, cfg["api"]["max_network_segments"]),
            reference_route=route_polylines(list(ref), g),
            title=f"student-test {p['trip_id']} ({p['model']} + HMM)")
        m.save(str(out / f"student_test_{p['trip_id']}.html"))
        print("map:", out / f"student_test_{p['trip_id']}.html")


if __name__ == "__main__":
    main()

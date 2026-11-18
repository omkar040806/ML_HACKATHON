"""Score the final system on the evaluator trips, exactly as the API does.

For every trip the pipeline first PREDICTS from track + nodes + arcs only
(PathSnapPipeline.predict never sees the route), then SCORES the prediction
against the reference route (PathSnapPipeline.evaluate).  Nothing is trained
or tuned here: the model bundle and HMM parameters are the committed ones.
These numbers must not be used to change the model or the decoder settings.

Input:  evaluator trips in data/raw/evaluator_test/<trip_id>/ (git-ignored),
        or the folder given by PATHSNAP_EVAL_DIR / --dir
Output: results/evaluator/per_trip_metrics.csv   (one row per trip x variant)
        results/evaluator/summary.csv            (pooled and mean metrics per variant)

Usage:  python scripts/score_evaluator.py [--dir C:/path/to/evaluator_test]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pathsnap.config import load_config, partition_dir, results_dir
from pathsnap.io import list_trips, load_reference_route, load_trip_input
from pathsnap.pipeline import PathSnapPipeline

log = logging.getLogger("score_evaluator")

METRICS = ["pointwise_accuracy", "direction_agnostic_accuracy", "route_continuity", "route_f1",
           "route_len_f1", "route_mismatch_fraction", "edit_similarity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="folder with one sub-folder per evaluator trip")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    pdir = Path(args.dir) if args.dir else partition_dir(cfg, "eval_test")
    ids = list_trips(pdir)
    if not ids:
        raise SystemExit(f"No evaluator trips found in {pdir}")
    pipe = PathSnapPipeline(cfg)

    rows = []
    for tid in ids:
        t0 = time.perf_counter()
        trip = load_trip_input(pdir / tid)
        res, pt = pipe.predict(trip)                       # prediction first, no route
        predict_s = time.perf_counter() - t0
        ev = pipe.evaluate(pt, res, load_reference_route(pdir / tid, len(trip.arcs)))
        for variant, key in (("classifier_only", "classifier_only"), ("hmm_classifier", "hmm")):
            m = ev[key]
            rows.append({"trip_id": tid, "variant": variant, "n_obs": trip.n_obs,
                         "n_labeled_obs": m["n_labeled_obs"], **{k: m[k] for k in METRICS},
                         "predict_s": predict_s})
        log.info("%s: classifier_only %.4f | hmm %.4f", tid, ev["classifier_only"]["pointwise_accuracy"],
                 ev["hmm"]["pointwise_accuracy"])

    df = pd.DataFrame(rows)
    out = results_dir(cfg, "evaluator")
    df.to_csv(out / "per_trip_metrics.csv", index=False)

    acc = df.pivot(index="trip_id", columns="variant", values="pointwise_accuracy")
    summ = []
    for variant, g in df.groupby("variant", sort=False):
        row = {"variant": variant, "n_trips": len(g), "n_labeled_obs": int(g.n_labeled_obs.sum()),
               "pooled_pointwise_accuracy": float((g.pointwise_accuracy * g.n_labeled_obs).sum()
                                                  / g.n_labeled_obs.sum())}
        row.update({k: float(g[k].mean()) for k in METRICS})
        if variant != "classifier_only":
            row["trips_better_than_classifier_only"] = int((acc[variant] > acc["classifier_only"]).sum())
            row["trips_worse_than_classifier_only"] = int((acc[variant] < acc["classifier_only"]).sum())
        row["mean_predict_s"] = float(g.predict_s.mean())
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.set_index("variant").T.to_string(float_format=lambda v: f"{v:.4f}"))
    log.info("written to %s", out)


if __name__ == "__main__":
    main()

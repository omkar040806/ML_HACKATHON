"""Classifier-only vs classifier + HMM/Viterbi on student-test trips (owner: Person 3).

Variants compared for every trip (same candidates, same model bundle):
  * classifier_only : independent argmax per observation, no sequence model
  * hmm_tuned       : HMM with the tuned emission mode from best_hmm_params.json (deployed system)
  * hmm_classifier  : HMM with pure classifier-probability emissions (comparison)
  * hmm_gaussian    : HMM with distance-only Gaussian emissions (Newson & Krumm
                      baseline, isolates the value of the classifier)

Decoder parameters come from results/hmm/best_hmm_params.json (written by
tune_hmm.py on TRAINING trips) if it exists, else from configs/pathsnap.yaml.

Outputs (results/hmm/):
  per_trip_metrics.csv, summary.csv, hmm_params_used.json,
  predictions/<trip>.json (for scripts/make_maps.py), figures/*.png

Usage:  python scripts/evaluate_hmm.py [--partition student_test]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import load_config, models_dir, partition_dir, results_dir
from pathsnap.dataset import process_trip_dir
from pathsnap.hmm.decoder import HMMMapMatcher, classifier_only, collapse
from pathsnap.hmm.route_metrics import evaluate_prediction
from pathsnap.io import list_trips, load_reference_route
from pathsnap.models.artifact import load_bundle

log = logging.getLogger("evaluate_hmm")


def load_hmm_params(cfg: dict) -> tuple[dict, str]:
    p = results_dir(cfg, "hmm") / "best_hmm_params.json"
    hmm = dict(cfg["hmm"])
    if p.exists():
        hmm.update(json.loads(p.read_text())["params"])
        return hmm, str(p)
    return hmm, "configs/pathsnap.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test", choices=["student_test", "train"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    out = results_dir(cfg, "hmm")
    pred_dir = results_dir(cfg, "hmm", "predictions")
    fig_dir = results_dir(cfg, "hmm", "figures")
    bundle = load_bundle(models_dir(cfg) / "final_model.joblib")
    hmm_cfg, src = load_hmm_params(cfg)
    (out / "hmm_params_used.json").write_text(json.dumps({"source": src, "params": hmm_cfg,
                                                          "model": bundle.model_name}, indent=1))
    log.info("model %s, HMM params from %s", bundle.model_name, src)

    pdir = partition_dir(cfg, args.partition)
    rows = []
    for tid in list_trips(pdir):
        pt = process_trip_dir(pdir / tid, cfg, with_labels=True)
        ref_route = load_reference_route(pdir / tid, len(pt.trip.arcs))
        t0 = time.perf_counter()
        prob = bundle.predict_proba(pt.cands)
        clf_s = time.perf_counter() - t0
        ref, lab = pt.obs.ref_arc.to_numpy(), pt.obs.labeled.to_numpy()
        mm = HMMMapMatcher(pt.graph, hmm_cfg)
        variants = {}
        co = classifier_only(pt.cands, prob, len(pt.x))
        variants["classifier_only"] = (co, collapse(co), {"decode_s": 0.0}, 1, 0)
        # hmm_tuned = the deployed decoder (emission mode from best_hmm_params.json, e.g. blend);
        # hmm_classifier / hmm_gaussian are fixed-mode comparisons with the same beta and gap settings.
        for name, mode in (("hmm_tuned", hmm_cfg["emission"]), ("hmm_classifier", "classifier"),
                           ("hmm_gaussian", "gaussian")):
            r = mm.decode(pt.cands, prob, pt.x, pt.y, pt.t, emission_mode=mode)
            variants[name] = (r.obs_arc, r.route, r.timings, r.n_segments, r.n_unfilled_gaps)
            if name == "hmm_tuned":
                raw_cont = evaluate_prediction(r.obs_arc, r.route_raw, ref, lab, ref_route, pt.graph)["route_continuity"]
        for v, (obs_arc, route, tim, nseg, unf) in variants.items():
            m = evaluate_prediction(obs_arc, route, ref, lab, ref_route, pt.graph)
            m.update({"trip_id": tid, "variant": v, "n_obs": len(pt.x), "n_segments": nseg,
                      "n_unfilled_gaps": unf, "decode_s": tim["decode_s"], "classifier_s": clf_s,
                      "candidate_miss_rate": float(1 - pt.obs.ref_in_cands[pt.obs.labeled].mean())})
            if v == "hmm_tuned":
                m["route_continuity_before_fill"] = raw_cont
            rows.append(m)
        (pred_dir / f"{tid}.json").write_text(json.dumps({
            "trip_id": tid, "model": bundle.model_name,
            "classifier_only_obs_arc": variants["classifier_only"][0].tolist(),
            "hmm_obs_arc": variants["hmm_tuned"][0].tolist(),
            "hmm_route": [int(a) for a in variants["hmm_tuned"][1]]}))
        acc = {v: rows[-len(variants) + k]["pointwise_accuracy"] for k, v in enumerate(variants)}
        log.info("%s: %s", tid, ", ".join(f"{k}={v:.4f}" for k, v in acc.items()))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"per_trip_metrics_{args.partition}.csv", index=False)
    metric_cols = ["pointwise_accuracy", "direction_agnostic_accuracy", "route_continuity", "route_f1",
                   "route_len_f1", "route_mismatch_fraction", "edit_similarity", "n_segments", "decode_s"]
    summ = df.groupby("variant")[metric_cols].mean()
    pooled = df.assign(correct=df.pointwise_accuracy * df.n_labeled_obs).groupby("variant") \
        .apply(lambda g: g.correct.sum() / g.n_labeled_obs.sum(), include_groups=False)
    summ.insert(0, "pooled_pointwise_accuracy", pooled)
    piv = df.pivot(index="trip_id", columns="variant", values="pointwise_accuracy")
    summ["trips_where_better_than_classifier_only"] = [
        int((piv[v] > piv["classifier_only"] + 1e-12).sum()) for v in summ.index]
    summ["trips_where_worse_than_classifier_only"] = [
        int((piv[v] < piv["classifier_only"] - 1e-12).sum()) for v in summ.index]
    summ.to_csv(out / f"summary_{args.partition}.csv")
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))

    for metric in ("pointwise_accuracy", "route_continuity", "route_len_f1"):
        p = df.pivot(index="trip_id", columns="variant", values=metric)
        ax = p[["classifier_only", "hmm_gaussian", "hmm_classifier", "hmm_tuned"]].plot.bar(figsize=(11, 4), width=0.8)
        ax.set(title=f"{args.partition}: {metric} per trip", ylabel=metric,
               ylim=(max(0, np.nanmin(p.values) - 0.05), 1.0))
        ax.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}_{args.partition}.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()

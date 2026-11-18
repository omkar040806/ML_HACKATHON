"""Tune HMM decoder parameters on TRAINING trips with out-of-fold probabilities (owner: Person 3).

Why out-of-fold: the final classifier was fitted on all training trips, so its
probabilities on those trips are over-confident.  Here the tuning trips are
split into 3 groups; for each group a classifier with the final model's
configuration is fitted on all *other* training trips and used to score the
group.  The decoder therefore sees realistic, unseen-trip probabilities.

Search (Optuna TPE, recorded): beta_m, emission mode, emission_weight,
sigma_m, gap_break_s.  Objective: pooled point-wise accuracy over the tuning
trips.  Student-test and eval-test trips are not used.

Outputs: results/hmm/tuning_trials.csv, results/hmm/best_hmm_params.json

Usage:  python scripts/tune_hmm.py [--n-trials 25] [--n-trips 20]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna

from pathsnap.config import load_config, models_dir, partition_dir, results_dir
from pathsnap.dataset import build_partition_tables, process_trip_dir
from pathsnap.hmm.decoder import HMMMapMatcher
from pathsnap.io import list_trips
from pathsnap.models.artifact import load_bundle
from pathsnap.models.cv import fit, predict_proba, training_rows

log = logging.getLogger("tune_hmm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--n-trips", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config()
    seed, hc, tc = cfg["seed"], cfg["hmm"], cfg["training"]
    out = results_dir(cfg, "hmm")
    bundle = load_bundle(models_dir(cfg) / "final_model.joblib")

    pdir = partition_dir(cfg, "train")
    all_ids = list_trips(pdir)
    rng = np.random.default_rng(seed)
    n_trips = min(args.n_trips or hc["tune_trips"], len(all_ids))
    tune_ids = sorted(rng.choice(all_ids, size=n_trips, replace=False).tolist())
    groups = [tune_ids[k::3] for k in range(3)]
    cands, _ = build_partition_tables(cfg, "train")

    trips = {}
    t0 = time.perf_counter()
    for k, grp in enumerate(groups):
        rows = training_rows(cands[~cands.trip_id.isin(grp)], tc["obs_stride"])
        model, _, _ = fit(bundle.model_name, bundle.params, rows, seed, args.n_jobs, tc["slow_model_max_rows"])
        for tid in grp:
            pt = process_trip_dir(pdir / tid, cfg, with_labels=True)
            prob, _ = predict_proba(model, pt.cands)
            trips[tid] = (pt, prob)
        log.info("OOF group %d/3 scored (%d trips)", k + 1, len(grp))
    log.info("OOF probabilities ready in %.0fs", time.perf_counter() - t0)

    def score(params: dict) -> tuple[float, float]:
        correct = total = 0
        dec_s = 0.0
        for pt, prob in trips.values():
            r = HMMMapMatcher(pt.graph, params).decode(pt.cands, prob, pt.x, pt.y, pt.t)
            lab = pt.obs.labeled.to_numpy()
            correct += int((r.obs_arc[lab] == pt.obs.ref_arc.to_numpy()[lab]).sum())
            total += int(lab.sum())
            dec_s += r.timings["decode_s"]
        return correct / total, dec_s

    def objective(trial: optuna.Trial) -> float:
        p = dict(hc)
        p["beta_m"] = trial.suggest_float("beta_m", 1.0, 50.0, log=True)
        p["emission"] = trial.suggest_categorical("emission", ["classifier", "blend"])
        p["emission_weight"] = trial.suggest_float("emission_weight", 0.3, 1.0) if p["emission"] == "blend" else 1.0
        p["sigma_m"] = trial.suggest_float("sigma_m", 3.0, 20.0) if p["emission"] == "blend" else hc["sigma_m"]
        p["gap_break_s"] = trial.suggest_categorical("gap_break_s", [30.0, 60.0, 120.0, 300.0])
        acc, dec_s = score(p)
        trial.set_user_attr("decode_s_total", dec_s)
        return acc

    base_acc, _ = score(dict(hc))
    log.info("config defaults: pooled accuracy %.4f", base_acc)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({"beta_m": hc["beta_m"], "emission": hc["emission"], "gap_break_s": hc["gap_break_s"]})
    t1 = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials or hc["tune_trials"])
    study.trials_dataframe().to_csv(out / "tuning_trials.csv", index=False)
    bp = study.best_params
    params = {"beta_m": bp["beta_m"], "emission": bp["emission"], "gap_break_s": bp["gap_break_s"],
              "emission_weight": bp.get("emission_weight", 1.0), "sigma_m": bp.get("sigma_m", hc["sigma_m"])}
    (out / "best_hmm_params.json").write_text(json.dumps({
        "params": params, "objective": "pooled point-wise accuracy, OOF probabilities, training trips",
        "best_value": study.best_value, "config_default_value": base_acc, "tune_trips": tune_ids,
        "n_trials": len(study.trials), "search_wall_s": time.perf_counter() - t1}, indent=1))
    log.info("best %.4f (defaults %.4f): %s", study.best_value, base_acc, params)


if __name__ == "__main__":
    main()

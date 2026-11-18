"""Train and compare candidate classifiers (owner: Person 2).

For every model in ``training.models``:
  1. GroupKFold cross-validation over the 80 training trips (trip-level groups)
  2. refit on all training trips
  3. evaluate on the 10 student-test trips (all observations)

Selection rule (decided BEFORE looking at student-test numbers): highest mean
CV observation-level Top-1 accuracy; ties (< 0.002) broken by lower CV log
loss, because the HMM consumes the probabilities.  Student-test results are
reported for comparison but are not used to choose.

Outputs (results/models/):
  cv_folds.csv, comparison.csv, confusion_matrices.json, model_configs.json,
  mlp_history.json, selection.json, run_log.txt

Usage:  python scripts/train_compare.py [--models lightgbm logreg ...] [--n-jobs -1] [--resume]
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import sklearn

from pathsnap.config import load_config, results_dir
from pathsnap.dataset import build_partition_tables
from pathsnap.models.cv import fit, group_cv, predict_proba, training_rows
from pathsnap.models.metrics import evaluate_all
from pathsnap.models.zoo import get_spec

log = logging.getLogger("train_compare")


def mlp_history(model) -> dict | None:
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "loss_curve_"):
        return {"train_loss": [float(v) for v in clf.loss_curve_],
                "val_accuracy": [float(v) for v in (clf.validation_scores_ or [])],
                "n_iter": int(clf.n_iter_)}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--resume", action="store_true",
                    help="keep results of models already in comparison.csv and only run the missing ones")
    args = ap.parse_args()
    cfg = load_config()
    out = results_dir(cfg, "models")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / "run_log.txt", mode="a" if args.resume else "w")])
    tc = cfg["training"]
    seed = cfg["seed"]
    names = args.models or tc["models"]
    log.info("python %s, sklearn %s, seed %d, models %s", platform.python_version(), sklearn.__version__, seed, names)

    tr_c, tr_o = build_partition_tables(cfg, "train")
    st_c, st_o = build_partition_tables(cfg, "student_test")
    log.info("train: %d trips, %d candidate rows; student_test: %d trips", tr_c.trip_id.nunique(), len(tr_c),
             st_c.trip_id.nunique())

    all_folds, rows, cms, configs, hist = [], [], {}, {}, {}
    if args.resume and (out / "comparison.csv").exists():
        prev = pd.read_csv(out / "comparison.csv")
        rows = prev.to_dict("records")          # keep every finished model
        done = {r["model"] for r in rows}
        pf = pd.read_csv(out / "cv_folds.csv")
        all_folds = [pf[pf.model.isin(done)]]
        cms = {k: v for k, v in json.loads((out / "confusion_matrices.json").read_text()).items() if k in done}
        configs = {k: v for k, v in json.loads((out / "model_configs.json").read_text()).items() if k in done}
        hist = {k: v for k, v in json.loads((out / "mlp_history.json").read_text()).items() if k in done}
        log.info("resume: keeping %s", sorted(done))
    else:
        done = set()

    def save() -> None:
        pd.concat(all_folds).to_csv(out / "cv_folds.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
        (out / "confusion_matrices.json").write_text(json.dumps(cms, indent=1))
        (out / "model_configs.json").write_text(json.dumps(configs, indent=1, default=str))
        (out / "mlp_history.json").write_text(json.dumps(hist, indent=1))

    for name in names:
        if name in done:
            continue
        spec = get_spec(name)
        log.info("=== %s (%s)", name, spec.family)
        t0 = time.perf_counter()
        folds = group_cv(name, None, tr_c, tr_o, tc["cv_folds"], tc["obs_stride"], seed, tc["threshold"],
                         args.n_jobs, tc["slow_model_max_rows"])
        folds.insert(0, "model", name)
        all_folds.append(folds)
        cv_s = time.perf_counter() - t0
        log.info("%s CV top1 %.4f +- %.4f | macroF1 %.4f | logloss %.4f (%.0fs)", name,
                 folds.top1_accuracy.mean(), folds.top1_accuracy.std(), folds.macro_f1.mean(),
                 folds.log_loss.mean(), cv_s)

        model, params, info = fit(name, None, training_rows(tr_c, tc["obs_stride"]), seed, args.n_jobs,
                                  tc["slow_model_max_rows"])
        prob, pred_s = predict_proba(model, st_c)
        st = evaluate_all(st_c, st_o, prob, tc["threshold"])
        cms[name] = st.pop("confusion")
        h = mlp_history(model)
        if h:
            hist[name] = h
        configs[name] = {"family": spec.family, "params": params, "notes": spec.notes, "slow": spec.slow,
                         "fit_rows": info["fit_rows"], "seed": seed, "obs_stride": tc["obs_stride"]}
        row = {"model": name, "family": spec.family, "key_config": json.dumps(params, default=str),
               "cv_top1_mean": folds.top1_accuracy.mean(), "cv_top1_std": folds.top1_accuracy.std(),
               "cv_macro_f1_mean": folds.macro_f1.mean(), "cv_macro_f1_std": folds.macro_f1.std(),
               "cv_log_loss_mean": folds.log_loss.mean(), "cv_f1_mean": folds.f1.mean(),
               "cv_route_continuity_mean": folds.route_continuity.mean(),
               "train_fit_s": info["fit_s"], "train_fit_rows": info["fit_rows"], "cv_total_s": cv_s}
        row.update({f"st_{k}": v for k, v in st.items()})
        row["st_predict_s"] = pred_s
        row["st_predict_ms_per_1k_obs"] = 1e6 * pred_s / len(st_o)
        rows.append(row)
        log.info("%s student-test top1 %.4f | macroF1 %.4f | logloss %.4f | continuity %.4f", name,
                 st["top1_accuracy"], st["macro_f1"], st["log_loss"], st["route_continuity"])
        save()          # after every model, so an interrupted run can --resume

    comp = pd.DataFrame(rows)

    best_top1 = comp.cv_top1_mean.max()
    close = comp[comp.cv_top1_mean >= best_top1 - 0.002]
    chosen = close.sort_values("cv_log_loss_mean").iloc[0]
    sel = {"selected_model": chosen.model,
           "rule": "max mean CV top-1 accuracy; within 0.002 of best, lowest mean CV log loss",
           "cv_top1_mean": float(chosen.cv_top1_mean), "cv_log_loss_mean": float(chosen.cv_log_loss_mean),
           "candidates_within_tolerance": close.model.tolist(),
           "ranking_by_cv_top1": comp.sort_values("cv_top1_mean", ascending=False).model.tolist()}
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    cols = ["model", "family", "cv_macro_f1_mean", "cv_top1_mean", "st_top1_accuracy", "st_route_continuity",
            "st_macro_f1", "st_log_loss", "st_predict_ms_per_1k_obs"]
    print(comp[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log.info("selected model: %s", sel["selected_model"])


if __name__ == "__main__":
    main()

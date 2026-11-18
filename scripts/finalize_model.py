"""Tuned vs untuned comparison and export of the final model (owner: Person 2).

Both configurations are evaluated with the SAME protocol:
  * GroupKFold CV on train (training.cv_folds, training.obs_stride)
  * refit on all training trips -> student-test metrics + inference time

The exported configuration is the one with the higher CV Top-1 accuracy
(normally the tuned one).  Student-test numbers are reported, not used to
decide.

Outputs:
  models/final_model.joblib                      (ModelBundle, see pathsnap/models/artifact.py)
  results/bayes_opt/tuned_vs_untuned.csv
  results/bayes_opt/tuned_vs_untuned_folds.csv
  results/models/final_model_card.json

Usage:  python scripts/finalize_model.py [--model lightgbm]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pathsnap.config import load_config, models_dir, results_dir
from pathsnap.dataset import build_partition_tables
from pathsnap.models.artifact import save_bundle
from pathsnap.models.cv import fit, group_cv, predict_proba, training_rows
from pathsnap.models.metrics import evaluate_all
from pathsnap.models.zoo import get_spec

log = logging.getLogger("finalize")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    tc, seed = cfg["training"], cfg["seed"]
    bo_dir, m_dir = results_dir(cfg, "bayes_opt"), results_dir(cfg, "models")
    name = args.model or json.loads((m_dir / "selection.json").read_text())["selected_model"]
    best_file = bo_dir / f"best_params_{name}.json"
    if not best_file.exists():
        raise SystemExit(f"{best_file} missing - run scripts/bayes_opt.py --model {name} first")
    tuned_params = json.loads(best_file.read_text())["best_params"]
    spec = get_spec(name)

    tr_c, tr_o = build_partition_tables(cfg, "train")
    st_c, st_o = build_partition_tables(cfg, "student_test")
    rows, fold_tabs, fitted = [], [], {}
    for variant, params in (("untuned", dict(spec.default_params)), ("tuned", tuned_params)):
        folds = group_cv(name, params, tr_c, tr_o, tc["cv_folds"], tc["obs_stride"], seed, tc["threshold"],
                         args.n_jobs, tc["slow_model_max_rows"])
        folds.insert(0, "variant", variant)
        fold_tabs.append(folds)
        model, p, info = fit(name, params, training_rows(tr_c, tc["obs_stride"]), seed, args.n_jobs,
                             tc["slow_model_max_rows"])
        prob, pred_s = predict_proba(model, st_c)
        st = evaluate_all(st_c, st_o, prob, tc["threshold"])
        st.pop("confusion")
        fitted[variant] = (model, p, info)
        rows.append({"model": name, "variant": variant, "params": json.dumps(p, default=str),
                     "cv_top1_mean": folds.top1_accuracy.mean(), "cv_top1_std": folds.top1_accuracy.std(),
                     "cv_macro_f1_mean": folds.macro_f1.mean(), "cv_log_loss_mean": folds.log_loss.mean(),
                     "st_top1_accuracy": st["top1_accuracy"], "st_macro_f1": st["macro_f1"],
                     "st_log_loss": st["log_loss"], "st_route_continuity": st["route_continuity"],
                     "fit_s": info["fit_s"], "st_predict_ms_per_1k_obs": 1e6 * pred_s / len(st_o)})
        log.info("%s: CV top1 %.4f, ST top1 %.4f, ST logloss %.4f", variant, rows[-1]["cv_top1_mean"],
                 st["top1_accuracy"], st["log_loss"])
    comp = pd.DataFrame(rows)
    comp.to_csv(bo_dir / "tuned_vs_untuned.csv", index=False)
    pd.concat(fold_tabs).to_csv(bo_dir / "tuned_vs_untuned_folds.csv", index=False)

    chosen = comp.sort_values("cv_top1_mean", ascending=False).iloc[0]
    model, params, info = fitted[chosen.variant]
    card = {"model_name": name, "variant": chosen.variant, "params": params, "seed": seed,
            "trained_on": "train partition (80 trips), obs_stride=%d" % tc["obs_stride"],
            "fit_rows": info["fit_rows"], "selection_rule": "higher CV top-1 accuracy (train GroupKFold)",
            "cv_top1_mean": float(chosen.cv_top1_mean), "st_top1_accuracy": float(chosen.st_top1_accuracy),
            "st_log_loss": float(chosen.st_log_loss), "config_candidates": cfg["candidates"],
            "config_features": cfg["features"]}
    path = save_bundle(models_dir(cfg) / "final_model.joblib", model, name, params, card)
    (m_dir / "final_model_card.json").write_text(json.dumps(card, indent=1, default=str))
    print(comp.drop(columns=["params"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    size_mb = path.stat().st_size / 2**20
    log.info("exported %s (%s) to %s (%.1f MB)", name, chosen.variant, path, size_mb)
    if size_mb > 90:
        log.warning("model file is %.0f MB - GitHub rejects files > 100 MB. Track it with Git LFS "
                    "(git lfs track 'models/*.joblib') or note in the report how to regenerate it.", size_mb)


if __name__ == "__main__":
    main()

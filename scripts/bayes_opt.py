"""Bayesian hyper-parameter optimisation with Optuna TPE (owner: Person 2).

* Model: the one chosen by train_compare.py (results/models/selection.json)
  unless ``--model`` is given.
* Objective: mean GroupKFold (trip-level) CV observation-level Top-1 accuracy
  on the 80 TRAINING trips (``bayes_opt.objective``; ``neg_log_loss`` also
  supported).  Student-test and eval-test data are never touched here.
* Sampler: TPE (Tree-structured Parzen Estimator) - a sequential model-based
  (Bayesian) optimiser - with a fixed seed; median pruning after each fold.
* Search record: every trial is stored in an SQLite Optuna study
  (resumable) and exported to CSV with params, per-trial metrics and cost.

Outputs (results/bayes_opt/):
  optuna_<model>.db, trials_<model>.csv, best_params_<model>.json,
  search_space_<model>.json, history_<model>.png

Usage:  python scripts/bayes_opt.py [--model lightgbm] [--n-trials 30]
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

from pathsnap.config import load_config, results_dir
from pathsnap.dataset import build_partition_tables
from pathsnap.models.cv import group_cv
from pathsnap.models.zoo import get_spec

log = logging.getLogger("bayes_opt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    bo, tc, seed = cfg["bayes_opt"], cfg["training"], cfg["seed"]
    out = results_dir(cfg, "bayes_opt")
    name = args.model
    if name is None:
        sel = results_dir(cfg, "models") / "selection.json"
        if not sel.exists():
            raise SystemExit("No --model given and results/models/selection.json missing; run train_compare.py")
        name = json.loads(sel.read_text())["selected_model"]
    spec = get_spec(name)
    if spec.search_space is None:
        raise SystemExit(f"model {name} has no search space")
    n_trials = args.n_trials or bo["n_trials"]
    objective_name = bo["objective"]

    cands, obs = build_partition_tables(cfg, "train")
    (out / f"search_space_{name}.json").write_text(json.dumps({
        "model": name, "objective": objective_name, "cv": f"GroupKFold({bo['cv_folds']}) on trip_id, train only",
        "obs_stride": bo["obs_stride"], "sampler": "TPESampler(seed=%d)" % seed,
        "pruner": "MedianPruner(n_startup_trials=5)", "default_params": spec.default_params,
        "search_space_source": inspect.getsource(spec.search_space)}, indent=1))

    def objective(trial: optuna.Trial) -> float:
        params = spec.search_space(trial)
        t0 = time.perf_counter()

        def on_fold(k, m):
            val = m["top1_accuracy"] if objective_name == "top1_accuracy" else -m["log_loss"]
            trial.report(val, k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        folds = group_cv(name, params, cands, obs, bo["cv_folds"], bo["obs_stride"], seed, tc["threshold"],
                         args.n_jobs, tc["slow_model_max_rows"], on_fold=on_fold)
        for k in ("top1_accuracy", "macro_f1", "log_loss", "route_continuity", "fit_s", "predict_ms_per_1k_obs"):
            trial.set_user_attr(f"cv_{k}_mean", float(folds[k].mean()))
        trial.set_user_attr("cv_top1_std", float(folds.top1_accuracy.std()))
        trial.set_user_attr("elapsed_s", time.perf_counter() - t0)
        return float(folds.top1_accuracy.mean()) if objective_name == "top1_accuracy" else -float(folds.log_loss.mean())

    storage = f"sqlite:///{(out / f'optuna_{name}.db').as_posix()}"
    study = optuna.create_study(study_name=f"pathsnap_{name}", storage=storage, load_if_exists=True,
                                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, n_trials - done)
    log.info("study %s: %d finished trials, running %d more", study.study_name, done, remaining)
    t0 = time.perf_counter()
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=bo.get("timeout_s"), gc_after_trial=True)
    total = time.perf_counter() - t0

    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs",
                                       "datetime_start", "datetime_complete", "duration"))
    df.to_csv(out / f"trials_{name}.csv", index=False)
    best = study.best_trial
    best_params = spec.search_space(optuna.trial.FixedTrial(best.params))
    (out / f"best_params_{name}.json").write_text(json.dumps({
        "model": name, "objective": objective_name, "best_value": best.value, "best_trial": best.number,
        "best_params": best_params, "raw_trial_params": best.params, "user_attrs": best.user_attrs,
        "n_trials_total": len(study.trials),
        "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "wall_time_this_run_s": total}, indent=1, default=str))

    comp = df[df.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(comp.number, comp.value, "o", label="trial value")
    ax.plot(comp.number, comp.value.cummax(), "-", label="best so far")
    ax.set(xlabel="trial", ylabel=objective_name, title=f"Optuna TPE history - {name}")
    ax.legend(); fig.tight_layout(); fig.savefig(out / f"history_{name}.png", dpi=120); plt.close(fig)
    log.info("best %s = %.4f with %s", objective_name, best.value, best_params)


if __name__ == "__main__":
    main()

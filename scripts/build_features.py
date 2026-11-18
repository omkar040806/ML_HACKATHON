"""Build and cache candidate/feature tables for train and student_test (owner: Person 1).

Usage:  python scripts/build_features.py [--n-jobs 2] [--no-cache]
Output: data/processed/{train,student_test}_{cands,obs}.pkl  (git-ignored)
        results/eda/feature_table_summary.csv                (committed)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pathsnap.config import load_config, results_dir
from pathsnap.dataset import build_partition_tables
from pathsnap.features import FEATURE_COLUMNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    rows = []
    for part in ("train", "student_test"):
        cands, obs = build_partition_tables(cfg, part, use_cache=not args.no_cache, n_jobs=args.n_jobs)
        lab = obs[obs.labeled]
        rows.append({"partition": part, "trips": obs.trip_id.nunique(), "observations": len(obs),
                     "labelled_obs": len(lab), "candidate_rows": len(cands),
                     "positive_rows": int(cands.label.sum()),
                     "candidate_miss_rate": float(1 - lab.ref_in_cands.mean()),
                     "obs_without_candidates": int((obs.n_cands == 0).sum())})
        desc = cands[FEATURE_COLUMNS].describe().T
        desc.to_csv(results_dir(cfg, "eda") / f"feature_stats_{part}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(results_dir(cfg, "eda") / "feature_table_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

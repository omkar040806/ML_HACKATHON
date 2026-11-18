"""Configuration loading (owner: Person 1).

The YAML file in ``configs/pathsnap.yaml`` is the single source of settings.
A few environment variables override paths so that the same code runs on
both laptops without edits:

* ``PATHSNAP_CONFIG``    - alternative YAML file
* ``PATHSNAP_DATA_DIR``  - folder that holds ``train/``, ``student_test/`` ...
* ``PATHSNAP_EVAL_DIR``  - folder whose trip sub-folders the API serves
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pathsnap.yaml"


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the YAML config, apply env overrides, resolve paths to absolute."""
    cfg_path = Path(path or os.environ.get("PATHSNAP_CONFIG", DEFAULT_CONFIG))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = copy.deepcopy(cfg)
    if overrides:
        _deep_update(cfg, overrides)

    paths = cfg.setdefault("paths", {})
    if os.environ.get("PATHSNAP_DATA_DIR"):
        paths["data_dir"] = os.environ["PATHSNAP_DATA_DIR"]
    for key in ("data_dir", "processed_dir", "results_dir", "models_dir"):
        if key in paths:
            paths[key] = str(_resolve(paths[key]))
    return cfg


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def partition_dir(cfg: dict, partition: str) -> Path:
    """Absolute folder of a partition (``train``, ``student_test``, ``eval_test``)."""
    if partition == "eval_test" and os.environ.get("PATHSNAP_EVAL_DIR"):
        return Path(os.environ["PATHSNAP_EVAL_DIR"]).resolve()
    name = cfg.get("partitions", {}).get(partition, partition)
    return Path(cfg["paths"]["data_dir"]) / name


def results_dir(cfg: dict, *sub: str) -> Path:
    p = Path(cfg["paths"]["results_dir"]).joinpath(*sub)
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["processed_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["models_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p

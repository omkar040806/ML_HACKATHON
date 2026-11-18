"""Exported model bundle (owner: Person 2; consumed by Person 3 and Person 4).

``models/final_model.joblib`` holds a :class:`ModelBundle`.  The bundle
contains the fitted estimator (including any fitted scaler inside its
sklearn Pipeline), the exact feature column order, the hyper-parameters and
provenance metadata.  Consumers must call :meth:`ModelBundle.predict_proba`
rather than the raw estimator so that feature order is always checked.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features import FEATURE_COLUMNS

BUNDLE_VERSION = 1


@dataclass
class ModelBundle:
    model: object
    model_name: str
    params: dict
    feature_columns: list[str]
    metadata: dict = field(default_factory=dict)
    version: int = BUNDLE_VERSION

    def predict_proba(self, cands: pd.DataFrame) -> np.ndarray:
        """P(candidate is the reference arc) for every row of ``cands``."""
        missing = [c for c in self.feature_columns if c not in cands.columns]
        if missing:
            raise KeyError(f"candidate table is missing features: {missing}")
        if len(cands) == 0:
            return np.zeros(0)
        return self.model.predict_proba(cands[self.feature_columns].to_numpy(np.float32))[:, 1]


def save_bundle(path: str | Path, model, model_name: str, params: dict, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    meta.setdefault("created_utc", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    bundle = ModelBundle(model=model, model_name=model_name, params=params,
                         feature_columns=list(FEATURE_COLUMNS), metadata=meta)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}. Run scripts/finalize_model.py first.")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    if bundle.feature_columns != list(FEATURE_COLUMNS):
        raise ValueError("Model was trained with a different feature set than pathsnap.features.FEATURE_COLUMNS; "
                         "rebuild features and retrain.")
    return bundle

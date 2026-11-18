"""Tests for Person 2 modules: zoo, metrics, cv, artifact."""
import numpy as np
import pandas as pd
import pytest

from pathsnap.dataset import build_partition_tables
from pathsnap.models.artifact import load_bundle, save_bundle
from pathsnap.models.cv import fit, group_folds, group_cv, predict_proba, training_rows
from pathsnap.models.metrics import argmax_per_obs, candidate_metrics, sequence_continuity, top1_and_continuity
from pathsnap.models.zoo import MODEL_SPECS


@pytest.fixture
def tables(synthetic_partition, cfg, tmp_path):
    root, _ = synthetic_partition
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"], data_dir=str(root.parent), processed_dir=str(tmp_path / "proc"))
    cfg["partitions"] = {"train": root.name}
    return build_partition_tables(cfg, "train")


def test_top1_counts_missing_reference_as_error():
    cands = pd.DataFrame({"trip_id": ["a"] * 4, "obs_idx": [0, 0, 1, 1], "arc_id": [5, 6, 5, 7],
                          "arc_from": [0, 1, 0, 2], "arc_to": [1, 2, 1, 3], "label": [1, 0, 0, 0],
                          "obs_labeled": [True] * 4})
    obs = pd.DataFrame({"trip_id": ["a", "a"], "obs_idx": [0, 1], "ref_arc": [5, 9], "labeled": [True, True],
                        "ref_in_cands": [True, False]})
    m = top1_and_continuity(cands, obs, np.array([0.9, 0.1, 0.6, 0.4]))
    assert m["top1_accuracy"] == 0.5 and m["top1_given_candidate"] == 1.0 and m["candidate_miss_rate"] == 0.5


def test_argmax_and_continuity():
    cands = pd.DataFrame({"trip_id": ["a"] * 3, "obs_idx": [0, 1, 2], "arc_id": [1, 2, 3],
                          "arc_from": [0, 1, 5], "arc_to": [1, 2, 6]})
    best = argmax_per_obs(cands, np.ones(3))
    frac, ok, n = sequence_continuity(best.arc_id, best.arc_from, best.arc_to)
    assert (ok, n) == (1, 2) and frac == 0.5


def test_candidate_metrics_shape():
    m = candidate_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.4, 0.6]))
    assert m["confusion"] == [[1, 1], [1, 1]] and 0 <= m["log_loss"]


def test_group_folds_disjoint(tables):
    cands, _ = tables
    for _, tr, va in group_folds(cands, 3, 0):
        assert not (tr & va) and va


@pytest.mark.parametrize("name", sorted(MODEL_SPECS))
def test_every_model_fits_and_predicts(tables, name):
    cands, _ = tables
    rows = training_rows(cands, 2)
    model, params, info = fit(name, None, rows, seed=0, n_jobs=1, slow_cap=5000)
    p, _ = predict_proba(model, cands)
    assert p.shape == (len(cands),) and np.all((p >= 0) & (p <= 1)) and info["fit_rows"] > 0


def test_group_cv_runs(tables):
    cands, obs = tables
    folds = group_cv("logreg", None, cands, obs, n_splits=3, stride=1, seed=0, n_jobs=1)
    assert len(folds) == 3 and folds.top1_accuracy.between(0, 1).all()


def test_bundle_roundtrip(tables, tmp_path):
    cands, _ = tables
    model, params, _ = fit("logreg", None, training_rows(cands, 1), seed=0, n_jobs=1)
    path = save_bundle(tmp_path / "m.joblib", model, "logreg", params, {"note": "test"})
    b = load_bundle(path)
    assert np.allclose(b.predict_proba(cands), model.predict_proba(
        cands[b.feature_columns].to_numpy(np.float32))[:, 1])
    with pytest.raises(KeyError):
        b.predict_proba(cands.drop(columns=["dist_m"]))

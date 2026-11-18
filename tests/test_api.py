"""Tests for Person 4 modules: pipeline, API, Folium maps."""
import json

import numpy as np
import pytest

from pathsnap.dataset import build_partition_tables
from pathsnap.io import load_trip_input
from pathsnap.models.artifact import save_bundle
from pathsnap.models.cv import fit, training_rows


@pytest.fixture
def served(synthetic_partition, cfg, tmp_path, monkeypatch):
    """Synthetic eval folder + tiny model bundle wired in through env vars."""
    root, infos = synthetic_partition
    c = dict(cfg)
    c["paths"] = dict(cfg["paths"], data_dir=str(root.parent), processed_dir=str(tmp_path / "proc"))
    c["partitions"] = {"train": root.name}
    cands, _ = build_partition_tables(c, "train")
    model, params, _ = fit("logreg", None, training_rows(cands, 1), seed=0, n_jobs=1)
    mp = save_bundle(tmp_path / "model.joblib", model, "logreg", params)
    (infos[1]["dir"] / f"{infos[1]['dir'].name}.route").unlink()   # one trip without reference
    monkeypatch.setenv("PATHSNAP_EVAL_DIR", str(root))
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", str(mp))
    return root, infos, mp


def test_pipeline_predicts_without_route(served, cfg):
    from pathsnap.pipeline import PathSnapPipeline
    root, infos, mp = served
    pipe = PathSnapPipeline(cfg, model_path=mp)
    trip = load_trip_input(infos[1]["dir"])          # this trip has no .route file
    res, pt = pipe.predict(trip)
    assert (res.hmm_obs_arc >= 0).mean() > 0.95 and len(res.hmm_route) >= 7
    with pytest.raises(TypeError):
        pipe.predict(str(infos[1]["dir"]))


def test_api_endpoints(served):
    from fastapi.testclient import TestClient
    from api.main import create_app
    root, infos, _ = served
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "ok"
        tl = client.get("/trips").json()
        assert len(tl["trips"]) == len(infos)
        assert "route" not in json.dumps(tl)
        r = client.post("/predict", json={"trip_id": infos[0]["dir"].name})
        assert r.status_code == 200
        body = r.json()
        assert body["evaluation"]["available"] and body["evaluation"]["hmm"]["pointwise_accuracy"] > 0.8
        assert body["predicted_route"] and body["gps"]
        # reference geometry/ids must never be in the response
        text = json.dumps(body)
        for k in ("ref_arc", "reference_route", "route_pos", "ref_dist"):
            assert k not in text
        r2 = client.post("/predict", json={"trip_id": infos[1]["dir"].name})
        assert r2.status_code == 200 and not r2.json()["evaluation"]["available"]
        assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
        assert client.post("/predict", json={"trip_id": "../etc"}).status_code == 422


def test_folium_map_builds():
    from pathsnap.viz.folium_maps import build_result_map
    gps = [[50.0, 10.0, 1.0], [50.001, 10.001, 2.0]]
    m = build_result_map(gps, hmm_route=[[[50.0, 10.0], [50.001, 10.001]]],
                         network_segments=[[[50.0, 10.0], [50.0, 10.001]]])
    html = m.get_root().render()
    assert "Predicted route" in html and "Reference route" not in html
    with pytest.raises(ValueError):
        build_result_map([])

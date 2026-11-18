"""Tests for Person 1 modules: io, geo, graph, labels, candidates, features, dataset."""
import numpy as np
import pytest

from pathsnap.candidates import generate_candidates
from pathsnap.dataset import attach_labels, build_partition_tables, prepare_trip, process_trip_dir
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.geo import LocalProjection, angle_diff_deg, haversine_m, point_segment_distance
from pathsnap.graph import RoadGraph
from pathsnap.io import TripFormatError, clean_track, load_reference_route, load_trip_input, trip_path
from pathsnap.labels import monotone_align


# ---------------------------------------------------------------- geo
@pytest.mark.parametrize("lat0", [0.0, 45.0, 61.0])
def test_projection_matches_haversine(lat0):
    p = LocalProjection(10.0, lat0)
    lon = np.array([10.0, 10.2, 9.85])
    lat = np.array([lat0, lat0 + 0.15, lat0 - 0.1])
    x, y = p.forward(lon, lat)
    for i in range(3):
        for j in range(i + 1, 3):
            planar = np.hypot(x[i] - x[j], y[i] - y[j])
            hv = haversine_m(lon[i], lat[i], lon[j], lat[j])
            assert abs(planar - hv) / hv < 0.01


def test_projection_roundtrip():
    p = LocalProjection(-80.4, 37.1)
    lon, lat = p.inverse(*p.forward([-80.5, -80.3], [37.0, 37.2]))
    assert np.allclose(lon, [-80.5, -80.3]) and np.allclose(lat, [37.0, 37.2])


def test_point_segment_distance():
    d, t, fx, fy = point_segment_distance([5, -3, 15, 5], [5, 0, 0, 0], 0, 0, 10, 0)
    assert np.allclose(d, [5, 3, 5, 0]) and np.allclose(t, [0.5, 0, 1, 0.5])
    d0, t0, *_ = point_segment_distance(3, 4, 0, 0, 0, 0)  # zero-length segment
    assert np.isclose(d0, 5) and t0 == 0


def test_angle_diff():
    assert np.allclose(angle_diff_deg([350, 10, 90], [10, 350, 270]), [20, 20, 180])


# ---------------------------------------------------------------- io
def test_load_trip_never_needs_route(tmp_path):
    from conftest import make_synthetic_trip
    info = make_synthetic_trip(tmp_path, with_route=False)
    trip = load_trip_input(info["dir"])
    assert trip.track.shape[1] == 3 and trip.n_obs > 10
    assert load_reference_route(info["dir"]) is None


def test_clean_track_drops_bad_rows():
    tr = np.array([[1, 1, 3], [np.nan, 1, 4], [1, 1, 1], [500, 1, 2], [1, 1, 3]], float)
    out, kept, issues = clean_track(tr)
    assert out[:, 2].tolist() == [1, 3] and len(issues) >= 3 and kept.tolist() == [2, 0]


def test_trip_path_rejects_traversal(tmp_path):
    with pytest.raises(TripFormatError):
        trip_path(tmp_path, "../secret")


def test_bad_arc_reference(tmp_path, synthetic_trip):
    d = synthetic_trip["dir"]
    (d / f"{d.name}.arcs").write_text("0\t99999\n")
    with pytest.raises(TripFormatError):
        load_trip_input(d)


# ---------------------------------------------------------------- graph
def test_graph_shortest_path_and_canonical(synthetic_trip):
    trip = load_trip_input(synthetic_trip["dir"])
    g = RoadGraph.from_trip(trip)
    route = synthetic_trip["route"]
    src, dst = g.arc_from[route[0]], g.arc_to[route[-1]]
    path = g.shortest_path_arcs(int(src), int(dst), 5000)
    assert path is not None and np.isclose(g.arc_len[path].sum(), 700, atol=5)
    assert g.network_distance(int(src), int(dst), 100) == np.inf
    assert (g.reverse_arc[route] >= 0).all()           # grid streets are two-way
    st = g.component_stats()
    assert st["n_weak_components"] == 1 and st["n_duplicate_arcs"] == 0


def test_duplicate_arcs_are_canonicalised():
    nodes = np.array([[10.0, 50.0], [10.001, 50.0]])
    arcs = np.array([[0, 1], [0, 1], [1, 0]])
    g = RoadGraph(nodes, arcs, LocalProjection(10.0, 50.0))
    assert g.canonical.tolist() == [0, 0, 2] and g.reverse_arc[0] == 2


# ---------------------------------------------------------------- labels
def test_monotone_align_is_monotone_and_optimal():
    d = np.array([[0, 5, 9], [4, 1, 9], [9, 0.5, 2], [9, 9, 0]], float)
    pos = monotone_align(d)
    assert pos.tolist() == [0, 1, 1, 2]


def test_labels_recover_true_arcs(synthetic_trip, cfg):
    pt = process_trip_dir(synthetic_trip["dir"], cfg, with_labels=True)
    acc = np.mean(pt.obs["ref_arc"].to_numpy() == synthetic_trip["true_arc"])
    assert acc > 0.9
    assert pt.obs["ref_in_cands"].mean() > 0.98


# ---------------------------------------------------------------- candidates / features
def test_candidates_respect_radius_and_k(synthetic_trip):
    trip = load_trip_input(synthetic_trip["dir"])
    g = RoadGraph.from_trip(trip)
    x, y = g.proj.forward(trip.track[:, 0], trip.track[:, 1])
    c = generate_candidates(x, y, g, radius_m=30, max_candidates=4, sample_spacing_m=7)
    assert (c.dist_m <= 30 + 1e-9).all() and (c.groupby("obs_idx").size() <= 4).all()
    assert c.groupby("obs_idx")["dist_m"].apply(lambda s: s.is_monotonic_increasing).all()
    assert g.is_canonical[c.arc_id].all()


def test_features_complete_and_finite(synthetic_trip, cfg):
    pt = prepare_trip(load_trip_input(synthetic_trip["dir"]), cfg)
    assert set(FEATURE_COLUMNS) <= set(pt.cands.columns)
    assert np.isfinite(pt.cands[FEATURE_COLUMNS].to_numpy(float)).all()
    assert "label" not in pt.cands.columns          # inference path has no labels


def test_heading_separates_two_way_arcs(synthetic_trip, cfg):
    pt = attach_labels(prepare_trip(load_trip_input(synthetic_trip["dir"]), cfg),
                       synthetic_trip["route"], cfg)
    c = pt.cands[pt.cands.heading_valid == 1]
    assert c.loc[c.label == 1, "heading_cos"].mean() > 0.9


def test_partition_tables_forbid_eval(cfg):
    with pytest.raises(PermissionError):
        build_partition_tables(cfg, "eval_test")
    with pytest.raises(PermissionError):           # the real folder name is blocked too
        build_partition_tables(cfg, cfg["partitions"]["eval_test"])


def test_partition_tables_build(synthetic_partition, cfg, tmp_path, monkeypatch):
    root, infos = synthetic_partition
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"], data_dir=str(root.parent), processed_dir=str(tmp_path / "proc"))
    cfg["partitions"] = {"train": root.name}
    cands, obs = build_partition_tables(cfg, "train")
    assert cands.trip_id.nunique() == len(infos) and {"label", "obs_labeled"} <= set(cands.columns)

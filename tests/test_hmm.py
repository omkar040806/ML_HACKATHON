"""Tests for Person 3 modules: emission, transition, viterbi, decoder, route metrics."""
import numpy as np
import pytest

from pathsnap.dataset import attach_labels, prepare_trip
from pathsnap.graph import RoadGraph
from pathsnap.hmm.decoder import HMMMapMatcher, classifier_only, collapse, fill_route
from pathsnap.hmm.emission import log_emission
from pathsnap.hmm.route_metrics import continuity, edit_similarity, evaluate_prediction, route_set_metrics
from pathsnap.hmm.transition import TransitionModel
from pathsnap.hmm.viterbi import viterbi
from pathsnap.io import load_trip_input


def test_viterbi_prefers_consistent_path():
    em = [np.log([0.6, 0.4]), np.log([0.4, 0.6]), np.log([0.6, 0.4])]
    stay = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    path, starts = viterbi(em, [None, stay, stay])
    assert path == [0, 0, 0] and starts == [0]


def test_viterbi_breaks_on_impossible_transition():
    em = [np.log([0.9, 0.1]), np.log([0.2, 0.8])]
    path, starts = viterbi(em, [None, np.full((2, 2), -np.inf)])
    assert starts == [0, 1] and path == [0, 1]


def test_viterbi_forced_break():
    em = [np.array([0.0, -1.0]), np.array([-1.0, 0.0])]
    path, starts = viterbi(em, lambda s: None)
    assert path == [0, 1] and starts == [0, 1]


def test_emission_modes():
    d, p = np.array([0.0, 10.0]), np.array([0.9, 0.0])
    assert log_emission(d, p, "classifier", min_prob=1e-6)[1] == pytest.approx(np.log(1e-6))
    assert log_emission(d, None, "gaussian", sigma_m=10)[1] == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        log_emission(d, None, "classifier")


@pytest.fixture
def grid(synthetic_trip):
    trip = load_trip_input(synthetic_trip["dir"])
    return RoadGraph.from_trip(trip), synthetic_trip


def test_transition_penalises_wrong_direction(grid):
    g, info = grid
    a = info["route"][0]                     # eastbound arc
    rev = int(g.reverse_arc[a])
    tm = TransitionModel(g, beta_m=5)
    rd = tm.route_distances(np.array([a]), np.array([0.2]), np.array([a, rev]), np.array([0.3, 0.7]), gc=10, dt=1)
    assert rd[0, 0] == pytest.approx(10, abs=0.5) and rd[0, 1] > 50


def test_decoder_recovers_route_with_perfect_probabilities(grid, cfg):
    g, info = grid
    pt = attach_labels(prepare_trip(load_trip_input(info["dir"]), cfg), info["route"], cfg)
    prob = np.where(pt.cands.label == 1, 0.7, 0.3)
    r = HMMMapMatcher(pt.graph, cfg["hmm"]).decode(pt.cands, prob, pt.x, pt.y, pt.t)
    m = evaluate_prediction(r.obs_arc, r.route, pt.obs.ref_arc.values, pt.obs.labeled.values, info["route"], pt.graph)
    assert m["pointwise_accuracy"] > 0.95 and m["route_continuity"] == 1.0 and m["route_f1"] > 0.9


def test_hmm_beats_noisy_classifier(grid, cfg):
    g, info = grid
    pt = attach_labels(prepare_trip(load_trip_input(info["dir"]), cfg), info["route"], cfg)
    rng = np.random.default_rng(0)
    prob = np.clip(np.where(pt.cands.label == 1, 0.55, 0.35) + rng.normal(0, 0.2, len(pt.cands)), 0.01, 0.99)
    co = classifier_only(pt.cands, prob, len(pt.x))
    r = HMMMapMatcher(pt.graph, cfg["hmm"]).decode(pt.cands, prob, pt.x, pt.y, pt.t)
    lab, ref = pt.obs.labeled.values, pt.obs.ref_arc.values
    assert (r.obs_arc[lab] == ref[lab]).mean() >= (co[lab] == ref[lab]).mean()
    assert continuity(r.route, pt.graph) >= continuity(collapse(co), pt.graph)


def test_gap_forces_segment_break(tmp_path, cfg):
    from conftest import make_synthetic_trip
    info = make_synthetic_trip(tmp_path, gap=True)
    pt = prepare_trip(load_trip_input(info["dir"]), cfg)
    r = HMMMapMatcher(pt.graph, cfg["hmm"]).decode(pt.cands, None, pt.x, pt.y, pt.t, emission_mode="gaussian")
    assert r.n_segments >= 2 and continuity(r.route, pt.graph) == 1.0   # gap filled by shortest path


def test_fill_route_connects(grid):
    g, info = grid
    r = info["route"]
    filled, unf = fill_route([int(r[0]), int(r[3])], g, 5000)
    assert unf == 0 and continuity(filled, g) == 1.0 and filled[0] == r[0] and filled[-1] == r[3]


def test_route_metrics_and_edit_similarity(grid):
    g, info = grid
    r = list(info["route"])
    assert route_set_metrics(r, r, g)["route_f1"] == 1.0
    assert route_set_metrics(r, r, g)["route_mismatch_fraction"] == 0.0
    assert edit_similarity([1, 2, 3], [1, 2, 3]) == 1.0
    assert edit_similarity([1, 2, 3], [1, 3]) == pytest.approx(2 / 3)
    assert edit_similarity([ord(c) for c in "kitten"], [ord(c) for c in "sitting"]) == pytest.approx(1 - 3 / 7)
    assert edit_similarity([1, 2, 3], [4, 5, 6]) == pytest.approx(0.0)

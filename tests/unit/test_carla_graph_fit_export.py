"""Scientific integrity fixtures for the FIT graph exporter; no fitting."""
import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from cognix.adapters.carla import cache_builder as cb
from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla import normality, real_agents


@pytest.fixture
def splits():
    return json.loads((__import__("pathlib").Path(__file__).resolve().parents[2] /
        "reports/carla_gat_preregistration_v1/split_manifest.json").read_text())


@pytest.fixture
def frozen_state():
    params, handoff = {}, {"calibration_parameters_raw": {}, "calibration_audit": {}}
    for name in ge.NODE_ORDER:
        lo, hi = ge.OFFSETS[name.lower()]
        for k in range(5):
            params[f"{name.lower()}_{k}_mean"] = np.full(hi - lo, 0.02 * k)
            params[f"{name.lower()}_{k}_precision"] = np.eye(hi - lo) * (k + 1)
            params[f"{name.lower()}_{k}_distance_scale"] = np.array(1.0 + k)
        handoff["calibration_parameters_raw"][name] = {"slope": 3.0, "intercept": -1.0, "scientific_valid": True}
        handoff["calibration_audit"][name] = {"finite_optimum": True, "status": "fixture_frozen_valid",
            "nll_initial": 1.0, "nll_final": 0.5, "n_iter": 2, "converged": True}
    return params, handoff


@pytest.fixture
def fixture_data(tmp_path, splits, frozen_state):
    rng = np.random.Generator(np.random.PCG64(7))
    n = 19
    for directory in ("rgb-front", "segmentation-front"):
        (tmp_path / directory).mkdir()
    for t in range(1, n + 1):
        Image.fromarray(rng.integers(20, 220, size=(16, 16, 3), dtype=np.uint8)).save(tmp_path / "rgb-front" / f"{t:06d}.jpg")
        seg = rng.integers(1, 5, size=(16, 16, 3), dtype=np.uint8)
        if t == 2:
            seg[:] = 1  # selected segmentation recipe has no compact effect
        Image.fromarray(seg).save(tmp_path / "segmentation-front" / f"{t:06d}.png")
    imu = rng.normal(size=(n + 1, 3))
    parents = (np.array([cb.extract_camera_features(tmp_path / "rgb-front" / f"{t:06d}.jpg") for t in range(1, n + 1)]),
        np.array([cb.extract_seg_features(tmp_path / "segmentation-front" / f"{t:06d}.png") for t in range(1, n + 1)]),
        rng.normal(size=(n, 8)),
        np.array([cb.imu_window_features(imu[max(0, t - 11):t + 1]) for t in range(1, n + 1)]))
    agents = ge.restore_agents(*frozen_state)
    clean = np.array([[ge.recompute_node(agents[name], parents[("camera", "seg", "gnss", "imu").index(name.lower())][i])
        for name in ge.NODE_ORDER] for i in range(n)])
    return {"scenario_id": splits["GRAPH_TRAIN"][0], "ticks": np.arange(1, n + 1),
        "parent_features": parents, "clean_nodes": clean, "raw": {"scenario_path": tmp_path, "imu": imu},
        "agents": agents, "splits": splits}


@pytest.fixture
def result(fixture_data):
    return ge.build_scenario_pairs(**fixture_data)


def assembled(result, fixture_data):
    sid = fixture_data["scenario_id"]
    return ge.assemble_artifact({sid: result}, {"protocol_sha256": ge.PROTOCOL_SHA256},
        [{"scenario_id": sid, "town": sid.split("/")[0], "partition": "FIT_NORMAL",
          "graph_split": ge.frozen_split(sid, fixture_data["splits"]), "source_hash_bundle_sha256": "a" * 64}],
        {n: {"scientific_valid": True} for n in ge.NODE_ORDER})


def test_node_order_and_precision(result, fixture_data):
    assert ge.NODE_ORDER == ("Camera", "IMU", "Seg")
    assert result["node_features"].dtype == np.float64
    for r, node in zip(result["rows"], result["node_features"]):
        if not r["is_pseudo"]:
            assert node.tobytes() == fixture_data["clean_nodes"][r["parent_tick"] - 1].tobytes()


def test_feature_order(frozen_state):
    agent = ge.restore_agents(*frozen_state)["Camera"]
    u = agent.estimate_uncertainty(np.arange(18) * 0.01)
    assert ge.FEATURE_ORDER == ("prob_normal", "epistemic", "aleatoric")
    assert ge.recompute_node(agent, np.arange(18) * 0.01).tolist() == [u.prediction, u.epistemic, u.aleatoric]


def test_fixed_adjacency():
    assert ge.ADJACENCY.tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert ge.ADJACENCY.dtype == np.uint8
    assert not np.diag(ge.ADJACENCY).any()


@pytest.mark.parametrize("is_pseudo,target", [(False, 1), (True, 0)])
def test_targets(result, is_pseudo, target):
    assert all(r["target_normal"] == target for r in result["rows"] if r["is_pseudo"] == is_pseudo)


@pytest.mark.parametrize("role", ["GRAPH_TRAIN", "GRAPH_VALIDATION"])
def test_all_frozen_fit_scenarios(splits, role):
    for sid in splits[role]:
        assert ge.frozen_split(sid, splits) == role


@pytest.mark.parametrize("sid", ["Town01/scenario-1", "Town01/scenario-4", "Town01/scenario-8", "Town01/scenario-9", "Town02/scenario-3", "Town99/scenario-1"])
def test_cal_and_unknown_rejected_before_raw_access(fixture_data, sid, monkeypatch):
    data = {**fixture_data, "scenario_id": sid}
    monkeypatch.setattr(cb, "_pseudo_for_scenario", lambda *a: pytest.fail("CAL reached pseudo path"))
    with pytest.raises(ge.GraphExportError):
        ge.build_scenario_pairs(**data)


@pytest.mark.parametrize("tick", [1, 2, 3, 4, 5, 6, 2999])
def test_recipe_rotation(tick):
    assert ge.selected_recipe(tick) == ge.RECIPES[tick % 5]


def test_deterministic_seed_and_replay(result, fixture_data):
    replay = ge.build_scenario_pairs(**fixture_data)
    assert replay["rows"] == result["rows"]
    assert replay["skips"] == result["skips"]
    assert replay["node_features"].tobytes() == result["node_features"].tobytes()
    assert all(r["generator_seed"] == r["parent_tick"] for r in result["rows"] if r["is_pseudo"])


def test_exactly_one_corrupted_modality_and_untouched_nodes(result):
    for i in range(0, len(result["rows"]), 2):
        pseudo = result["rows"][i + 1]
        j = ge.NODE_ORDER.index(pseudo["corrupted_agent"])
        for k in set(range(3)) - {j}:
            assert result["node_features"][i, k].tobytes() == result["node_features"][i + 1, k].tobytes()


def test_only_corrupted_node_complete_frozen_path(fixture_data, monkeypatch):
    calls = Counter()
    def no_fit(*a, **kw):
        pytest.fail("an upstream parameter was refitted")
    for cls in (real_agents.RealNormalityAgent, normality.MahalanobisNormality,
                normality.BootstrapNormalityEnsemble, normality.EnsemblePredictiveCalibrator):
        monkeypatch.setattr(cls, "fit", no_fit)
    monkeypatch.setattr(real_agents.RealNormalityAgent, "fit_calibrator", no_fit)
    for name, agent in fixture_data["agents"].items():
        original = agent.estimate_uncertainty
        def wrap(x, original=original, name=name):
            calls[name] += 1
            return original(x)
        monkeypatch.setattr(agent, "estimate_uncertainty", wrap)
    score = normality.MahalanobisNormality.predict_normality
    prob = normality.EnsemblePredictiveCalibrator.prob_normal
    score_calls, probability_calls = [], []
    def member(x, v):
        score_calls.append(v.copy())
        return score(x, v)
    def mapping(x, s, **kw):
        probability_calls.append(s)
        return prob(x, s, **kw)
    monkeypatch.setattr(normality.MahalanobisNormality, "predict_normality", member)
    monkeypatch.setattr(normality.EnsemblePredictiveCalibrator, "prob_normal", mapping)
    result = ge.build_scenario_pairs(**fixture_data)
    expected = Counter(r["corrupted_agent"] for r in result["rows"] if r["is_pseudo"])
    assert calls == expected
    assert len(score_calls) == len(probability_calls) == 5 * sum(expected.values())


def test_no_effect_removes_both_without_replacement(result):
    assert {"scenario_id": result["rows"][0]["scenario_id"], "tick": 2,
        "recipe_id": "seg_region_corruption", "modality": "seg", "reason": "no_effect"} in result["skips"]
    assert not any(r["parent_tick"] == 2 for r in result["rows"])
    assert len(result["rows"]) == 2 * (result["candidate_parents"] - len(result["skips"]))
    assert all(r["recipe_id"] == ge.selected_recipe(r["parent_tick"]) for r in result["rows"] if r["is_pseudo"])


def test_balance_identity_and_split_inheritance(result):
    groups = {}
    for r in result["rows"]:
        groups.setdefault(r["pair_key"], []).append(r)
        assert r["pair_key"] == ge.pair_id(r["scenario_id"], r["parent_tick"])
    for pair in groups.values():
        assert len(pair) == 2 and [r["target_normal"] for r in pair] == [1, 0]
        assert len({r["graph_split"] for r in pair}) == 1
        assert len({(r["scenario_id"], r["parent_tick"]) for r in pair}) == 1


def test_pair_id_canonical_and_collision_safe():
    payload = {"scenario_id": "Town01/scenario-10", "parent_tick": 3, "protocol_sha256": ge.PROTOCOL_SHA256}
    assert ge.pair_id(payload["scenario_id"], 3) == hashlib.sha256(ge.canonical_json(payload)).hexdigest()
    assert ge.pair_id("Town01/scenario-1", 23) != ge.pair_id("Town01/scenario-12", 3)


def test_no_gnss_nodes_or_recipes(result):
    assert "GNSS" not in ge.NODE_ORDER
    assert not any("gnss" in rid for rid in ge.RECIPES)
    assert all(r["corrupted_agent"] != "GNSS" for r in result["rows"])


def test_causal_imu_payload(fixture_data, monkeypatch):
    original = cb._generate
    observed = []
    def generate(payload, rid, sid, tick, seed, severity):
        if rid.startswith("imu"):
            expected = fixture_data["raw"]["imu"][max(0, tick - 11):tick + 1]
            assert np.array_equal(payload["IMU"], expected)
            assert len(payload["IMU"]) <= 12
            observed.append(tick)
        return original(payload, rid, sid, tick, seed, severity)
    monkeypatch.setattr(cb, "_generate", generate)
    result = ge.build_scenario_pairs(**fixture_data)
    assert observed == [int(t) for t in fixture_data["ticks"] if t % 5 in (3, 4)]
    for row in result["rows"]:
        assert row["window_start_tick"] == max(0, row["parent_tick"] - 11)
        if row["is_pseudo"] and row["corrupted_agent"] == "IMU":
            assert row["corruption_start_tick"] == row["window_start_tick"]


def test_schema_hash_determinism_and_roundtrip(result, fixture_data, tmp_path):
    arrays, schema, counts, skips, collisions = assembled(result, fixture_data)
    first = ge.scientific_content_hash(arrays, schema)
    assert first == ge.scientific_content_hash(dict(reversed(list(arrays.items()))), json.loads(ge.canonical_json(schema)))
    for dirname in ("a", "b"):
        ge.write_artifact(tmp_path / dirname, arrays, schema, counts, skips, collisions, {})
    assert (tmp_path / "a" / "schema.json").read_bytes() == (tmp_path / "b" / "schema.json").read_bytes()
    with np.load(tmp_path / "a" / "graphs.npz", allow_pickle=False) as z:
        restored = dict(z)
    ge.validate_artifact(restored, schema)
    assert ge.scientific_content_hash(restored, schema) == first
    changed = {**arrays, "node_features": arrays["node_features"].copy()}
    changed["node_features"][0, 0, 0] = np.nextafter(changed["node_features"][0, 0, 0], 1.0)
    assert ge.scientific_content_hash(changed, schema) != first


@pytest.mark.parametrize("failure", ["invalid", "null_slope", "nan_intercept", "invalid_audit", "bad_member"])
def test_invalid_upstream_mapping_rejected(frozen_state, failure):
    params, handoff = copy.deepcopy(frozen_state)
    mapping = handoff["calibration_parameters_raw"]["Camera"]
    if failure == "invalid":
        mapping["scientific_valid"] = False
    elif failure == "null_slope":
        mapping["slope"] = None
    elif failure == "nan_intercept":
        mapping["intercept"] = np.nan
    elif failure == "invalid_audit":
        handoff["calibration_audit"]["Camera"]["finite_optimum"] = False
    else:
        params["camera_0_mean"][0] = np.nan
    with pytest.raises(ge.GraphExportError):
        ge.restore_agents(params, handoff)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_clean_nodes_rejected(fixture_data, value):
    data = {**fixture_data, "clean_nodes": fixture_data["clean_nodes"].copy()}
    data["clean_nodes"][0, 0, 0] = value
    with pytest.raises(ge.GraphExportError):
        ge.build_scenario_pairs(**data)


def test_no_raw_payload_or_model_state(result, fixture_data):
    arrays, schema, *_ = assembled(result, fixture_data)
    forbidden = ("image", "archive", "raw", "gnss", "precision", "mean", "features_camera", "member_normality")
    assert not any(any(word in key.lower() for word in forbidden) for key in arrays)
    assert all(a.dtype.kind != "O" for a in arrays.values())
    assert schema["raw_data_present"] is False
    assert schema["graph_model_parameters_present"] is False


def test_output_collisions_retained(fixture_data, monkeypatch):
    clean = fixture_data["clean_nodes"].copy()
    clean[:] = [0.5, 0.0, 0.69]
    monkeypatch.setattr(ge, "recompute_node", lambda *a: np.array([0.5, 0.0, 0.69]))
    result = ge.build_scenario_pairs(**{**fixture_data, "clean_nodes": clean})
    assert len(result["collisions"]) == len(result["rows"]) // 2
    assert len(result["skips"]) + len(result["collisions"]) == result["candidate_parents"]


def test_untouched_compact_tampering_fails_closed(fixture_data, monkeypatch):
    original = cb._pseudo_for_scenario
    def tamper(*a):
        meta, feats, attempts, skips = original(*a)
        feats[0][-1] += 1  # first recipe camera; this IMU coordinate is untouched
        return meta, feats, attempts, skips
    monkeypatch.setattr(cb, "_pseudo_for_scenario", tamper)
    with pytest.raises(ge.GraphExportError, match="untouched compact"):
        ge.build_scenario_pairs(**fixture_data)


def test_artifact_untouched_tampering_rejected(result, fixture_data):
    a, schema, *_ = assembled(result, fixture_data)
    a["node_features"][1, 1, 0] += 0.01
    with pytest.raises(ge.GraphExportError, match="untouched node"):
        ge.validate_artifact(a, schema)


def test_preregistered_graph_hash_model_view():
    nodes = np.array([[0.1, 0.01, 0.2], [0.3, 0.04, 0.5], [0.6, 0.07, 0.8]], dtype=np.float64)
    provenance = {"is_pseudo": False, "recipe_id": None}
    h = hashlib.sha256(ge.canonical_json(provenance))
    h.update(nodes.astype("<f4").tobytes())
    h.update(nodes[:, 1].astype("<f8").tobytes())
    h.update(ge.ADJACENCY.tobytes())
    assert ge.graph_content_hash(provenance, nodes) == h.hexdigest()

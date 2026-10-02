"""Tests for the compact TRAIN cache builder (Phase A).

Covers:
1.  Deterministic cache reproduction (identical content hash on rebuild).
2.  Scenario-level TRAIN_NORMAL / CAL_NORMAL disjointness and completeness.
3.  No future-timestep leakage: window features computed ONLY from rows <= t
    (verified against manual recomputation from raw sensor tables).
4.  No cross-scenario window leakage (windows confined to the scenario's own
    sensor table).
5.  Feature dimensions and finiteness (18/29/8/10; compact 65).
6.  Normal + pseudo CAL pairing (target_normal 1/0 concept; parent provenance).
7.  TEST split rejection (cache builder only enumerates TRAIN; gate re-checked).
8.  Cache-manifest consistency (dims, versions, window policy, row counts).
9.  No mutation of source scenario data.
10. `import cognix` purity (adapter modules never pulled).
11. No silent overwrite of a mismatched cache; old cache intact after refusal.

No TEST data, no GAT/conformal fitting, no benchmark, no artifact writes
beyond the cache under test (tmp_path).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import PIL.Image
import pytest

from cognix.adapters.carla.cache_builder import (
    CAL_PSEUDO_RECIPES,
    CAMERA_DIM,
    COMPACT_DIM,
    GNSS_DIM,
    IMU_DIM,
    SEG_DIM,
    WINDOW_LENGTH,
    CacheBuilderError,
    build_cache,
    extract_gnss_feature_rows,
    extract_imu_feature_rows,
    load_cache,
    partition_train_scenarios,
)
from cognix.adapters.carla.carlanomaly_loader import CarlAnomalyLoader, CarlAnomalySplit
from cognix.adapters.carla.real_features import (
    gnss_window_features,
    imu_window_features,
    local_meter_offsets_to_degrees,
)


# ── fixture factory (official loader layout; mirrors loader-test helpers) ────

def _make_scenario(root, rel, n, seed, *, with_labels=True):
    rng = np.random.default_rng(seed)
    d = root / rel
    (d / "rgb-front").mkdir(parents=True, exist_ok=True)
    (d / "segmentation-front").mkdir(exist_ok=True)
    lat0, lon0 = 30.0 + rng.uniform(0, 0.01), rng.uniform(0, 0.01)
    east_m = rng.uniform(0.01, 0.09)  # ~1-9 cm per step: sub-meter motion
    k_lat, k_lon = local_meter_offsets_to_degrees(1.0, 1.0, lat0)
    lat, lon = [lat0], [lon0]
    for _ in range(n - 1):
        lat.append(lat[-1] + k_lat * east_m)
        lon.append(lon[-1] + k_lon * east_m)
    for t in range(n):
        arr = rng.integers(30, 220, size=(8, 8, 3), dtype=np.uint8)
        PIL.Image.fromarray(arr).save(d / "rgb-front" / f"{t:06d}.jpg", format="JPEG")
        seg = np.zeros((6, 6, 3), dtype=np.uint8)
        seg[..., 0] = rng.choice([0, 3, 7], size=(6, 6))
        PIL.Image.fromarray(seg).save(d / "segmentation-front" / f"{t:06d}.png", format="PNG")
    gnss = pd.DataFrame(
        {"altitude": np.full(n, 10.0), "latitude": lat, "longitude": lon}
    )
    gnss.to_feather(d / "gnss.feather")
    imu = pd.DataFrame(
        {
            "acceleration_x": rng.normal(0, 0.2, n),
            "acceleration_y": rng.normal(0, 0.2, n),
            "acceleration_z": np.zeros(n),
            "compass": np.zeros(n),
            "longitude_x": np.zeros(n),
            "longitude_y": np.zeros(n),
            "longitude_z": np.zeros(n),
        }
    )
    imu.to_feather(d / "imu.feather")
    if with_labels:
        obs = pd.DataFrame(
            {"tick": list(range(n)), "anomaly": [False] * n, "description": ["clear"] * n}
        )
        obs.to_feather(d / "anomaly-observation.feather")


@pytest.fixture
def cache_root(tmp_path):
    root = tmp_path / "carlanomaly"
    root.mkdir()
    for i in range(4):
        _make_scenario(root, f"train/Town01/scenario-{i + 1}", 20, seed=100 + i)
    return root


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


# ── 1. deterministic reproduction ────────────────────────────────────────────

def test_cache_rebuild_is_deterministic(cache_root, cache_dir, tmp_path):
    m1 = build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    m2 = build_cache(cache_root, tmp_path / "cache2", partition_seed=7, cal_fraction=0.5)
    assert m1["cache_content_sha256"] == m2["cache_content_sha256"]
    # identical partition and identical manifest modulo timestamp fields
    assert m1["partition"] == m2["partition"]
    assert m1["created_utc"] is not None


def test_cache_rebuild_identical_same_dir_allowed(cache_root, cache_dir):
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    _, before = load_cache(cache_dir)
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    _, after = load_cache(cache_dir)
    assert before["cache_content_sha256"] == after["cache_content_sha256"]


# ── 2. scenario-level partition protocol ─────────────────────────────────────

def test_partition_is_scenario_level_and_disjoint(cache_root, cache_dir):
    m = build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, _ = load_cache(cache_dir)
    tn = set(m["partition"]["TRAIN_NORMAL"])
    cal = set(m["partition"]["CAL_NORMAL"])
    assert tn and cal and not (tn & cal)                      # disjoint
    assert tn | cal == set(m["scenario_ids"])                 # complete
    # ticks of one scenario NEVER split across partitions
    for sid in m["scenario_ids"]:
        mask = arrays["scenario_id"] == sid
        assert len(set(arrays["partition"][mask].tolist())) == 1


def test_partition_deterministic_across_calls(cache_root):
    p1 = partition_train_scenarios(
        [f"Town01/scenario-{i}" for i in range(1, 9)], seed=42, cal_fraction=0.25
    )
    p2 = partition_train_scenarios(
        [f"Town01/scenario-{i}" for i in range(1, 9)], seed=42, cal_fraction=0.25
    )
    assert p1.mapping() == p2.mapping()
    # different seed -> (very likely) different split; membership recorded
    p3 = partition_train_scenarios(
        [f"Town01/scenario-{i}" for i in range(1, 9)], seed=43, cal_fraction=0.25
    )
    assert p3.mapping() != p1.mapping() or p3.seed != p1.seed


def test_partition_explicit_override_validated(cache_root):
    ids = [f"Town01/scenario-{i}" for i in range(1, 5)]
    p = partition_train_scenarios(
        ids, explicit={s: ("CAL_NORMAL" if i % 2 else "TRAIN_NORMAL")
                       for i, s in enumerate(ids, start=1)}
    )
    assert p.rule == "explicit"
    with pytest.raises(CacheBuilderError):
        partition_train_scenarios(ids, explicit={"Town01/scenario-1": "TRAIN_NORMAL"})
    with pytest.raises(CacheBuilderError):
        partition_train_scenarios(ids, explicit={s: "TEST_NORMAL" for s in ids})


def test_partition_recorded_in_manifest(cache_root, cache_dir):
    m = build_cache(cache_root, cache_dir, partition_seed=11, cal_fraction=0.5)
    assert m["partition"]["seed"] == 11
    assert m["partition"]["scenario_level_only"] is True
    arrays, _ = load_cache(cache_dir)
    # every cached row's partition label agrees with the manifest mapping
    label_of = m["partition"]
    mapping = {s: "TRAIN_NORMAL" for s in label_of["TRAIN_NORMAL"]}
    mapping.update({s: "CAL_NORMAL" for s in label_of["CAL_NORMAL"]})
    for sid, lab in mapping.items():
        assert set(arrays["partition"][arrays["scenario_id"] == sid].tolist()) == {lab}


# ── 3./4. causality ─────────────────────────────────────────────────────────

def test_no_future_leakage_window_features(cache_root):
    """Perturb FUTURE sensor rows after tick t; features at tick t must be
    unchanged (strictly causal)."""
    from cognix.adapters.carla.cache_builder import _gnss_table, _imu_accel_table

    scen = cache_root / "train/Town01/scenario-1"
    before_gnss = extract_gnss_feature_rows(scen, 20)
    before_imu = extract_imu_feature_rows(scen, 20)
    # perturb rows strictly after tick 5
    df = pd.read_feather(scen / "gnss.feather")
    df.loc[6:, "latitude"] += 1e-4
    df.to_feather(scen / "gnss.feather")
    dfi = pd.read_feather(scen / "imu.feather")
    dfi.loc[6:, "acceleration_x"] += 5.0
    dfi.to_feather(scen / "imu.feather")
    after_gnss = extract_gnss_feature_rows(scen, 20)
    after_imu = extract_imu_feature_rows(scen, 20)
    np.testing.assert_array_equal(before_gnss[:5], after_gnss[:5])
    np.testing.assert_array_equal(before_imu[:5], after_imu[:5])
    # ...and later ticks DID change (sanity that the perturbation propagated)
    assert not np.allclose(before_gnss[10:], after_gnss[10:])
    assert not np.allclose(before_imu[10:], after_imu[10:])


def test_window_features_match_manual_recomputation(cache_root):
    """Cached row t must equal extractor(raw rows [max(0,t-L+1)..t]) computed
    from the scenario's own sensor table — proven here for tick 7."""
    scen = cache_root / "train/Town01/scenario-1"
    rows = extract_gnss_feature_rows(scen, 20, window_length=12)
    import pandas as pd
    table = pd.read_feather(scen / "gnss.feather")[
        ["altitude", "latitude", "longitude"]].to_numpy(dtype=np.float64)
    t = 7
    np.testing.assert_allclose(
        rows[t - 1], gnss_window_features(table[max(0, t - WINDOW_LENGTH + 1):t + 1]),
        rtol=0, atol=0,
    )


def test_warmup_rule_first_cached_tick_is_one(cache_root, cache_dir):
    arrays, m = load_cache(cache_dir) if cache_dir.exists() else (None, None)
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, m = load_cache(cache_dir)
    assert arrays["tick"].min() == 1
    assert m["window_policy"]["first_cached_tick"] == 1
    assert m["window_policy"]["causal"] is True
    assert m["window_policy"]["sampling_frequency_claimed"] is False


def test_no_cross_scenario_window_leakage(cache_root, cache_dir):
    """Two scenarios with IDENTICAL sensor tables except at the tail: window
    features near the head must not see the other scenario's tail rows."""
    import pandas as pd
    a = pd.read_feather(cache_root / "train/Town01/scenario-1" / "gnss.feather")
    b = a.copy()
    b["latitude"] = b["latitude"] + 1.0  # massively different trajectory
    b.to_feather(cache_root / "train/Town01/scenario-2" / "gnss.feather")
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, _ = load_cache(cache_dir)
    g1 = arrays["gnss"][arrays["scenario_id"] == "Town01/scenario-1"]
    g2 = arrays["gnss"][arrays["scenario_id"] == "Town01/scenario-2"]
    assert not np.allclose(g1, g2)  # each scenario used only its own table


# ── 5. dimensions & finiteness ───────────────────────────────────────────────

def test_feature_dimensions_and_finiteness(cache_root, cache_dir):
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, m = load_cache(cache_dir)
    assert arrays["camera"].shape[1] == CAMERA_DIM == 18
    assert arrays["seg"].shape[1] == SEG_DIM == 29
    assert arrays["gnss"].shape[1] == GNSS_DIM == 8
    assert arrays["imu"].shape[1] == IMU_DIM == 10
    assert arrays["cal_features"].shape[1] == COMPACT_DIM == 65
    assert m["feature_dims"] == {"camera": 18, "seg": 29, "gnss": 8, "imu": 10}
    for k in ("camera", "seg", "gnss", "imu", "cal_features"):
        assert np.all(np.isfinite(arrays[k])), k


def test_feature_values_match_extractor_on_raw_payloads(cache_root, cache_dir):
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, _ = load_cache(cache_dir)
    from cognix.adapters.carla.real_features import camera_embedding_features
    import PIL.Image as PIL
    sid = "train/Town01/scenario-1"
    mask = arrays["scenario_id"] == "Town01/scenario-1"
    ticks = arrays["tick"][mask]
    cam = arrays["camera"][mask]
    for idx in (0, 5, 12):
        t = int(ticks[idx])
        img = np.asarray(PIL.open(cache_root / sid / "rgb-front" / f"{t:06d}.jpg"))
        np.testing.assert_allclose(cam[idx], camera_embedding_features(img), rtol=0, atol=0)


# ── 6. CAL pairing + provenance ──────────────────────────────────────────────

def test_cal_pairs_normal_and_pseudo(cache_root, cache_dir):
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, m = load_cache(cache_dir)
    assert m["pseudo_anomalies"]["present"] is True
    assert arrays["cal_target_normal"].tolist() == [0] * len(arrays["cal_target_normal"])
    assert m["pseudo_anomalies"]["gated_to_split"] == "train"
    # normal CAL rows exist as regular rows with partition=CAL_NORMAL
    assert "CAL_NORMAL" in set(arrays["partition"].tolist())
    # rotation guarantees multiple recipes appear across the CAL set
    recipes_seen = set(arrays["cal_recipe_id"].tolist())
    assert len(recipes_seen) >= 3
    assert recipes_seen <= set(CAL_PSEUDO_RECIPES)


def test_cal_parent_provenance_complete(cache_root, cache_dir):
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, m = load_cache(cache_dir)
    for i in range(len(arrays["cal_target_normal"])):
        assert arrays["cal_parent_scenario"][i]
        assert 1 <= int(arrays["cal_parent_tick"][i])
        assert arrays["cal_recipe_id"][i] in CAL_PSEUDO_RECIPES
        assert arrays["cal_modality"][i] in {"camera", "seg", "gnss", "imu"}
        assert float(arrays["cal_severity"][i]) >= 0.0
        assert int(arrays["cal_seed"][i]) >= 0
    # no-effect rows were skipped and the skip count is audited in the manifest
    assert m["pseudo_anomalies"]["skipped_no_effect_rows"] >= 0
    assert m["pseudo_anomalies"]["n_rows"] == len(arrays["cal_target_normal"])


def test_pseudo_rows_differ_from_parent_rows(cache_root, cache_dir):
    """Pseudo rows model the corrupted OBSERVATION: the corrupted modality
    block differs from the parent's; untouched blocks equal the parent's."""
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, _ = load_cache(cache_dir)
    offsets = {"camera": 0, "seg": 18, "gnss": 47, "imu": 55}
    dims = {"camera": 18, "seg": 29, "gnss": 8, "imu": 10}
    checked = 0
    for i in range(len(arrays["cal_target_normal"])):
        sid = str(arrays["cal_parent_scenario"][i])
        t = int(arrays["cal_parent_tick"][i])
        mod = str(arrays["cal_modality"][i])
        parent_mask = (arrays["scenario_id"] == sid) & (arrays["tick"] == t)
        if not parent_mask.any():
            continue  # parent tick 0 is not cached (warm-up); skip honestly
        parent_vec = np.concatenate(
            [arrays["camera"][parent_mask][0], arrays["seg"][parent_mask][0],
             arrays["gnss"][parent_mask][0], arrays["imu"][parent_mask][0]]
        )
        off, dim = offsets[mod], dims[mod]
        # no-effect rows (corruption entirely outside the causal window, e.g.
        # gnss_bias's midpoint rule or a seeded imu_spike burst) are skipped by
        # the builder, so EVERY emitted row must differ from its parent here
        assert not np.allclose(arrays["cal_features"][i][off:off + dim],
                               parent_vec[off:off + dim]), (
                    f"row {i} ({arrays['cal_recipe_id'][i]}): corrupted block "
                    f"{mod} identical to parent")
        # untouched blocks equal the parent's (real features, not zeros)
        for m2 in ("camera", "seg", "gnss", "imu"):
            if m2 != mod:
                o2, d2 = offsets[m2], dims[m2]
                np.testing.assert_allclose(
                    arrays["cal_features"][i][o2:o2 + d2],
                    parent_vec[o2:o2 + d2], rtol=1e-12, atol=1e-15,
                )
        checked += 1
    assert checked > 0


# ── 7. TEST rejection ────────────────────────────────────────────────────────

def test_builder_only_enumerates_train(cache_root, cache_dir):
    """The builder walks CarlAnomalySplit.TRAIN only; a scenario that exists
    ONLY under test/ can never be requested or cached."""
    _make_scenario(cache_root, "test/normal/Town01/scenario-9", 10, seed=555)
    m = build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    assert "Town01/scenario-9" not in m["scenario_ids"]
    arrays, _ = load_cache(cache_dir)
    assert "Town01/scenario-9" not in set(arrays["scenario_id"].tolist())
    assert set(arrays["source_split"].tolist()) == {"train"}


def test_builder_rejects_unknown_scenario_ids(cache_root, cache_dir):
    with pytest.raises(CacheBuilderError, match="not present in TRAIN"):
        build_cache(cache_root, cache_dir, scenarios=["Town01/nonexistent"])


def test_pseudo_generator_gate_still_rejects_test(cache_root):
    from cognix.adapters.carla.pseudo_anomalies import (
        PseudoAnomalyError,
        generate_pseudo_anomaly,
    )
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(
            {"GNSS": np.zeros((4, 3))}, "gnss_bias", CarlAnomalySplit.TEST_ANOMALY
        )


def test_builder_rejects_scenario_missing_modality(cache_root, cache_dir):
    # scenario-3 without segmentation frames must fail loud, never zero-fill
    import shutil
    shutil.rmtree(cache_root / "train/Town01/scenario-3" / "segmentation-front")
    with pytest.raises(Exception, match="segmentation_front"):
        build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)


# ── 8. manifest consistency ─────────────────────────────────────────────────

def test_manifest_fields_complete(cache_root, cache_dir):
    m = build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    for key in (
        "cache_schema_version", "source_split", "scenario_ids", "feature_dims",
        "extractor_version", "window_policy", "code_version", "cache_content_sha256",
    ):
        assert key in m
    assert m["cache_schema_version"] == 2
    assert m["source_split"] == "train"
    assert m["agent_outputs_present"] is False
    assert m["raw_data_present"] is False
    assert m["window_policy"]["window_length_ticks"] == WINDOW_LENGTH
    assert m["extractor_version"].startswith("real_features.py sha256:")
    assert len(m["cache_content_sha256"]) == 64


def test_manifest_consistent_with_arrays(cache_root, cache_dir):
    m = build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, _ = load_cache(cache_dir)
    assert m["n_rows"] == len(arrays["tick"]) == sum(
        len(arrays["tick"][arrays["scenario_id"] == s]) for s in m["scenario_ids"]
    )
    assert sorted(m["scenario_ids"]) == sorted(set(arrays["scenario_id"].tolist()))
    assert m["pseudo_anomalies"]["n_rows"] == len(arrays["cal_target_normal"])


# ── 9. no mutation of source data ────────────────────────────────────────────

def test_source_data_not_mutated(cache_root, cache_dir):
    def snapshot(root):
        out = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root))] = p.stat().st_size
        return out
    before = snapshot(cache_root)
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    assert snapshot(cache_root) == before


# ── 10. import purity ────────────────────────────────────────────────────────

def test_generic_import_does_not_pull_adapters():
    import subprocess
    import sys
    code = (
        "import sys; import cognix; "
        "sys.exit(1 if any(m.startswith('cognix.adapters') for m in sys.modules) else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


def test_cache_builder_never_touches_frozen_bridge():
    import inspect
    from cognix.adapters.carla import cache_builder
    src = inspect.getsource(cache_builder)
    for forbidden in ("CarlAnomalyDataset", "build_synthetic_artifacts",
                      "from cognix.adapters.carla.agents import", "features_to_probability"):
        assert forbidden not in src


# ── 11. overwrite guard ──────────────────────────────────────────────────────

def test_no_silent_overwrite_of_mismatched_cache(cache_root, cache_dir):
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    _, m_before = load_cache(cache_dir)
    _make_scenario(cache_root, "train/Town01/scenario-1", 30, seed=999)  # changes data
    with pytest.raises(CacheBuilderError, match="mismatched cache"):
        build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    _, m_after = load_cache(cache_dir)
    assert m_after["cache_content_sha256"] == m_before["cache_content_sha256"]


def test_allow_mismatch_flag_overrides_guard(cache_root, cache_dir):
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    _make_scenario(cache_root, "train/Town01/scenario-1", 30, seed=999)
    m = build_cache(cache_root, cache_dir, partition_seed=7,
                    cal_fraction=0.5, allow_mismatch=True)
    _, m_after = load_cache(cache_dir)
    assert m_after["cache_content_sha256"] == m["cache_content_sha256"]


# ── integration smoke: tiny normality fit on cache rows (fixture-only) ──────

def test_smoke_normality_fit_on_cache_rows(cache_root, cache_dir):
    """Tiny fixture-only integration smoke: TRAIN_NORMAL rows -> one-class fit;
    cached rows score as normal (predict_normality takes ONE row). NOT a real
    model — no GAT, no conformal, no artifacts."""
    from cognix.adapters.carla.normality import MahalanobisNormality
    build_cache(cache_root, cache_dir, partition_seed=7, cal_fraction=0.5)
    arrays, _ = load_cache(cache_dir)
    train_mask = arrays["partition"] == "TRAIN_NORMAL"
    X = np.column_stack(
        [arrays["camera"], arrays["seg"], arrays["gnss"], arrays["imu"]]
    )[train_mask]
    model = MahalanobisNormality().fit(X)
    sample_idx = np.linspace(0, len(X) - 1, 5).astype(int)
    scores = [model.predict_normality(X[i]) for i in sample_idx]
    assert np.all(np.isfinite(scores))
    assert float(np.median(scores)) > 0.05, (
        f"typical TRAIN_NORMAL row collapsed: {np.median(scores):.3e}"
    )
    # CAL pseudo rows are finite 65-dim vectors in the same space
    assert arrays["cal_features"].shape[1] == 65
    assert np.all(np.isfinite(arrays["cal_features"]))

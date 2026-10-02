"""Tests for TRAIN-only pseudo-anomaly generation (Part B).

Covers:
1.  Hard split gate: TEST_NORMAL / TEST_ANOMALY / unknown / missing / wrong-type
    provenance are rejected at every entry point; there is NO default split, so
    "forgetting" to specify a split cannot reach the generator with test data.
2.  Determinism, purity (no source mutation), finiteness, schema preservation,
    explicit PCG64 seeding (no salted hash()).
3.  Per-recipe behavior: camera brightness/occlusion/blackout, seg region
    corruption restricted to SOURCE-OBSERVED class ids, GNSS bias/drift in
    unit-correct meters, IMU spike/bias-scale preserving orientation columns.
4.  Provenance records (recipe_id, modality, seed, severity, source_split,
    source_scenario, source_tick, synthetic_corruption).
5.  Path through real_features.py extractors and the Stage-1 agent calibration
    lifecycle (normal + pseudo-anomaly CAL pairs -> fit_calibrator -> predict).
6.  Scientific wording: corruptions are NOT claimed to replicate the real
    CarlAnomaly anomaly distribution; severities are fixed a priori.

No downloads, no TEST data, no artifact fitting, no GAT/conformal changes.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from cognix.adapters.carla.carlanomaly_loader import CarlAnomalySplit
from cognix.adapters.carla.pseudo_anomalies import (
    PseudoAnomalyError,
    PseudoAnomalySample,
    build_calibration_pairs,
    generate_pseudo_anomaly,
    recipe_registry,
    require_train_split,
)
from cognix.adapters.carla.real_features import (
    camera_embedding_features,
    gnss_window_features,
    imu_window_features,
    local_meter_offsets_to_degrees,
    segmentation_histogram_features,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def camera_frame():
    rng = np.random.default_rng(20)
    return rng.integers(40, 220, size=(24, 32, 3), dtype=np.uint8)


@pytest.fixture
def seg_map():
    """Two-class map (both ids 'already present'; no invented semantics)."""
    seg = np.zeros((16, 16), dtype=np.uint8)
    seg[8:, :] = 3
    return seg


@pytest.fixture
def gnss_rows():
    """Gently moving trajectory: 0.5 m east steps at lat 30, alt 100 m."""
    n = 12
    return np.stack(
        [np.full(n, 100.0), np.full(n, 30.0), np.linspace(0.0, 5.5e-6, n)],
        axis=1,
    )


@pytest.fixture
def imu_accel():
    rng = np.random.default_rng(21)
    return rng.normal(0.0, 0.3, size=(12, 3))


@pytest.fixture
def observation(camera_frame, seg_map, gnss_rows, imu_accel):
    return {
        "Camera": camera_frame,
        "Seg": seg_map,
        "GNSS": gnss_rows,
        "IMU": imu_accel,
        "scenario_id": "train-scenario-7",
        "tick": 42,
    }


ALL_RECIPE_IDS = [rid for rid, _mod in recipe_registry()]


# ── 1. hard split gate ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "bad_split",
    [
        CarlAnomalySplit.TEST_NORMAL,
        CarlAnomalySplit.TEST_ANOMALY,
        "test_normal",
        "test_anomaly",
        "train_but_wrong",   # unknown string
        "test",              # partial match must not sneak through
        42,                  # wrong type
        object(),            # wrong type
        None,                # missing provenance
    ],
)
def test_gate_rejects_non_train_provenance(bad_split, observation):
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(observation, "gnss_bias", bad_split)


def test_gate_accepts_train_enum_and_train_string(observation):
    s1 = generate_pseudo_anomaly(observation, "gnss_bias", CarlAnomalySplit.TRAIN)
    s2 = generate_pseudo_anomaly(observation, "gnss_bias", "train")
    assert s1.source_split == s2.source_split == "train"


def test_no_default_split_exists():
    """Omitting the split is ITSELF an error: the parameter has no default, so
    a caller cannot reach the generator with TEST data by forgetting it."""
    sig = inspect.signature(generate_pseudo_anomaly)
    assert sig.parameters["source_split"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        generate_pseudo_anomaly(observation=None, recipe_id="gnss_bias")  # type: ignore[call-arg]


def test_gate_function_direct():
    """Direct unit coverage of require_train_split."""
    assert require_train_split(CarlAnomalySplit.TRAIN) is CarlAnomalySplit.TRAIN
    with pytest.raises(PseudoAnomalyError):
        require_train_split(None)
    with pytest.raises(PseudoAnomalyError):
        require_train_split(CarlAnomalySplit.TEST_ANOMALY)


def test_observation_split_mismatch_rejected(observation):
    """An observation carrying test provenance must be refused even if the
    caller passes TRAIN as source_split."""
    tainted = dict(observation)
    tainted["__split__"] = "test_anomaly"
    with pytest.raises(PseudoAnomalyError, match="mismatch"):
        generate_pseudo_anomaly(tainted, "gnss_bias", CarlAnomalySplit.TRAIN)


def test_observation_unknown_split_string_rejected(observation):
    tainted = dict(observation)
    tainted["__split__"] = "not-a-split"
    with pytest.raises(PseudoAnomalyError, match="unknown __split__"):
        generate_pseudo_anomaly(tainted, "gnss_bias", CarlAnomalySplit.TRAIN)


def test_observation_explicit_none_split_rejected(observation):
    tainted = dict(observation)
    tainted["__split__"] = None
    with pytest.raises(PseudoAnomalyError, match="missing __split__"):
        generate_pseudo_anomaly(tainted, "gnss_bias", CarlAnomalySplit.TRAIN)


def test_calibration_builder_gated_too(observation):
    with pytest.raises(PseudoAnomalyError):
        build_calibration_pairs(
            [observation], CarlAnomalySplit.TEST_NORMAL, ["gnss_bias"]
        )
    with pytest.raises(PseudoAnomalyError):
        build_calibration_pairs([observation], None, ["gnss_bias"])


def test_unknown_recipe_id_rejected(observation):
    with pytest.raises(PseudoAnomalyError, match="unknown recipe_id"):
        generate_pseudo_anomaly(
            observation, "does_not_exist", CarlAnomalySplit.TRAIN
        )


# ── 2. determinism, purity, finiteness, schema ──────────────────────────────

@pytest.mark.parametrize("recipe_id", ALL_RECIPE_IDS)
def test_recipe_deterministic_for_fixed_seed(observation, recipe_id):
    a = generate_pseudo_anomaly(
        observation, recipe_id, CarlAnomalySplit.TRAIN, seed=7
    )
    b = generate_pseudo_anomaly(
        observation, recipe_id, CarlAnomalySplit.TRAIN, seed=7
    )
    assert isinstance(a, PseudoAnomalySample)
    if isinstance(a.data, dict):
        for col in a.data:
            np.testing.assert_array_equal(np.asarray(a.data[col]), np.asarray(b.data[col]))
    else:
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))


@pytest.mark.parametrize("recipe_id", ALL_RECIPE_IDS)
def test_recipe_never_mutates_source(observation, recipe_id):
    snapshot = {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in observation.items()
    }
    generate_pseudo_anomaly(observation, recipe_id, CarlAnomalySplit.TRAIN, seed=3)
    for k, v in observation.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(v, snapshot[k], err_msg=f"{recipe_id} mutated {k}")


@pytest.mark.parametrize("recipe_id", ALL_RECIPE_IDS)
def test_recipe_output_finite_and_schema_preserved(observation, recipe_id):
    s = generate_pseudo_anomaly(observation, recipe_id, CarlAnomalySplit.TRAIN, seed=1)
    if isinstance(s.data, dict):
        for col, arr in s.data.items():
            assert np.all(np.isfinite(np.asarray(arr, dtype=np.float64))), col
            assert np.asarray(s.data[col]).shape == np.asarray(observation["IMU"][col]).shape or col not in ("acceleration_x",)
    else:
        arr = np.asarray(s.data)
        assert np.all(np.isfinite(arr.astype(np.float64)))
        assert arr.shape == np.asarray(observation[_payload_key(recipe_id)]).shape
        assert arr.dtype == np.asarray(observation[_payload_key(recipe_id)]).dtype


def _payload_key(recipe_id: str) -> str:
    return {
        "camera": "Camera", "seg": "Seg", "gnss": "GNSS", "imu": "IMU",
    }[recipe_id.split("_")[0]]


def test_no_salted_hash_anywhere_in_module():
    from cognix.adapters.carla import pseudo_anomalies
    src = inspect.getsource(pseudo_anomalies)
    # the dangerous pattern is seeding from Python's salted hash(); the docstring
    # mentioning "hash()" prose is not a violation
    assert "abs(hash(" not in src
    assert "default_rng" in src  # explicit PCG64 seeding is the only randomness

def test_stochastic_recipes_vary_with_seed(seg_map):
    """Recipes with positional randomness must actually use the seed."""
    base = np.zeros((16, 16, 3), dtype=np.uint8)
    base[..., 0] = seg_map
    outputs = {
        np.asarray(
            generate_pseudo_anomaly(
                {"Camera": base}, "camera_occlusion", CarlAnomalySplit.TRAIN, seed=s
            ).data
        ).tobytes()
        for s in range(8)
    }
    assert len(outputs) >= 2, "different seeds produced identical occlusions"


# ── 3a. camera recipes ───────────────────────────────────────────────────────

def test_camera_brightness_down_reduces_intensity(camera_frame):
    s = generate_pseudo_anomaly(
        {"Camera": camera_frame}, "camera_brightness_shift",
        CarlAnomalySplit.TRAIN, severity=0.5,
    )
    out = np.asarray(s.data)
    assert out.shape == camera_frame.shape and out.dtype == camera_frame.dtype
    assert np.all(out.astype(np.int64) <= camera_frame.astype(np.int64))
    assert out.mean() < camera_frame.mean()


def test_camera_brightness_severity_one_blacks_out(camera_frame):
    s = generate_pseudo_anomaly(
        {"Camera": camera_frame}, "camera_brightness_shift",
        CarlAnomalySplit.TRAIN, severity=1.0,
    )
    assert np.asarray(s.data).max() == 0


def test_camera_brightness_up_increases_and_validates(camera_frame):
    s = generate_pseudo_anomaly(
        {"Camera": camera_frame}, "camera_brightness_shift",
        CarlAnomalySplit.TRAIN, severity=0.5,
    )
    # default direction is 'down'; call the recipe-level 'up' through the
    # registry function to keep the entry-point surface minimal.
    from cognix.adapters.carla.pseudo_anomalies import camera_brightness_shift
    up = camera_brightness_shift(
        {"Camera": camera_frame}, severity=0.5,
        rng=np.random.default_rng(0), direction="up",
    )
    assert np.asarray(up.data).mean() > camera_frame.mean()
    with pytest.raises(PseudoAnomalyError):
        camera_brightness_shift(
            {"Camera": camera_frame}, severity=2.0,
            rng=np.random.default_rng(0), direction="down",
        )


def test_camera_occlusion_area_and_validity(camera_frame):
    sev = 0.25
    s = generate_pseudo_anomaly(
        {"Camera": camera_frame}, "camera_occlusion",
        CarlAnomalySplit.TRAIN, severity=sev,
    )
    out = np.asarray(s.data)
    zeroed = int((out == 0).all(axis=2).sum())
    assert zeroed > 0
    assert zeroed / out.shape[0] / out.shape[1] == pytest.approx(sev, rel=0.35)
    assert abs(s.severity - sev) < 1e-12


def test_camera_blackout_full_frame(camera_frame):
    s = generate_pseudo_anomaly(
        {"Camera": camera_frame}, "camera_blackout",
        CarlAnomalySplit.TRAIN, severity=1.0,
    )
    assert np.asarray(s.data).max() == 0
    # blackout at 0.5 must coincide with occlusion at 0.5 (same worker, same seed)
    occ = generate_pseudo_anomaly(
        {"Camera": camera_frame}, "camera_occlusion",
        CarlAnomalySplit.TRAIN, severity=0.5,
    )
    blk = generate_pseudo_anomaly(
        {"Camera": camera_frame}, "camera_blackout",
        CarlAnomalySplit.TRAIN, severity=0.5,
    )
    np.testing.assert_array_equal(np.asarray(occ.data), np.asarray(blk.data))


@pytest.mark.parametrize("recipe_id", ["camera_occlusion", "camera_blackout"])
def test_camera_rect_recipes_reject_bad_severity(camera_frame, recipe_id):
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(
            {"Camera": camera_frame}, recipe_id, CarlAnomalySplit.TRAIN, severity=0.0
        )
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(
            {"Camera": camera_frame}, recipe_id, CarlAnomalySplit.TRAIN, severity=1.5
        )


def test_camera_recipes_reject_bad_shape():
    with pytest.raises(PseudoAnomalyError, match="H, W, 3"):
        generate_pseudo_anomaly(
            {"Camera": np.zeros((5, 5), dtype=np.uint8)},
            "camera_occlusion", CarlAnomalySplit.TRAIN,
        )


# ── 3b. segmentation recipe ─────────────────────────────────────────────────

def test_seg_corruption_uses_only_source_class_ids(seg_map):
    s = generate_pseudo_anomaly(
        {"Seg": seg_map}, "seg_region_corruption", CarlAnomalySplit.TRAIN, seed=2
    )
    out_classes = np.asarray(s.data)
    assert set(np.unique(out_classes).tolist()) <= set(np.unique(seg_map).tolist())


def test_seg_corruption_changes_histogram(seg_map):
    before = segmentation_histogram_features(seg_map)
    s = generate_pseudo_anomaly(
        {"Seg": seg_map}, "seg_region_corruption", CarlAnomalySplit.TRAIN,
        seed=2, severity=0.3,
    )
    after = segmentation_histogram_features(np.asarray(s.data))
    assert np.all(np.isfinite(after))
    assert after.sum() == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(before, after)


def test_seg_single_class_map_is_noop_but_valid():
    seg = np.full((8, 8), 6, dtype=np.uint8)
    s = generate_pseudo_anomaly(
        {"Seg": seg}, "seg_region_corruption", CarlAnomalySplit.TRAIN, seed=0
    )
    # Only class 6 exists in the source, so the replacement can only be 6:
    # labels unchanged, output valid, no invented ids ever appear.
    np.testing.assert_array_equal(np.asarray(s.data), seg)


def test_seg_corruption_handles_3d_r_channel_maps(seg_map):
    seg3 = np.zeros((16, 16, 3), dtype=np.uint8)
    seg3[..., 0] = seg_map
    seg3[..., 1] = 200  # non-class channels must be ignored AND preserved
    s = generate_pseudo_anomaly(
        {"Seg": seg3}, "seg_region_corruption", CarlAnomalySplit.TRAIN, seed=2
    )
    out = np.asarray(s.data)
    assert out.shape == seg3.shape
    assert set(np.unique(out[..., 0]).tolist()) <= {0, 3}
    # non-class channels preserved byte-identical
    np.testing.assert_array_equal(out[..., 1], seg3[..., 1])
    np.testing.assert_array_equal(out[..., 2], seg3[..., 2])


def test_seg_empty_map_rejected():
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(
            {"Seg": np.empty((0,), dtype=np.uint8)},
            "seg_region_corruption", CarlAnomalySplit.TRAIN,
        )


# ── 3c. GNSS recipes (unit-correct meters) ──────────────────────────────────

def test_gnss_bias_displacement_in_meters(gnss_rows):
    """The step-bias recipe must produce a GPS-jump of EXACTLY the documented
    magnitude, expressed in true meters via the small-area approximation."""
    sev = 40.0
    s = generate_pseudo_anomaly(
        {"GNSS": gnss_rows}, "gnss_bias", CarlAnomalySplit.TRAIN, severity=sev
    )
    corrupted = np.asarray(s.data)
    assert corrupted.shape == gnss_rows.shape
    # first half untouched, second half offset by the (north, east, alt) vector
    mid = len(gnss_rows) // 2
    np.testing.assert_array_equal(corrupted[:mid], gnss_rows[:mid])
    dlat, dlon = local_meter_offsets_to_degrees(sev, 0.6 * sev, lat_deg=30.0)
    np.testing.assert_allclose(corrupted[mid:, 1], gnss_rows[mid:, 1] + dlat, atol=1e-18)
    np.testing.assert_allclose(corrupted[mid:, 2], gnss_rows[mid:, 2] + dlon, atol=1e-18)
    np.testing.assert_allclose(corrupted[mid:, 0], gnss_rows[mid:, 0] + sev / 5.0, atol=1e-15)
    # the window feature that sees the jump: mean/max step magnitude explode;
    # max step ~= jump magnitude (tiny rel-slack: the fixture's own sub-meter
    # motion adds vectorially to the jump step)
    biased = gnss_window_features(corrupted)
    clean = gnss_window_features(gnss_rows)
    assert biased[5] == pytest.approx(np.hypot(sev, 0.6 * sev), rel=1e-3)
    assert biased[4] > clean[4]


def test_gnss_drift_ramps_to_total_severity(gnss_rows):
    sev = 80.0
    s = generate_pseudo_anomaly(
        {"GNSS": gnss_rows}, "gnss_drift", CarlAnomalySplit.TRAIN, severity=sev
    )
    corrupted = np.asarray(s.data)
    f = gnss_window_features(corrupted)
    # uniform ramp: total displacement hypot(sev, 0.4*sev) spread over n-1 equal
    # steps; altitude ramps 0 -> sev/5 (fixture range is 0, so range == ramp)
    added = np.hypot(sev, 0.4 * sev)
    n = len(gnss_rows)
    assert f[4] == pytest.approx(added / (n - 1), rel=1e-2)  # mean step (base motion ~sub-meter)
    assert f[6] == pytest.approx(sev / 5.0, rel=1e-9)        # altitude range == ramp
    assert f[7] == pytest.approx(1.0, abs=1e-9)              # zero relative motion -> straight
    # gradual ramp: per-step magnitude far below the step-bias window's jump
    bias = generate_pseudo_anomaly(
        {"GNSS": gnss_rows}, "gnss_bias", CarlAnomalySplit.TRAIN, severity=sev
    )
    assert f[4] < gnss_window_features(np.asarray(bias.data))[4]


def test_gnss_recipes_reject_bad_schema():
    with pytest.raises(PseudoAnomalyError, match="n>=2, 3"):
        generate_pseudo_anomaly(
            {"GNSS": np.zeros((4, 2))}, "gnss_bias", CarlAnomalySplit.TRAIN
        )
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(
            {"GNSS": np.zeros((1, 3))}, "gnss_drift", CarlAnomalySplit.TRAIN
        )
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(
            {"GNSS": np.zeros((4, 3))}, "gnss_bias", CarlAnomalySplit.TRAIN,
            severity=-1.0,
        )


# ── 3d. IMU recipes (orientation preserved) ─────────────────────────────────

IMU_DICT = {
    "acceleration_x": np.linspace(-1, 1, 12),
    "acceleration_y": np.zeros(12),
    "acceleration_z": np.full(12, 0.5),
    "compass": np.linspace(0, 360, 12),
    "longitude_x": np.full(12, 0.01),
}

# Module-level dicts are shared across the whole pytest session; recipes must
# never mutate them, but a test must not RELY on later tests seeing clean data.
def _fresh_imu_dict():
    return {k: np.array(v, copy=True) for k, v in IMU_DICT.items()}


def test_imu_spike_changes_change_statistic():
    accel = np.zeros((12, 3))
    s = generate_pseudo_anomaly(
        {"IMU": accel}, "imu_spike", CarlAnomalySplit.TRAIN, seed=4, severity=9.0
    )
    f_clean = imu_window_features(accel)
    f_corr = imu_window_features(np.asarray(s.data))
    # spike axis/position are deterministic given the seed; assert against the
    # recipe's actual choice instead of hardcoding it.
    spike_axis = int(np.argmax(np.ptp(np.asarray(s.data), axis=0)))
    assert np.ptp(np.asarray(s.data)[:, spike_axis]) == pytest.approx(9.0)
    assert f_corr[9] == pytest.approx(9.0)   # max |acceleration change| = spike
    assert f_clean[9] == pytest.approx(0.0)
    # orientation untouched, single-modality array payload preserved as array
    assert np.asarray(s.data).shape == accel.shape


def test_imu_dict_payload_preserves_orientation():
    d = _fresh_imu_dict()
    s = generate_pseudo_anomaly(
        {"IMU": d}, "imu_spike", CarlAnomalySplit.TRAIN, seed=4, severity=5.0
    )
    out = s.data
    assert isinstance(out, dict)
    # orientation channels pass through byte-identical
    np.testing.assert_array_equal(np.asarray(out["compass"]), IMU_DICT["compass"])
    np.testing.assert_array_equal(np.asarray(out["longitude_x"]), IMU_DICT["longitude_x"])
    # exactly ONE acceleration column carries the seeded spike; the others are
    # byte-identical; schema preserved (spike axis is deterministic given seed 4)
    from cognix.adapters.carla.real_features import IMU_ACCEL_COLUMNS
    changed = [c for c in IMU_ACCEL_COLUMNS if not np.array_equal(np.asarray(out[c]), IMU_DICT[c])]
    assert len(changed) == 1
    for c in IMU_ACCEL_COLUMNS:
        if c != changed[0]:
            np.testing.assert_array_equal(np.asarray(out[c]), IMU_DICT[c])
    assert set(out) == set(IMU_DICT)


def test_imu_bias_scale_exact_affine_and_orientation_kept():
    d = _fresh_imu_dict()
    s = generate_pseudo_anomaly(
        {"IMU": d}, "imu_bias_scale", CarlAnomalySplit.TRAIN, severity=0.5
    )
    out = s.data
    expected_x = 1.5 * IMU_DICT["acceleration_x"] + 0.5
    np.testing.assert_allclose(np.asarray(out["acceleration_x"]), expected_x, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(out["longitude_x"]), IMU_DICT["longitude_x"])
    np.testing.assert_array_equal(np.asarray(out["compass"]), IMU_DICT["compass"])
    # input dict arrays were not mutated
    np.testing.assert_array_equal(d["acceleration_x"], np.linspace(-1, 1, 12))


def test_imu_dict_missing_accel_column_rejected():
    broken = {k: v for k, v in IMU_DICT.items() if k != "acceleration_y"}
    with pytest.raises(PseudoAnomalyError, match="acceleration_y"):
        generate_pseudo_anomaly(
            {"IMU": broken}, "imu_spike", CarlAnomalySplit.TRAIN
        )


# ── 4. provenance ────────────────────────────────────────────────────────────

def test_provenance_record_complete(observation):
    s = generate_pseudo_anomaly(
        observation, "gnss_bias", CarlAnomalySplit.TRAIN,
        source_scenario="train-scenario-7", source_tick=42, seed=11, severity=30.0,
    )
    prov = s.provenance()
    assert prov == {
        "recipe_id": "gnss_bias",
        "modality": "gnss",
        "seed": 11,
        "severity": 30.0,
        "source_split": "train",
        "source_scenario": "train-scenario-7",
        "source_tick": 42,
        "synthetic_corruption": True,
    }


def test_frozen_dataclass_provenance_immutable():
    s = PseudoAnomalySample(
        data=np.zeros(3), recipe_id="r", modality="imu", seed=0, severity=1.0,
        source_split="train", source_scenario="sc", source_tick=0,
        synthetic_corruption=True,
    )
    with pytest.raises(Exception):
        s.synthetic_corruption = False  # type: ignore[misc]


# ── 5. path through real_features.py and the calibration lifecycle ──────────

def test_corrupted_camera_through_extractor_changes_features(camera_frame):
    clean = camera_embedding_features(camera_frame)
    for rid, sev in (("camera_brightness_shift", 0.6), ("camera_blackout", 1.0)):
        s = generate_pseudo_anomaly(
            {"Camera": camera_frame}, rid, CarlAnomalySplit.TRAIN, severity=sev
        )
        v = camera_embedding_features(np.asarray(s.data))
        assert v.shape == (18,) and np.all(np.isfinite(v))
        assert not np.allclose(clean, v), rid


def test_corrupted_gnss_through_extractor_changes_features(gnss_rows):
    clean = gnss_window_features(gnss_rows)
    s = generate_pseudo_anomaly(
        {"GNSS": gnss_rows}, "gnss_drift", CarlAnomalySplit.TRAIN, severity=60.0
    )
    v = gnss_window_features(np.asarray(s.data))
    assert v.shape == (8,) and np.all(np.isfinite(v))
    assert not np.allclose(clean, v)


def test_unrelated_modalities_unchanged_in_cal_pairs(observation):
    obs, labels, rows = build_calibration_pairs(
        [observation], CarlAnomalySplit.TRAIN, ["gnss_bias"], seed=0
    )
    corrupted = obs[1]
    assert labels[1] == 0.0
    np.testing.assert_array_equal(corrupted["Camera"], observation["Camera"])
    np.testing.assert_array_equal(corrupted["Seg"], observation["Seg"])
    np.testing.assert_array_equal(np.asarray(corrupted["IMU"]), observation["IMU"])
    assert not np.array_equal(corrupted["GNSS"], observation["GNSS"])


def test_calibration_pair_lifecycle_through_real_agent(observation, gnss_rows):
    """Normal + pseudo-corrupted CAL pairs drive the Stage-1 lifecycle end to
    end: both classes present, calibrator fits, prediction stays canonical."""
    from cognix.adapters.carla.real_agents import RealGNSSAgent

    # normality model over CLEAN window features only (TRAIN-derived):
    # 30 trajectories with healthy variance in ALL 8 window-feature dims —
    # mean east step brackets the CAL motion (~0.045 m/step), per-step jitter
    # varies (step-std dims), altitude ramps vary (range dim) — so the Mahalanobis
    # metric is non-degenerate and the NORMAL CAL window sits inside the cloud.
    from math import cos, degrees as _deg, radians as _rad

    from cognix.adapters.carla.real_features import EARTH_RADIUS_M

    k_lat = _deg(1.0 / EARTH_RADIUS_M)
    k_lon = k_lat / cos(_rad(30.0))
    rng = np.random.default_rng(5)
    train_feats = []
    for i in range(30):
        east_m = 0.01 + 0.08 * i / 29                  # 0.01..0.09 m per step
        north_m = float(rng.normal(0.0, 0.005))        # brackets 0
        jitter = 0.001 + 0.003 * (i % 5) / 4.0         # per-step noise scale
        alt_ramp = 0.05 * (i % 4)                      # total altitude drift
        steps_n = north_m + rng.normal(0.0, jitter, 12)
        steps_e = east_m + rng.normal(0.0, jitter, 12)
        rows = np.stack(
            [
                100.0 + alt_ramp * np.arange(12) / 11.0,
                30.0 + np.cumsum(steps_n) * k_lat,
                np.cumsum(steps_e) * k_lon,
            ],
            axis=1,
        )
        train_feats.append(gnss_window_features(rows))
    clean_feats = np.array(train_feats)
    agent = RealGNSSAgent(seed=42, n_members=3)
    agent.fit(clean_feats)

    cal_obs, labels, rows = build_calibration_pairs(
        [observation],
        CarlAnomalySplit.TRAIN,
        ["gnss_drift"],
        seed=3,
        severities={"gnss_drift": 120.0},
    )
    # extract window features for the normal + corrupted CAL observations
    feats = np.array([gnss_window_features(o["GNSS"]) for o in cal_obs])
    assert labels.shape == (2,) and set(labels.tolist()) == {0.0, 1.0}
    member_matrix = np.array([agent.predict_normality(f) for f in feats])
    agent.fit_calibrator(member_matrix, labels)  # must NOT raise: both classes
    p_normal = agent.predict(feats[0], diagnostic=True).value
    p_pseudo = agent.predict(feats[1], diagnostic=True).value
    assert 0.0 <= p_pseudo < p_normal <= 1.0


def test_calibration_pairs_provenance_and_parent_link(observation):
    obs, labels, rows = build_calibration_pairs(
        [observation], CarlAnomalySplit.TRAIN, ["gnss_bias", "imu_spike"], seed=9
    )
    assert len(obs) == 3 and len(rows) == 3
    assert rows[0]["kind"] == "normal" and rows[0]["recipe_id"] is None
    assert rows[1]["recipe_id"] == "gnss_bias" and rows[2]["recipe_id"] == "imu_spike"
    for r in rows:
        assert r["source_split"] == "train"
        assert r["source_scenario"] == "train-scenario-7"
        assert r["source_tick"] == 42
    assert rows[1]["synthetic_corruption"] is True
    # every pseudo row keeps the parent link explicit
    assert rows[1]["source_scenario"] == rows[0]["source_scenario"]


def test_calibration_pairs_reject_test_observations(observation):
    tainted = dict(observation)
    tainted["__split__"] = "test_anomaly"
    with pytest.raises(PseudoAnomalyError):
        build_calibration_pairs([tainted], CarlAnomalySplit.TRAIN, ["gnss_bias"])


# ── 6. scientific wording ────────────────────────────────────────────────────

def test_module_does_not_claim_real_anomaly_replication():
    from cognix.adapters.carla import pseudo_anomalies
    src = inspect.getsource(pseudo_anomalies)
    assert "NOT replicas" in src
    assert "held-out" in src
    assert "synthetic_corruption" in src


def test_recipe_registry_is_auditable():
    reg = recipe_registry()
    assert len(reg) == 8
    ids = [rid for rid, _ in reg]
    assert len(set(ids)) == len(ids)
    mods = {m for _, m in reg}
    assert mods == {"camera", "seg", "gnss", "imu"}

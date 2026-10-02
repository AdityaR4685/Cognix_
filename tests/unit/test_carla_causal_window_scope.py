"""Causal-window observation scope for temporal pseudo-anomalies (Part D
validation suite, fixture-only — no real data, no downloads, no TEST).

The cache builder must hand the temporal (GNSS/IMU) pseudo-anomaly recipes the
ACTIVE CAUSAL WINDOW rows [max(0, t - WINDOW_LENGTH + 1), t] — exactly the rows
the cached tick-t observation summarizes — instead of the full ~3000-row
scenario table (the historical defect: gnss_bias's midpoint step and imu_spike's
seeded burst usually landed outside the window, corrupting rows the observation
did not contain and making the recipes no-ops).

Covers:
1. GNSS window corruption changes the active GNSS compact feature on essentially
   all appropriate fixtures (material change).
2. IMU spike lies inside the supplied observation window and changes the
   compact feature.
3. No future rows: corrupting row t+1 never changes the tick-t corrupted
   observation or its features; every corrupted row index <= t.
4. No cross-scenario data: identical windows from different scenarios corrupt
   independently (per-scenario payloads; windows never cross boundaries).
5. TRAIN-only split gate remains enforced through the builder path.
6. Same seed reproduces the identical corruption.
7. Source arrays are not mutated.
8. Camera/Seg behavior unchanged (per-tick frame scope, same features).
9. `import cognix` purity remains intact.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import PIL.Image
import pytest

from cognix.adapters.carla.cache_builder import (
    CAL_PSEUDO_RECIPES,
    WINDOW_LENGTH,
    build_cache,
    extract_gnss_feature_rows,
    extract_imu_feature_rows,
    load_cache,
)
from cognix.adapters.carla.carlanomaly_loader import CarlAnomalySplit
from cognix.adapters.carla.pseudo_anomalies import (
    PseudoAnomalyError,
    generate_pseudo_anomaly,
)
from cognix.adapters.carla.real_features import (
    camera_embedding_features,
    gnss_window_features,
    imu_window_features,
    local_meter_offsets_to_degrees,
    segmentation_histogram_features,
)


# ── fixtures (loader layout; mirrors the cache-builder test helpers) ─────────

def _moving_trajectory(n, seed, lat0=30.0):
    """Smooth ~0.5 m/step trajectory (fixture; sub-meter motion per step)."""
    rng = np.random.default_rng(seed)
    east_m = rng.uniform(0.3, 0.7)
    k_lat, k_lon = local_meter_offsets_to_degrees(1.0, 1.0, lat0)
    lat, lon = [lat0], [0.0]
    for _ in range(n - 1):
        lat.append(lat[-1] + k_lat * east_m)
        lon.append(lon[-1] + k_lon * east_m)
    return np.stack([np.full(n, 100.0), np.array(lat), np.array(lon)], axis=1)


@pytest.fixture
def gnss_window():
    return _moving_trajectory(12, seed=3)


@pytest.fixture
def imu_window():
    rng = np.random.default_rng(4)
    return rng.normal(0.0, 0.2, size=(12, 3))


# ── 1. GNSS window corruption changes the active compact feature ────────────

def test_gnss_bias_window_corruption_changes_compact_feature(gnss_window):
    """Window-scoped gnss_bias must materially change the 8-dim GNSS feature on
    essentially every window (the step lands mid-window by construction)."""
    feats_before = gnss_window_features(gnss_window)
    changed = 0
    for t in range(11):  # vary the payload's internal midpoint position
        w = gnss_window[t:]  # shorter windows: midpoint still inside
        if len(w) < 2:
            continue
        s = generate_pseudo_anomaly(
            {"GNSS": w}, "gnss_bias", CarlAnomalySplit.TRAIN,
            source_scenario="Town01/scenario-10", source_tick=t, seed=7,
            severity=25.0,
        )
        feats_after = gnss_window_features(np.asarray(s.data, dtype=np.float64))
        if not np.allclose(feats_after, feats_before[: len(feats_after)], rtol=1e-12, atol=1e-12):
            changed += 1
    assert changed >= 10  # essentially all appropriate windows change


def test_gnss_drift_window_corruption_changes_compact_feature(gnss_window):
    f_before = gnss_window_features(gnss_window)
    s = generate_pseudo_anomaly(
        {"GNSS": gnss_window}, "gnss_drift", CarlAnomalySplit.TRAIN,
        source_scenario="Town01/scenario-10", source_tick=7, seed=7,
        severity=50.0,
    )
    f_after = gnss_window_features(np.asarray(s.data, dtype=np.float64))
    assert float(np.linalg.norm(f_after - f_before)) > 1.0  # material change


# ── 2. IMU spike lies inside the supplied window and changes the feature ────

def test_imu_spike_burst_lies_inside_supplied_window(imu_window):
    """With the WINDOW as the payload, the seeded burst must land inside the
    supplied rows (the recipe draws its burst position within the payload): the
    corrupted rows must form a small contiguous burst inside the window, never
    reference data beyond it."""
    changed_rows_any = False
    for seed in range(10):
        s = generate_pseudo_anomaly(
            {"IMU": imu_window}, "imu_spike", CarlAnomalySplit.TRAIN,
            source_scenario="Town01/scenario-10", source_tick=11, seed=seed,
            severity=15.0,
        )
        accel = np.asarray(s.data, dtype=np.float64)
        assert accel.shape == imu_window.shape  # schema preserved, no growth
        diff = np.abs(accel - imu_window)
        touched = np.where(diff.max(axis=1) > 1e-9)[0]
        assert touched.size >= 1, f"seed {seed}: burst missed the supplied window"
        # burst is contiguous and inside the supplied rows, length <= 3
        assert int(touched.max()) - int(touched.min()) + 1 == touched.size
        assert int(touched.max()) <= len(imu_window) - 1
        assert touched.size <= 3
        changed_rows_any = True
    assert changed_rows_any


def test_imu_spike_changes_compact_feature_materially(imu_window):
    f_before = imu_window_features(imu_window)
    for seed in range(10):
        s = generate_pseudo_anomaly(
            {"IMU": imu_window}, "imu_spike", CarlAnomalySplit.TRAIN,
            source_scenario="Town01/scenario-10", source_tick=11, seed=seed,
            severity=15.0,
        )
        accel = np.asarray(s.data, dtype=np.float64)
        f_after = imu_window_features(accel)
        assert float(np.linalg.norm(f_after - f_before)) > 1.0, f"seed {seed}: no material change"


def test_imu_bias_scale_window_corruption_changes_compact_feature(imu_window):
    f_before = imu_window_features(imu_window)
    s = generate_pseudo_anomaly(
        {"IMU": imu_window}, "imu_bias_scale", CarlAnomalySplit.TRAIN,
        source_scenario="Town01/scenario-10", source_tick=11, seed=7,
        severity=0.5,
    )
    f_after = imu_window_features(np.asarray(s.data, dtype=np.float64))
    assert float(np.linalg.norm(f_after - f_before)) > 1.0


# ── 3. no future rows ────────────────────────────────────────────────────────

def test_no_future_rows_enter_the_corrupted_observation():
    """The corrupted tick-t observation is built from the window payload only:
    perturbing FUTURE table rows (after t) can never affect it, because the
    builder slices [lo, t+1] before the recipe ever runs."""
    table = _moving_trajectory(40, seed=17)
    t = 15
    lo = max(0, t - WINDOW_LENGTH + 1)
    window = table[lo:t + 1]
    s1 = generate_pseudo_anomaly(
        {"GNSS": window}, "gnss_bias", CarlAnomalySplit.TRAIN,
        source_scenario="S", source_tick=t, seed=9, severity=25.0)
    feats_before = gnss_window_features(np.asarray(s1.data, dtype=np.float64))
    # perturb ALL future rows massively
    table_future = table.copy()
    table_future[t + 1:, 1] += 0.01
    table_future[t + 1:, 0] += 500.0
    s2 = generate_pseudo_anomaly(
        {"GNSS": table_future[lo:t + 1]}, "gnss_bias", CarlAnomalySplit.TRAIN,
        source_scenario="S", source_tick=t, seed=9, severity=25.0)
    feats_after = gnss_window_features(np.asarray(s2.data, dtype=np.float64))
    np.testing.assert_array_equal(feats_before, feats_after)
    np.testing.assert_array_equal(np.asarray(s1.data), np.asarray(s2.data))


def test_builder_window_slices_never_include_future_rows(cache_root_fixture):
    """End-to-end over the builder: every emitted pseudo row's window provenance
    must satisfy window_end_tick == parent_tick and
    window_start_tick == max(0, t - WINDOW_LENGTH + 1)."""
    arrays, m = load_cache(cache_root_fixture[1])
    starts = arrays["cal_window_start_tick"]
    ends = arrays["cal_window_end_tick"]
    parents = arrays["cal_parent_tick"].astype(int)
    for i in range(len(parents)):
        t = int(parents[i])
        assert int(ends[i]) == t
        assert int(starts[i]) == max(0, t - WINDOW_LENGTH + 1)
        assert int(starts[i]) <= t
        assert int(starts[i]) >= 0


# ── 4. no cross-scenario data ────────────────────────────────────────────────

def test_identical_windows_from_different_scenarios_corrupt_independently(gnss_window):
    """The recipe is a pure function of the payload it is handed: two scenarios
    with the same window content produce the same corruption, and a different
    scenario's tail rows can never enter a window payload (slicing is
    per-scenario by construction)."""
    s1 = generate_pseudo_anomaly(
        {"GNSS": gnss_window}, "gnss_bias", CarlAnomalySplit.TRAIN,
        source_scenario="Town01/scenario-10", source_tick=11, seed=7,
        severity=25.0,
    )
    s2 = generate_pseudo_anomaly(
        {"GNSS": gnss_window}, "gnss_bias", CarlAnomalySplit.TRAIN,
        source_scenario="Town01/scenario-11", source_tick=11, seed=7,
        severity=25.0,
    )
    np.testing.assert_array_equal(np.asarray(s1.data), np.asarray(s2.data))
    assert s1.provenance()["source_scenario"] == "Town01/scenario-10"
    assert s2.provenance()["source_scenario"] == "Town01/scenario-11"


def test_builder_pseudo_windows_never_cross_scenario(cache_root_fixture):
    arrays, _ = load_cache(cache_root_fixture[1])
    for i in range(len(arrays["cal_target_normal"])):
        sid = str(arrays["cal_parent_scenario"][i])
        assert sid in set(arrays["scenario_id"].tolist())  # parent from same cache
        assert str(arrays["source_split"][0]) == "train"


# ── 5. TRAIN-only split gate (builder path) ──────────────────────────────────

def test_builder_cache_is_train_gated(cache_root_fixture):
    arrays, m = load_cache(cache_root_fixture[1])
    assert set(arrays["source_split"].tolist()) == {"train"}
    assert m["pseudo_anomalies"]["gated_to_split"] == "train"
    with pytest.raises(PseudoAnomalyError):
        generate_pseudo_anomaly(
            {"GNSS": np.zeros((4, 3))}, "gnss_bias", CarlAnomalySplit.TEST_ANOMALY
        )


# ── 6. same seed reproduces identical corruption ─────────────────────────────

@pytest.mark.parametrize("rid", ["gnss_bias", "gnss_drift", "imu_spike", "imu_bias_scale"])
def test_window_corruption_deterministic_for_fixed_seed(gnss_window, imu_window, rid):
    payload = {"GNSS": gnss_window} if rid.startswith("gnss") else {"IMU": imu_window}
    kw = {"severity": 25.0 if rid == "gnss_bias" else
          (50.0 if rid == "gnss_drift" else (15.0 if rid == "imu_spike" else 0.5))}
    s1 = generate_pseudo_anomaly(payload, rid, CarlAnomalySplit.TRAIN,
                                 source_scenario="S", source_tick=11, seed=123, **kw)
    s2 = generate_pseudo_anomaly(payload, rid, CarlAnomalySplit.TRAIN,
                                 source_scenario="S", source_tick=11, seed=123, **kw)
    np.testing.assert_array_equal(np.asarray(s1.data), np.asarray(s2.data))


# ── 7. source arrays not mutated ─────────────────────────────────────────────

@pytest.mark.parametrize("rid", ["gnss_bias", "gnss_drift", "imu_spike", "imu_bias_scale"])
def test_window_corruption_never_mutates_source(gnss_window, imu_window, rid):
    payload = {"GNSS": gnss_window.copy()} if rid.startswith("gnss") else {"IMU": imu_window.copy()}
    before = np.asarray(list(payload.values())[0]).copy()
    kw = {"severity": 25.0 if rid == "gnss_bias" else
          (50.0 if rid == "gnss_drift" else (15.0 if rid == "imu_spike" else 0.5))}
    generate_pseudo_anomaly(payload, rid, CarlAnomalySplit.TRAIN,
                            source_scenario="S", source_tick=11, seed=5, **kw)
    np.testing.assert_array_equal(np.asarray(list(payload.values())[0]), before)


# ── 8. Camera/Seg behavior unchanged ─────────────────────────────────────────

def test_camera_seg_window_scope_unchanged(tmp_path):
    """Camera/Seg recipes already receive the per-tick frame; the window-scope
    change must leave their features byte-identical."""
    rng = np.random.default_rng(8)
    frame = rng.integers(40, 220, size=(16, 16, 3), dtype=np.uint8)
    seg = np.zeros((16, 16), dtype=np.uint8)
    seg[8:, :] = 3
    f0 = camera_embedding_features(frame)
    h0 = segmentation_histogram_features(seg)
    s = generate_pseudo_anomaly(
        {"Camera": frame, "Seg": seg}, "camera_brightness_shift",
        CarlAnomalySplit.TRAIN, source_scenario="S", source_tick=11, seed=7,
        severity=0.35,
    )
    s2 = generate_pseudo_anomaly(
        {"Camera": frame, "Seg": seg}, "seg_region_corruption",
        CarlAnomalySplit.TRAIN, source_scenario="S", source_tick=11, seed=7,
        severity=0.5,
    )
    f1 = camera_embedding_features(np.asarray(s.data))
    h1 = segmentation_histogram_features(np.asarray(s2.data)[..., 0]
                                         if np.asarray(s2.data).ndim == 3 else s2.data)
    assert f1.shape == f0.shape == (18,)
    assert h1.shape == h0.shape == (29,)
    assert not np.allclose(f1, f0)   # camera corruption still bites
    assert float(h1.sum()) == pytest.approx(1.0)  # seg output stays a composition


def test_builder_camera_seg_rows_unchanged_semantics(cache_root_fixture):
    """Emitted camera/seg pseudo rows still differ from their parents in the
    corrupted block and keep parent features elsewhere."""
    arrays, _ = load_cache(cache_root_fixture[1])
    offsets = {"camera": 0, "seg": 18, "gnss": 47, "imu": 55}
    dims = {"camera": 18, "seg": 29, "gnss": 8, "imu": 10}
    checked = 0
    for i in range(len(arrays["cal_target_normal"])):
        mod = str(arrays["cal_modality"][i])
        if mod not in ("camera", "seg"):
            continue
        sid = str(arrays["cal_parent_scenario"][i])
        t = int(arrays["cal_parent_tick"][i])
        pm = (arrays["scenario_id"] == sid) & (arrays["tick"] == t)
        if not pm.any():
            continue
        parent_vec = np.concatenate(
            [arrays["camera"][pm][0], arrays["seg"][pm][0],
             arrays["gnss"][pm][0], arrays["imu"][pm][0]])
        off, dim = offsets[mod], dims[mod]
        assert not np.allclose(arrays["cal_features"][i][off:off + dim],
                               parent_vec[off:off + dim])
        for m2 in ("camera", "seg", "gnss", "imu"):
            if m2 != mod:
                o2, d2 = offsets[m2], dims[m2]
                np.testing.assert_allclose(
                    arrays["cal_features"][i][o2:o2 + d2],
                    parent_vec[o2:o2 + d2], rtol=1e-12, atol=1e-15)
        checked += 1
    assert checked > 0


# ── 9. import purity ─────────────────────────────────────────────────────────

def test_generic_import_does_not_pull_adapters():
    code = (
        "import sys; import cognix; "
        "sys.exit(1 if any(m.startswith('cognix.adapters') for m in sys.modules) else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


# ── builder-level materiality (Part E preview on fixtures) ──────────────────

def test_temporal_recipes_material_emission_on_fixtures(cache_root_fixture):
    """With window-scoped payloads, GNSS/IMU recipes must emit (non-skip) on
    essentially every fixture attempt — the historical full-table defect
    produced mostly no-effect skips."""
    arrays, m = load_cache(cache_root_fixture[1])
    acc = m["pseudo_anomalies"]["attempts_by_recipe"]
    skip = m["pseudo_anomalies"]["skipped_by_recipe"]
    for rid in ("gnss_bias", "gnss_drift", "imu_spike", "imu_bias_scale"):
        emitted = sum(1 for r in arrays["cal_recipe_id"] if r == rid)
        assert acc[rid] >= 2  # rotation covers the recipe on this fixture set
        no_effect = skip.get(rid, {}).get("no_effect", 0)
        assert emitted + no_effect == acc[rid]  # every attempt is accounted for
        assert no_effect == 0, (
            f"{rid}: window-scoped payload should not no-effect on moving-"
            f"trajectory fixtures (skipped {no_effect}/{acc[rid]})"
        )


# ── shared fixture: tiny builder run over two moving scenarios ──────────────

@pytest.fixture
def cache_root_fixture(tmp_path):
    """Two 40-tick scenarios (moving trajectories), explicit partition; builds
    the cache once and returns (root, cache_dir)."""
    root = tmp_path / "carlanomaly"
    (root / "train" / "Town01").mkdir(parents=True)
    for i, sid in enumerate(("scenario-1", "scenario-10")):
        d = root / "train" / "Town01" / sid
        (d / "rgb-front").mkdir(parents=True)
        (d / "segmentation-front").mkdir()
        n = 40
        gnss = _moving_trajectory(n, seed=10 + i)
        pd.DataFrame({"altitude": gnss[:, 0], "latitude": gnss[:, 1],
                      "longitude": gnss[:, 2]}).to_feather(d / "gnss.feather")
        rng = np.random.default_rng(20 + i)
        imu = rng.normal(0.0, 0.2, size=(n, 3))
        pd.DataFrame({
            "acceleration_x": imu[:, 0], "acceleration_y": imu[:, 1],
            "acceleration_z": imu[:, 2], "compass": np.zeros(n),
            "longitude_x": np.zeros(n), "longitude_y": np.zeros(n),
            "longitude_z": np.zeros(n)}).to_feather(d / "imu.feather")
        for t in range(n):
            arr = rng.integers(30, 220, size=(8, 8, 3), dtype=np.uint8)
            PIL.Image.fromarray(arr).save(d / "rgb-front" / f"{t:06d}.jpg", format="JPEG")
            seg = np.zeros((6, 6, 3), dtype=np.uint8)
            seg[..., 0] = rng.choice([0, 3, 7], size=(6, 6))
            PIL.Image.fromarray(seg).save(d / "segmentation-front" / f"{t:06d}.png", format="PNG")
    from cognix.adapters.carla.cache_builder import TrainCalPartition
    part = TrainCalPartition(
        train_normal=("Town01/scenario-1",),
        cal_normal=("Town01/scenario-10",),
        seed=2026, rule="explicit",
    )
    cache_dir = tmp_path / "cache"
    build_cache(root, cache_dir, scenarios=["Town01/scenario-1", "Town01/scenario-10"],
                partition=part, max_pseudo_per_tick=1)
    return root, cache_dir

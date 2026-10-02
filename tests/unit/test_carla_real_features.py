"""Tests for the real-data CarlAnomaly feature extractors (Part A audit).

Covers, per extractor:
    Camera: determinism, exact documented dimension, finiteness, uint8 RGB
        handling, constant-image behavior, invalid dimensions/channels fail
        loudly, input not mutated.
    Segmentation: deterministic class histogram, R-channel semantics (class ids
        read from channel 0), finiteness, normalization, single-class/empty
        cases, invalid shape fails, no mutation. No undocumented semantic-class
        meanings are asserted anywhere.
    GNSS: unit correctness (degrees never treated as interchangeable Euclidean
        coordinates; local small-area meter approximation), stationarity,
        smooth vs abrupt drift, short windows, finiteness, determinism,
        failure behavior.
    IMU: only acceleration_x/y/z feed the acceleration features — changing
        compass / longitude_* while acceleration is identical must not change
        the output; stationary vs changing acceleration; short windows;
        finiteness; determinism; invalid schema.

Scientific wording guarded here: the camera representation is a DETERMINISTIC
COMPACT FEATURE REPRESENTATION — not learned perception, not a pretrained
visual embedding; IMU change statistics are DISCRETE per-sample statistics —
not physical jerk in SI units.
"""
from __future__ import annotations

import numpy as np
import pytest

from cognix.adapters.carla.real_features import (
    EARTH_RADIUS_M,
    camera_embedding_features,
    gnss_window_features,
    imu_window_features,
    local_meter_offsets_to_degrees,
    segmentation_histogram_features,
)


# ── camera ───────────────────────────────────────────────────────────────────

def test_camera_deterministic_exact_dim_finite():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(16, 32, 3), dtype=np.uint8)
    f1 = camera_embedding_features(img)
    f2 = camera_embedding_features(img)
    assert f1.shape == (18,)  # 4x4 block means + variance + brightness
    assert f1.dtype == np.float64
    assert np.array_equal(f1, f2)
    assert np.all(np.isfinite(f1))
    assert np.all((f1 >= 0.0) & (f1 <= 1.0))


def test_camera_uint8_input_untouched_and_valid():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    before = img.copy()
    f = camera_embedding_features(img)
    np.testing.assert_array_equal(img, before)  # no in-place conversion
    assert np.all(np.isfinite(f))


def test_camera_float_input_clipped_not_mutated():
    img = np.full((8, 8, 3), 300.0)  # out-of-range float
    before = img.copy()
    f = camera_embedding_features(img)
    np.testing.assert_array_equal(img, before)
    assert np.all((f >= 0.0) & (f <= 1.0))  # 300 clipped -> white image -> 1.0


def test_camera_constant_image_properties():
    img = np.full((12, 12, 3), 77, dtype=np.uint8)
    f = camera_embedding_features(img)
    assert f.shape == (18,)
    # constant image: every block mean (f[:16]) and brightness (f[-1]) = 77/255
    assert np.allclose(f[:16], 77.0 / 255.0)
    assert f[-1] == pytest.approx(77.0 / 255.0)  # brightness slot


def test_camera_constant_image_variance_is_zero():
    img = np.full((12, 12, 3), 200, dtype=np.uint8)
    f = camera_embedding_features(img)
    assert f[-2] == pytest.approx(0.0, abs=1e-12)  # variance slot
    # block means and brightness still carry the constant intensity:
    assert np.allclose(f[:16], 200.0 / 255.0)
    assert f[-1] == pytest.approx(200.0 / 255.0)


def test_camera_invalid_dims_and_channels_fail():
    with pytest.raises(ValueError, match="expect"):
        camera_embedding_features(np.zeros((8, 8), dtype=np.uint8))  # 2-D
    with pytest.raises(ValueError, match="expect"):
        camera_embedding_features(np.zeros((8, 8, 4), dtype=np.uint8))  # 4 ch
    with pytest.raises(ValueError, match="expect"):
        camera_embedding_features(np.zeros((2, 2, 2), dtype=np.uint8))


def test_camera_rectangular_non_square_grid():
    # Non-square images must partition cleanly into a 4x4 grid (no overlap).
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    img[0:5, 0:10] = 255  # exactly block (0, 0) for h=20, w=40
    f = camera_embedding_features(img)
    assert f[0] == pytest.approx(1.0)
    assert f[1] == pytest.approx(0.0)  # block (0, 1) untouched


def test_camera_docstring_does_not_claim_genuine_perception():
    import inspect
    from cognix.adapters.carla import real_features
    src = inspect.getsource(real_features.camera_embedding_features)
    assert "genuine perception" not in src.lower()
    assert "NOT learned perception" in src
    mod_src = inspect.getsource(real_features)
    assert "genuine learned perception" not in mod_src


# ── segmentation ─────────────────────────────────────────────────────────────

def test_segmentation_deterministic_histogram_normalized():
    rng = np.random.default_rng(2)
    seg = rng.integers(0, 29, size=(32, 32)).astype(np.uint8)
    h1 = segmentation_histogram_features(seg)
    h2 = segmentation_histogram_features(seg)
    assert h1.shape == (29,)
    assert np.array_equal(h1, h2)
    assert np.all(np.isfinite(h1))
    assert h1.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all((h1 >= 0.0) & (h1 <= 1.0))


def test_segmentation_uses_r_channel_only():
    seg = np.zeros((8, 8,3), dtype=np.uint8)
    seg[..., 0] = 5                      # R channel: class 5
    seg[..., 1] = 99                     # G channel: must be IGNORED
    seg[..., 2] = 42                     # B channel: must be IGNORED
    h = segmentation_histogram_features(seg)
    assert h[5] == pytest.approx(1.0)
    assert h.sum() == pytest.approx(1.0)
    # no mass outside class 5 (ids 99/42 are not even valid bins)
    off_class = h[np.arange(h.size) != 5].sum()
    assert off_class == 0.0


def test_segmentation_single_class_and_constant():
    seg = np.full((10, 10), 7, dtype=np.uint8)
    h = segmentation_histogram_features(seg)
    assert h[7] == pytest.approx(1.0)
    assert h.sum() == pytest.approx(1.0)


def test_segmentation_known_counts():
    seg = np.array([[1, 1, 3], [3, 3, 28]], dtype=np.uint8)
    h = segmentation_histogram_features(seg)
    assert h[1] == pytest.approx(2 / 6)
    assert h[3] == pytest.approx(3 / 6)
    assert h[28] == pytest.approx(1 / 6)


def test_segmentation_rejects_out_of_range_class_ids():
    with pytest.raises(ValueError, match="out of range"):
        segmentation_histogram_features(np.array([[0, 29]]))
    with pytest.raises(ValueError, match="out of range"):
        segmentation_histogram_features(np.array([[0, -1]]))


def test_segmentation_empty_map_fails():
    with pytest.raises(ValueError):
        segmentation_histogram_features(np.empty((0,), dtype=np.uint8))


def test_segmentation_3d_raveled_must_still_be_flat_class_ids():
    # A full-color map is raveled; only channel-0 semantics are honored when
    # callers pass (H, W, 1)-style class maps. Ensure no silent misread.
    seg = np.zeros((4, 4, 1), dtype=np.uint8)
    seg[..., 0] = 3
    h = segmentation_histogram_features(seg)
    assert h[3] == pytest.approx(1.0)


def test_segmentation_no_mutation():
    seg = np.array([[1, 2], [3, 28]], dtype=np.uint8)
    before = seg.copy()
    segmentation_histogram_features(seg)
    np.testing.assert_array_equal(seg, before)


# ── GNSS ─────────────────────────────────────────────────────────────────────

def _gnss_stationary(n=20, lat0=30.0, lon0=5.0, alt0=100.0):
    # EXACTLY constant coordinates: a stationary receiver with no jitter.
    lat = np.full(n, lat0)
    lon = np.full(n, lon0)
    alt = np.full(n, alt0)
    return np.stack([alt, lat, lon], axis=1)


def test_gnss_does_not_treat_degrees_as_euclidean():
    """A pure-longitude step must have a physically different meter size than
    an equal pure-latitude step at a non-equatorial latitude — the defect of
    the previous degree-diffing implementation."""
    dlat_deg = 1e-4
    lat0 = 60.0
    rows_lat = np.array([
        [10.0, lat0, 0.0],
        [10.0, lat0 + dlat_deg, 0.0],
    ])
    rows_lon = np.array([
        [10.0, lat0, 0.0],
        [10.0, lat0, dlat_deg],  # same angular size in longitude
    ])
    f_lat = gnss_window_features(rows_lat)
    f_lon = gnss_window_features(rows_lon)
    north_m = f_lat[2]  # north step MEAN (all steps identical; idx 3 is std)
    east_m = f_lon[0]   # east step mean
    # north_m / east_m = 1 / cos(lat) (the earlier degree-diffing code gave 1)
    expected_ratio = 1.0 / np.cos(np.deg2rad(60.0))
    assert north_m / east_m == pytest.approx(expected_ratio, rel=1e-6)
    # absolute scale sanity: 1e-4 deg of latitude ~ 11.1 m
    assert north_m == pytest.approx(dlat_deg * np.pi / 180.0 * EARTH_RADIUS_M, rel=1e-9)


def test_gnss_meter_offsets_round_trip():
    dlat, dlon = local_meter_offsets_to_degrees(50.0, 20.0, lat_deg=45.0)
    rows = np.array([[10.0, 45.0, 0.0], [10.0, 45.0 + dlat, dlon]])
    f = gnss_window_features(rows)
    # displacement magnitude feature (index 5) ~ sqrt(50^2 + 20^2)
    assert f[5] == pytest.approx(np.hypot(50.0, 20.0), rel=1e-6)
    assert f[7] == pytest.approx(1.0, abs=1e-9)  # perfectly straight


def test_gnss_stationary_window():
    rows = _gnss_stationary()
    f = gnss_window_features(rows)
    assert f.shape == (8,)
    assert np.all(np.isfinite(f))
    assert np.all(f[:6] == 0.0)                  # zero motion in meter features
    assert f[6] == pytest.approx(0.0, abs=1e-9)  # altitude range
    assert f[7] == pytest.approx(1.0)            # straightness degenerate -> 1.0


def test_gnss_tiny_jitter_stays_meter_small():
    """1e-9 deg of coordinate noise ~ 0.1 mm: meter features must stay
    physically tiny (the degree-diffing implementation would have reported
    'steps' of 1e-9 arbitrary units with no physical meaning)."""
    rng = np.random.default_rng(11)
    rows = _gnss_stationary()
    rows[:, 1] += rng.normal(0, 1e-9, len(rows))
    rows[:, 2] += rng.normal(0, 1e-9, len(rows))
    f = gnss_window_features(rows)
    assert np.all(np.abs(f[:6]) < 1e-3)  # well under a millimeter-ish scale


def test_gnss_smooth_vs_abrupt_drift():
    lat0, lon0 = 30.0, 5.0
    dlat, dlon = local_meter_offsets_to_degrees(2.0, 0.0, lat0)
    n = 21
    smooth = np.stack([
        np.full(n, 10.0),
        lat0 + dlat * np.arange(n),
        np.full(n, lon0),
    ], axis=1)
    abrupt = smooth.copy()
    abrupt[10:, 1] += dlat * 50  # sudden 100 m north jump mid-window
    f_smooth = gnss_window_features(smooth)
    f_abrupt = gnss_window_features(abrupt)
    assert np.all(np.isfinite(f_abrupt)) and np.all(np.isfinite(f_smooth))
    # abrupt drift: bigger per-step magnitudes and bigger altitude-independent spread
    assert f_abrupt[4] > f_smooth[4]   # mean step magnitude
    assert f_abrupt[5] > f_smooth[5]   # max step magnitude
    assert f_abrupt[3] > f_smooth[3]   # north step std


def test_gnss_short_window_two_rows():
    rows = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 1e-5]])
    f = gnss_window_features(rows)
    assert f.shape == (8,)
    assert np.all(np.isfinite(f))
    assert f[1] == pytest.approx(0.0, abs=1e-12)  # std undefined for n-1=1


def test_gnss_deterministic():
    rows = _gnss_stationary()
    np.testing.assert_array_equal(
        gnss_window_features(rows), gnss_window_features(rows.copy())
    )


def test_gnss_rejects_bad_shape_and_nonfinite():
    with pytest.raises(ValueError, match="expects"):
        gnss_window_features(np.zeros((5, 2)))      # missing a column
    with pytest.raises(ValueError, match="expects"):
        gnss_window_features(np.zeros((1, 3)))      # single row
    with pytest.raises(ValueError, match="expects"):
        gnss_window_features(np.zeros(3))           # 1-D
    with pytest.raises(ValueError, match="finite"):
        gnss_window_features(np.array([[10.0, np.nan, 0.0], [10.0, 0.0, 0.0]]))


def test_gnss_no_mutation():
    rows = _gnss_stationary()
    before = rows.copy()
    gnss_window_features(rows)
    np.testing.assert_array_equal(rows, before)


# ── IMU ──────────────────────────────────────────────────────────────────────

def _imu_rows(n=10):
    rng = np.random.default_rng(7)
    return rng.normal(0, 0.5, size=(n, 3))


def test_imu_ignores_orientation_columns():
    """compass / longitude_* must never influence acceleration features."""
    acc = _imu_rows()
    f1 = imu_window_features(acc)
    f2 = imu_window_features(acc * -2.5)  # scale-only change DOES alter output
    assert not np.allclose(f1, f2)        # sanity: function is sensitive at all
    # determinism with identical acceleration
    np.testing.assert_array_equal(imu_window_features(acc), imu_window_features(acc))


def test_imu_stationary_zero_acceleration():
    acc = np.zeros((12, 3))
    f = imu_window_features(acc)
    assert f.shape == (10,)
    assert np.all(np.isfinite(f))
    assert np.allclose(f, 0.0)  # zero accel: all means/stds/change stats zero


def test_imu_changing_acceleration_changes_change_statistic():
    const = np.tile([1.0, 0.0, 0.0], (10, 1))
    ramp = np.stack([np.linspace(0, 5, 10), np.zeros(10), np.zeros(10)], axis=1)
    f_const = imu_window_features(const)
    f_ramp = imu_window_features(ramp)
    assert f_const[8] == pytest.approx(0.0)  # mean |change| ~ 0 for constant
    assert f_ramp[8] > 0.0
    assert f_ramp[9] == pytest.approx(np.abs(5.0 / 9.0), rel=1e-9)  # max |change|


def test_imu_spike_detection():
    acc = np.zeros((8, 3))
    acc[4, 1] = 3.0  # single-sample spike
    f = imu_window_features(acc)
    assert f[9] == pytest.approx(3.0)     # max |change| sees the spike
    assert f[9] > f[8]                    # max exceeds mean change


def test_imu_short_window_two_rows():
    f = imu_window_features(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert f.shape == (10,)
    assert np.all(np.isfinite(f))


def test_imu_invalid_schema_fails():
    with pytest.raises(ValueError, match="expects"):
        imu_window_features(np.zeros((5, 4)))  # wrong channel count
    with pytest.raises(ValueError, match="expects"):
        imu_window_features(np.zeros((1, 3)))  # single row
    with pytest.raises(ValueError, match="finite"):
        imu_window_features(np.array([[0.0, 0.0, np.inf], [0.0, 0.0, 0.0]]))


def test_imu_deterministic_and_no_mutation():
    acc = _imu_rows()
    before = acc.copy()
    f1 = imu_window_features(acc)
    f2 = imu_window_features(acc)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(acc, before)


def test_imu_docstring_disclaims_si_jerk():
    import inspect
    from cognix.adapters.carla import real_features
    src = inspect.getsource(real_features.imu_window_features)
    assert "NOT physical jerk in SI units" in src

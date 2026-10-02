"""
Unit tests for the offline CarlAnomaly loader (read/validate only).

Covers:
1.  Fail-loud on missing root / missing split / malformed scenario IDs.
2.  Scenario discovery for train, test_normal, test_anomaly (nested anomaly-type).
3.  Strict town-layout validation.
4.  Explicit modality manifest (present modalities declared; absent never zero-filled).
5.  Timestep ground truth from anomaly-observation.feather.
6.  Strict scenario/timestep synchronization (count, range, monotonic ticks).
7.  Compact bridge statistics computed from real fixture data (variance/brightness,
    min_depth, density, entropy, GNSS jump, IMU jerk) with hand-computed values.
8.  require() raises on absent modalities.
9.  Frame fields for absent modalities are None (explicit absence), not zeros.
10. Split isolation: train scenarios are not reachable via test splits.
11. Verified real-dataset schemas (empirical probe of carlanomaly-base-test.tar.gz):
    - gnss.feather / imu.feather have NO tick column -> positional alignment
    - anomaly-observation.feather carries an optional 'description' column
    - depth PNGs are 2-D grayscale (mode 'L'); 3-channel handled defensively
    - scenario-level anomaly category is independent of per-timestep labels
      (all labels may be False inside an anomaly-directory scenario)

No dataset download, no training, no artifact writes, no synthetic-path changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import PIL.Image
import pytest

from cognix.adapters.carla.carlanomaly_loader import (
    CarlAnomalyLoader,
    CarlAnomalyLoaderError,
    CarlAnomalySplit,
    Modality,
)
from cognix.adapters.carla.dataset import CarlAnomalyFrame


# ── fixture construction helpers ─────────────────────────────────────────────

def _write_rgb(path, half_black_white: bool = False, size: tuple = (4, 4)):
    """Write a small JPEG. Default: uniform gray (var≈0). Optional: half black/half white."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if half_black_white:
        arr = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        arr[: size[0] // 2] = 255
    else:
        arr = np.full((size[0], size[1], 3), 100, dtype=np.uint8)
    PIL.Image.fromarray(arr).save(path, format="JPEG")


def _write_depth(path, r_values, three_channel: bool = False):
    """Write a depth PNG.

    Default: 2-D grayscale uint8 (PIL mode 'L') — the format verified in the
    official CarlAnomaly depth example. With three_channel=True, writes an RGB
    PNG whose R channel carries the same values (defensive-format coverage).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(r_values, dtype=np.uint8)
    if three_channel:
        arr = np.zeros((len(vals), 1, 3), dtype=np.uint8)
        arr[:, 0, 0] = vals
    else:
        arr = vals.reshape(-1, 1)  # 2-D grayscale, one pixel per value
    PIL.Image.fromarray(arr).save(path, format="PNG")


def _write_seg(path, r_values):
    """Write a segmentation PNG whose R channel encodes per-pixel classes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(r_values)
    arr = np.zeros((n, 1, 3), dtype=np.uint8)
    arr[:, 0, 0] = np.asarray(r_values, dtype=np.uint8)
    PIL.Image.fromarray(arr).save(path, format="PNG")


def _write_feather(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(path)


def _make_scenario(
    root,
    rel: str,
    n_ticks: int,
    labels,
    with_gnss: bool = True,
    with_imu: bool = True,
    with_labels: bool = True,
    with_depth: bool = False,
    with_lidar: bool = False,
    with_seg: bool = False,
    anomaly_at: int | None = None,
    depth_three_channel: bool = False,
    labels_all_false: bool = False,
):
    """Create one scenario directory with the official layout under root/rel."""
    scen = root / rel
    for t in range(n_ticks):
        _write_rgb(scen / "rgb-front" / f"{t:06d}.jpg", half_black_white=(t == anomaly_at))
    if with_depth:
        # depth values chosen so min_depth is distinguishable per tick
        _write_depth(scen / "depth-front" / f"{n_ticks - 1:06d}.png", [25, 255, 255],
                     three_channel=depth_three_channel)
        for t in range(n_ticks - 1):
            _write_depth(scen / "depth-front" / f"{t:06d}.png", [25, 255, 255],
                         three_channel=depth_three_channel)
    if with_lidar:
        n_pts = 5000
        df = pd.DataFrame(
            {
                "x": np.zeros(n_pts),
                "y": np.zeros(n_pts),
                "z": np.zeros(n_pts),
                "angle": np.zeros(n_pts),
                "object_id": np.zeros(n_pts, dtype=np.int64),
                "class_id": np.zeros(n_pts, dtype=np.int64),
            }
        )
        for t in range(n_ticks):
            _write_feather(scen / "pointclouds" / f"{t:06d}.feather", df)
    if with_seg:
        for t in range(n_ticks):
            _write_seg(scen / "segmentation-front" / f"{t:06d}.png", [0, 0, 0, 7])
    if with_gnss:
        # Verified real gnss.feather schema: altitude, latitude, longitude
        # (float64) and NO tick/frame column — rows align positionally.
        gnss = pd.DataFrame(
            {
                "altitude": [0.0, 0.0, 0.0, 0.0][:n_ticks],
                "latitude": [0.0, 1.0, 1.0, 2.0][:n_ticks],
                "longitude": [0.0, 0.0, 1.0, 1.0][:n_ticks],
            }
        )
        _write_feather(scen / "gnss.feather", gnss)
    if with_imu:
        # Verified real imu.feather schema: acceleration_x/y/z, compass,
        # longitude_x/y/z (float64) and NO tick/frame column.
        imu = pd.DataFrame(
            {
                "acceleration_x": [0.0, 1.0, 1.0, 2.0][:n_ticks],
                "acceleration_y": [0.0, 0.0, 1.0, 1.0][:n_ticks],
                "acceleration_z": [0.0, 0.0, 0.0, 0.0][:n_ticks],
                "compass": [0.0, 0.0, 0.0, 0.0][:n_ticks],
                "longitude_x": [0.0, 0.0, 0.0, 0.0][:n_ticks],
                "longitude_y": [0.0, 0.0, 0.0, 0.0][:n_ticks],
                "longitude_z": [0.0, 0.0, 0.0, 0.0][:n_ticks],
            }
        )
        _write_feather(scen / "imu.feather", imu)
    if with_labels:
        if labels_all_false:
            label_vals = [False] * n_ticks
        else:
            label_vals = [t == anomaly_at for t in range(n_ticks)]
        # Verified real schema: anomaly (bool), tick (int64), description (str)
        obs = pd.DataFrame(
            {
                "tick": list(range(n_ticks)),
                "anomaly": label_vals,
                "description": [
                    "clear" if not v else "debris" for v in label_vals
                ],
            }
        )
        _write_feather(scen / "anomaly-observation.feather", obs)
    return scen


@pytest.fixture
def root(tmp_path):
    """Tiny fixture tree mimicking the official CarlAnomaly layout."""
    # train: 1 normal scenario, 4 ticks, no anomaly, with depth/lidar/seg
    _make_scenario(
        tmp_path, "train/Town01/scenario-1", 4,
        labels=None, with_depth=True, with_lidar=True, with_seg=True, anomaly_at=None,
    )
    # test/normal: 1 scenario, 3 ticks, base-only (no depth/lidar/seg)
    _make_scenario(tmp_path, "test/normal/Town01/scenario-1", 3, labels=None)
    # test/anomaly: debris-spawn scenario, 4 ticks, anomaly at tick 2 and 3
    _make_scenario(
        tmp_path, "test/anomaly/Town02/debris-spawn/scenario-1", 4, labels=None, anomaly_at=2,
    )
    # test/anomaly second town/type to test town discovery
    _make_scenario(
        tmp_path, "test/anomaly/Town01/vanish-actor/scenario-9", 3, labels=None, anomaly_at=0,
    )
    # change-weather-like scenario: anomaly-split directory but ALL timestep
    # labels False (verified pattern from the real change-weather/scenario-1)
    _make_scenario(
        tmp_path, "test/anomaly/Town01/change-weather/scenario-1", 3, labels=None,
        labels_all_false=True,
    )
    return tmp_path


def _loader(root) -> CarlAnomalyLoader:
    return CarlAnomalyLoader(root=root, strict_layout=True)


# ── 1. fail-loud construction ────────────────────────────────────────────────

def test_missing_root_raises():
    with pytest.raises(CarlAnomalyLoaderError, match="CARLANOMALY_ROOT"):
        CarlAnomalyLoader(root=None)


def test_nonexistent_root_raises(tmp_path):
    with pytest.raises(CarlAnomalyLoaderError, match="does not exist"):
        CarlAnomalyLoader(root=tmp_path / "nope")


def test_missing_split_raises(root):
    # train split exists in fixture; build a loader on a root that lacks test/
    import shutil
    partial = root / "_partial"
    partial.mkdir()
    shutil.copytree(root / "train", partial / "train")
    l2 = CarlAnomalyLoader(root=partial)
    with pytest.raises(CarlAnomalyLoaderError, match="test_anomaly"):
        l2.list_scenarios(CarlAnomalySplit.TEST_ANOMALY)


# ── 2. scenario discovery ────────────────────────────────────────────────────

def test_list_scenarios_train(root):
    loader = _loader(root)
    assert loader.list_scenarios(CarlAnomalySplit.TRAIN) == ["Town01/scenario-1"]


def test_list_scenarios_test_normal(root):
    loader = _loader(root)
    assert loader.list_scenarios(CarlAnomalySplit.TEST_NORMAL) == ["Town01/scenario-1"]


def test_list_scenarios_test_anomaly_nested_by_type(root):
    loader = _loader(root)
    scen = loader.list_scenarios(CarlAnomalySplit.TEST_ANOMALY)
    assert "Town02/debris-spawn/scenario-1" in scen
    assert "Town01/vanish-actor/scenario-9" in scen
    assert "Town01/change-weather/scenario-1" in scen
    assert len(scen) == 3


def test_anomaly_type_from_directory(root):
    loader = _loader(root)
    assert loader.anomaly_type_of(CarlAnomalySplit.TRAIN, "Town01/scenario-1") == "NORMAL"
    assert loader.anomaly_type_of(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1") == "NORMAL"
    assert (
        loader.anomaly_type_of(CarlAnomalySplit.TEST_ANOMALY, "Town02/debris-spawn/scenario-1")
        == "debris-spawn"
    )


def test_malformed_scenario_id_raises(root):
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="Malformed scenario ID"):
        loader.scenario_path(CarlAnomalySplit.TEST_ANOMALY, "Town02/debris-spawn")


# ── 3. strict town layout ────────────────────────────────────────────────────

def test_strict_layout_rejects_unknown_town(root):
    loader = CarlAnomalyLoader(root=root, strict_layout=True)
    bogus = root / "test" / "normal" / "NotATown"
    bogus.mkdir()
    try:
        with pytest.raises(CarlAnomalyLoaderError, match="Unexpected town directories"):
            loader.list_towns(CarlAnomalySplit.TEST_NORMAL)
    finally:
        bogus.rmdir()


def test_non_strict_layout_accepts_unknown_town(root):
    (root / "test" / "normal" / "CustomTown").mkdir()
    loader = CarlAnomalyLoader(root=root, strict_layout=False)
    assert "CustomTown" in loader.list_towns(CarlAnomalySplit.TEST_NORMAL)


# ── 4. manifest ──────────────────────────────────────────────────────────────

def test_manifest_full_scenario(root):
    loader = _loader(root)
    m = loader.build_manifest(CarlAnomalySplit.TRAIN, "Town01/scenario-1")
    assert m.n_ticks == 4
    assert Modality.RGB_FRONT in m.available_modalities
    assert Modality.DEPTH_FRONT in m.available_modalities
    assert Modality.LIDAR in m.available_modalities
    assert Modality.SEGMENTATION_FRONT in m.available_modalities
    assert Modality.GNSS in m.available_modalities
    assert Modality.IMU in m.available_modalities
    assert Modality.TIMESTEP_LABELS in m.available_modalities
    assert m.n_rgb_ticks == 4
    assert m.n_feather_rows["gnss"] == 4
    assert m.anomaly_type == "NORMAL"


def test_manifest_base_only_scenario_declares_absent_modalities(root):
    loader = _loader(root)
    m = loader.build_manifest(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1")
    assert Modality.RGB_FRONT in m.available_modalities
    assert Modality.DEPTH_FRONT not in m.available_modalities
    assert Modality.LIDAR not in m.available_modalities
    assert Modality.SEGMENTATION_FRONT not in m.available_modalities
    assert m.n_depth_ticks is None
    assert m.n_lidar_ticks is None


def test_require_raises_on_missing_modality(root):
    loader = _loader(root)
    m = loader.build_manifest(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1")
    with pytest.raises(CarlAnomalyLoaderError, match="depth_front"):
        m.require(Modality.DEPTH_FRONT)


# ── 5. timestep ground truth ────────────────────────────────────────────────

def test_timestep_labels_train_all_normal(root):
    loader = _loader(root)
    labels = loader.load_timestep_labels(CarlAnomalySplit.TRAIN, "Town01/scenario-1")
    assert [lb.label for lb in labels] == [0, 0, 0, 0]
    assert [lb.tick for lb in labels] == [0, 1, 2, 3]
    assert all(lb.anomaly_type == "NORMAL" for lb in labels)


def test_timestep_labels_anomaly_ticks(root):
    loader = _loader(root)
    labels = loader.load_timestep_labels(
        CarlAnomalySplit.TEST_ANOMALY, "Town02/debris-spawn/scenario-1"
    )
    # fixture marks tick 2 (and only tick 2) as anomalous
    assert [lb.label for lb in labels] == [0, 0, 1, 0]
    assert all(lb.anomaly_type == "debris-spawn" for lb in labels)


def test_missing_label_table_fails_loud(root):
    scen = root / "test" / "normal" / "Town02"
    _make_scenario(root, "test/normal/Town02/scenario-x", 2, labels=None, with_labels=False)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="missing modalities"):
        loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town02/scenario-x")


# ── 6. synchronization ──────────────────────────────────────────────────────

def test_label_count_mismatch_raises(root):
    # scenario with 3 frames but only 2 label rows
    scen = _make_scenario(root, "test/normal/Town03/scenario-bad", 3, labels=None)
    df = pd.DataFrame({"tick": [0, 1], "anomaly": [False, False]})
    _write_feather(scen / "anomaly-observation.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="does not match synchronized frame count"):
        loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-bad")


def test_label_range_mismatch_raises(root):
    scen = _make_scenario(root, "test/normal/Town03/scenario-shift", 3, labels=None)
    df = pd.DataFrame({"tick": [1, 2, 3], "anomaly": [False, False, False]})
    _write_feather(scen / "anomaly-observation.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="does not match frame range"):
        loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-shift")


def test_non_monotonic_ticks_raise(root):
    scen = _make_scenario(root, "test/normal/Town03/scenario-dup", 3, labels=None)
    df = pd.DataFrame({"tick": [0, 1, 1], "anomaly": [False, False, False]})
    _write_feather(scen / "anomaly-observation.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="not strictly increasing"):
        loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-dup")


def test_invalid_label_values_raise(root):
    scen = _make_scenario(root, "test/normal/Town03/scenario-vals", 3, labels=None)
    df = pd.DataFrame({"tick": [0, 1, 2], "anomaly": [0, 1, 7]})
    _write_feather(scen / "anomaly-observation.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="boolean or 0/1"):
        loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-vals")


def test_corrupt_label_columns_raise(root):
    scen = _make_scenario(root, "test/normal/Town03/scenario-cols", 3, labels=None)
    df = pd.DataFrame({"time": [0, 1, 2], "anomaly": [0, 0, 0]})
    _write_feather(scen / "anomaly-observation.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="missing required column"):
        loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-cols")


# ── 7. frame loading + bridge statistics ────────────────────────────────────

def test_load_frame_train_base_modalities_and_stats(root):
    loader = _loader(root)
    frame = loader.load_frame(CarlAnomalySplit.TRAIN, "Town01/scenario-1", tick=0)

    assert isinstance(frame, CarlAnomalyFrame)
    assert frame.label == 0
    assert frame.anomaly_type == "NORMAL"
    assert frame.timestep == 0

    # Uniform gray image: variance ~0, brightness ~100/255
    assert frame.rgb.shape == (1, 2)
    assert frame.rgb[0, 1] == pytest.approx(100 / 255, abs=0.02)
    assert frame.rgb[0, 0] < 0.01

    # GNSS: pos (0,0)->(1,0): jump at tick 1 = 1.0; tick 0 = 0 by construction
    assert frame.gnss.shape == (1, 1)
    assert frame.gnss[0, 0] == pytest.approx(0.0)
    f1 = loader.load_frame(CarlAnomalySplit.TRAIN, "Town01/scenario-1", tick=1)
    assert f1.gnss[0, 0] == pytest.approx(1.0)

    # IMU jerk: |a_norm diff|; a_norm = [0, 1, sqrt2, sqrt5] -> jerk[1] = 1.0
    assert f1.imu[0, 0] == pytest.approx(1.0)

    # Depth: R channel min = 25 -> 25/255
    assert f1.depth.shape == (1, 1)
    assert f1.depth[0, 0] == pytest.approx(25 / 255)

    # LiDAR density: 5000 points / 10000 = 0.5
    assert f1.lidar.shape == (1, 1)
    assert f1.lidar[0, 0] == pytest.approx(0.5)

    # Segmentation entropy: classes [0,0,0,7] -> probs .75/.25
    expected_entropy = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    assert f1.segmentation[0, 0] == pytest.approx(expected_entropy, rel=1e-6)


def test_load_frame_absent_modalities_are_none_not_zeros(root):
    loader = _loader(root)
    frame = loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1", tick=0)
    assert frame.rgb is not None
    assert frame.gnss is not None
    assert frame.imu is not None
    # Absent modalities: explicit None, never zeros
    assert frame.depth is None
    assert frame.lidar is None
    assert frame.segmentation is None


def test_load_frame_anomaly_scenario_label_and_rgb_response(root):
    loader = _loader(root)
    normal = loader.load_frame(CarlAnomalySplit.TEST_ANOMALY, "Town02/debris-spawn/scenario-1", tick=0)
    hazard = loader.load_frame(CarlAnomalySplit.TEST_ANOMALY, "Town02/debris-spawn/scenario-1", tick=2)
    assert normal.label == 0
    assert hazard.label == 1
    assert hazard.anomaly_type == "debris-spawn"
    # Anomalous frame (half black/white) has much higher variance than uniform frame
    assert hazard.rgb[0, 0] > 0.2
    assert normal.rgb[0, 0] < 0.01


def test_load_frame_tick_out_of_range_raises(root):
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="out of range"):
        loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1", tick=99)
    with pytest.raises(CarlAnomalyLoaderError, match="out of range"):
        loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1", tick=-1)


def test_load_frame_require_modalities_raises(root):
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="missing modalities"):
        loader.load_frame(
            CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1", tick=0,
            require_modalities=(Modality.DEPTH_FRONT,),
        )


def test_load_frame_without_label_table_not_emitted(root):
    _make_scenario(root, "test/normal/Town04/scenario-nolabel", 2, labels=None, with_labels=False)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="ground-truth labels"):
        loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town04/scenario-nolabel", tick=0)


def test_load_frame_label_gap_blocks_frame(root):
    # Label table with a gap (tick 2 missing, count 3 != 4 frames): the strict
    # synchronization guard must fire before any frame is emitted.
    scen = _make_scenario(root, "test/normal/Town03/scenario-gap", 4, labels=None)
    df = pd.DataFrame({"tick": [0, 1, 3], "anomaly": [False, False, False]})
    _write_feather(scen / "anomaly-observation.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="does not match synchronized frame count"):
        loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-gap", tick=2)


def test_load_frame_imu_jerk_second_step(root):
    loader = _loader(root)
    f = loader.load_frame(CarlAnomalySplit.TRAIN, "Town01/scenario-1", tick=2)
    # a_norm = [0, 1, sqrt(2), sqrt(5)]; jerk[2] = |sqrt(2) - 1|
    assert f.imu[0, 0] == pytest.approx(np.sqrt(2) - 1, rel=1e-6)


def test_load_frame_gnss_ignores_non_numeric_columns(root):
    scen = _make_scenario(root, "test/normal/Town03/scenario-meta", 3, labels=None)
    df = pd.DataFrame(
        {
            "tick": [0, 1, 2],
            "lat": [0.0, 3.0, 3.0],
            "lon": [0.0, 0.0, 4.0],
            "meta": ["a", "b", "c"],
        }
    )
    _write_feather(scen / "gnss.feather", df)
    loader = _loader(root)
    f = loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-meta", tick=1)
    # jump = |(3,0) - (0,0)| = 3.0; meta column ignored
    assert f.gnss[0, 0] == pytest.approx(3.0)


def test_gnss_without_numeric_columns_raises(root):
    scen = _make_scenario(root, "test/normal/Town03/scenario-nognum", 2, labels=None)
    df = pd.DataFrame({"tick": [0, 1], "meta": ["a", "b"]})
    _write_feather(scen / "gnss.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="no numeric measurement columns"):
        loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town03/scenario-nognum", tick=0)


# ── 8. split isolation ──────────────────────────────────────────────────────

def _relative_parts(loader, split, scen_id):
    """Path components of a scenario dir, relative to the dataset root."""
    path = loader.scenario_path(split, scen_id)
    return path.relative_to(loader.root).parts


def test_split_isolation_paths_do_not_cross(root):
    """The same scenario ID may exist in multiple splits (as in the real
    dataset, where towns recur); isolation means each split resolves IDs
    strictly within its own directory tree. Compare path components, not
    substrings (the tmp fixture dir itself contains 'test')."""
    loader = _loader(root)
    for scen_id in loader.list_scenarios(CarlAnomalySplit.TRAIN):
        parts = _relative_parts(loader, CarlAnomalySplit.TRAIN, scen_id)
        assert parts[0] == "train"
    for split in (CarlAnomalySplit.TEST_NORMAL, CarlAnomalySplit.TEST_ANOMALY):
        for scen_id in loader.list_scenarios(split):
            parts = _relative_parts(loader, split, scen_id)
            assert parts[0] == "test"
            assert "train" not in parts


def test_scenario_path_resolves_only_within_split(root):
    loader = _loader(root)
    # 'Town01/scenario-1' exists in both train and test_normal; ensure each split
    # resolves to its own directory
    p_train = loader.scenario_path(CarlAnomalySplit.TRAIN, "Town01/scenario-1")
    p_test = loader.scenario_path(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1")
    assert p_train != p_test
    assert "train" in str(p_train)
    assert "test" in str(p_test)


def test_scenario_without_rgb_frames_raises(root):
    scen = root / "test" / "normal" / "Town05"
    scen.mkdir(parents=True)
    (scen / "scenario-empty").mkdir()
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="no rgb-front frames"):
        loader.build_manifest(CarlAnomalySplit.TEST_NORMAL, "Town05/scenario-empty")


# ── 9. env-var root resolution ──────────────────────────────────────────────

def test_env_var_root_resolution(root, monkeypatch):
    monkeypatch.setenv("CARLANOMALY_ROOT", str(root))
    loader = CarlAnomalyLoader()  # root=None -> env var
    assert loader.list_scenarios(CarlAnomalySplit.TRAIN) == ["Town01/scenario-1"]


# ── 10. verified real-dataset schemas (empirical probe) ─────────────────────

def test_scenario_category_independent_of_all_false_labels(root):
    """Verified real pattern (change-weather/scenario-1): every per-timestep
    label is False even though the scenario directory sits in the anomaly split.
    label must stay the official per-timestep boolean; anomaly_type must keep
    the scenario-level category. Never collapse frames to label=1."""
    loader = _loader(root)
    sid = "Town01/change-weather/scenario-1"
    labels = loader.load_timestep_labels(CarlAnomalySplit.TEST_ANOMALY, sid)
    assert [lb.label for lb in labels] == [0, 0, 0]
    assert all(lb.anomaly_type == "change-weather" for lb in labels)
    for tick in range(3):
        frame = loader.load_frame(CarlAnomalySplit.TEST_ANOMALY, sid, tick=tick)
        assert frame.label == 0, "per-timestep label must not be forced to 1"
        assert frame.anomaly_type == "change-weather"


def test_gnss_imu_without_index_column_align_positionally(root):
    """Real gnss/imu feathers carry no tick/frame column; row i <-> timestep i."""
    loader = _loader(root)
    m = loader.build_manifest(CarlAnomalySplit.TEST_ANOMALY, "Town01/change-weather/scenario-1")
    assert Modality.GNSS in m.available_modalities
    assert Modality.IMU in m.available_modalities
    f = loader.load_frame(CarlAnomalySplit.TEST_ANOMALY, "Town01/change-weather/scenario-1", tick=1)
    # fixture rows: lat (0,1,1), lon (0,0,1) -> jump at row 1 = |(1,0)-(0,0)| = 1
    assert f.gnss[0, 0] == pytest.approx(1.0)
    # a_mag rows: (0, 1, sqrt(2)) -> jerk at row 1 = |1 - 0| = 1
    assert f.imu[0, 0] == pytest.approx(1.0)


def test_imu_jerk_ignores_compass_and_orientation_columns(root):
    """Regression guard: jerk must use ONLY acceleration_x/y/z. Constant
    acceleration with substantially varying compass/longitude_* values must
    yield exactly zero jerk (the old all-numeric-column norm inflated jerk)."""
    scen = _make_scenario(root, "test/normal/Town05/scenario-orient", 4, labels=None)
    imu = pd.DataFrame(
        {
            "acceleration_x": [5.0, 5.0, 5.0, 5.0],  # constant
            "acceleration_y": [0.0, 0.0, 0.0, 0.0],  # constant
            "acceleration_z": [9.8, 9.8, 9.8, 9.8],  # constant
            "compass": [0.0, 1.5, 3.1, 6.2],  # varies substantially
            "longitude_x": [0.0, 0.9, -0.5, 2.0],  # varies substantially
            "longitude_y": [0.0, -1.1, 0.7, 1.9],  # varies substantially
            "longitude_z": [0.0, 0.4, -0.8, 0.3],  # varies substantially
        }
    )
    _write_feather(scen / "imu.feather", imu)
    loader = _loader(root)
    for tick in range(4):
        f = loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town05/scenario-orient", tick=tick)
        assert f.imu[0, 0] == pytest.approx(0.0), (
            "orientation columns must not contribute to the acceleration magnitude"
        )


def test_imu_jerk_uses_acceleration_norm_for_real_schema(root):
    """With the verified real 7-column schema, jerk equals |Δ sqrt(ax²+ay²+az²)|,
    not |Δ norm including compass/longitude_*| (which would differ here)."""
    scen = _make_scenario(root, "test/normal/Town05/scenario-acc", 3, labels=None)
    imu = pd.DataFrame(
        {
            "acceleration_x": [3.0, 0.0, 0.0],
            "acceleration_y": [0.0, 4.0, 0.0],
            "acceleration_z": [0.0, 0.0, 0.0],
            "compass": [0.0, 0.0, 0.0],
            "longitude_x": [9.0, 9.0, 9.0],
            "longitude_y": [9.0, 9.0, 9.0],
            "longitude_z": [9.0, 9.0, 9.0],
        }
    )
    _write_feather(scen / "imu.feather", imu)
    loader = _loader(root)
    f = loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town05/scenario-acc", tick=1)
    # a_mag = [3, 4, 0] -> jerk[1] = |4 - 3| = 1 (a all-numeric-column norm —
    # including compass and longitude_* — would give |sqrt(259)-sqrt(252)| ≈ 0.22)
    assert f.imu[0, 0] == pytest.approx(1.0)


def test_imu_missing_acceleration_columns_fail_loud(root):
    """A real-style IMU table without the full acceleration triple must raise,
    not silently compute a magnitude from unrelated numeric columns."""
    scen = _make_scenario(root, "test/normal/Town05/scenario-badimu", 2, labels=None)
    imu = pd.DataFrame(
        {
            "acceleration_x": [0.0, 1.0],
            "compass": [0.0, 1.0],
            "longitude_x": [0.0, 1.0],
        }
    )
    _write_feather(scen / "imu.feather", imu)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="acceleration"):
        loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town05/scenario-badimu", tick=0)


def test_gnss_with_explicit_frame_column_uses_lookup(root):
    """If a 'frame' index column is present it is used for lookup, not position."""
    scen = _make_scenario(root, "test/normal/Town05/scenario-idx", 3, labels=None)
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "latitude": [0.0, 3.0, 3.0],
            "longitude": [0.0, 0.0, 4.0],
        }
    )
    _write_feather(scen / "gnss.feather", df)
    loader = _loader(root)
    f = loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town05/scenario-idx", tick=1)
    assert f.gnss[0, 0] == pytest.approx(3.0)


def test_gnss_index_gap_fails_loud_no_interpolation(root):
    scen = _make_scenario(root, "test/normal/Town05/scenario-gapidx", 3, labels=None)
    df = pd.DataFrame(
        {
            "tick": [0, 2, 4],
            "latitude": [0.0, 1.0, 2.0],
            "longitude": [0.0, 0.0, 0.0],
        }
    )
    _write_feather(scen / "gnss.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="interpolation is not permitted"):
        loader.load_frame(CarlAnomalySplit.TEST_NORMAL, "Town05/scenario-gapidx", tick=1)


def test_grayscale_depth_png_verified_format(root):
    """Official depth PNGs are 2-D grayscale (mode 'L'); min/255 is a normalized
    intensity, verified against the fixture values."""
    loader = _loader(root)
    f = loader.load_frame(CarlAnomalySplit.TRAIN, "Town01/scenario-1", tick=0)
    assert f.depth.shape == (1, 1)
    assert f.depth[0, 0] == pytest.approx(25 / 255)
    with PIL.Image.open(loader.scenario_path(CarlAnomalySplit.TRAIN, "Town01/scenario-1") / "depth-front" / "000000.png") as im:
        assert im.mode == "L"


def test_three_channel_depth_defensively_supported(root):
    """A 3-channel depth array falls back to channel 0 and yields the same statistic."""
    _make_scenario(
        root, "train/Town02/scenario-rgb-depth", 2, labels=None,
        with_depth=True, depth_three_channel=True,
    )
    loader = _loader(root)
    f = loader.load_frame(CarlAnomalySplit.TRAIN, "Town02/scenario-rgb-depth", tick=0)
    assert f.depth[0, 0] == pytest.approx(25 / 255)


def test_anomaly_observation_optional_columns_accepted(root):
    """Real schema carries 'description'; undocumented optional columns
    (anomaly_obj_ids etc.) must never be required."""
    loader = _loader(root)
    labels = loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town01/scenario-1")
    assert [lb.label for lb in labels] == [0, 0, 0]
    # The fixture's obs table has (tick, anomaly, description) only.


# Existing keep-guard for the strict sync checks (unchanged behavior, asserted
# again here so future relaxations are a conscious decision):
def test_strict_sync_checks_still_enforced(root):
    scen = _make_scenario(root, "test/normal/Town05/scenario-short", 3, labels=None)
    df = pd.DataFrame({"tick": [0, 1], "anomaly": [False, False], "description": [None, None]})
    _write_feather(scen / "anomaly-observation.feather", df)
    loader = _loader(root)
    with pytest.raises(CarlAnomalyLoaderError, match="does not match synchronized frame count"):
        loader.load_timestep_labels(CarlAnomalySplit.TEST_NORMAL, "Town05/scenario-short")

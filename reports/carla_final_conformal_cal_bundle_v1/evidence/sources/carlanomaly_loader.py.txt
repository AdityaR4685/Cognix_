"""
CarlAnomaly Offline Loader — read-only access to the real CarlAnomaly dataset.

References
----------
Official dataset documentation:
    https://carlanomaly.de/dataset/   (structure, sensors, labels)
    https://carlanomaly.de/download/  (parts and archive sizes)

Official directory layout (verified against the docs above):

    carlanomaly/
    ├── train/
    │   └── Town01/ ... Town05, Town10HD/
    │       └── scenario-N/
    └── test/
        ├── normal/
        │   └── Town01/ ... /scenario-N/
        └── anomaly/
            └── Town01/ ... /<anomaly-type>/scenario-N/

Per-scenario contents (only what this loader consumes):

    scenario/
    ├── rgb-front/000000.jpg            (Base part)
    ├── segmentation-front/000000.png   (Base part — empirically verified inside
    │                                   carlanomaly-base-test.tar.gz)
    ├── anomaly-front/000000.png        (Base part; pixel masks, not consumed here)
    ├── gnss.feather                    (Base part)
    ├── imu.feather                     (Base part)
    ├── depth-front/000000.png          (Depth Maps part; NOT in Base)
    ├── pointclouds/000000.feather      (LiDAR part; NOT in Base)
    └── anomaly-observation.feather     (Base part; timestep-level anomaly labels)

    Empirically verified feather schemas (carlanomaly-base-test.tar.gz probe):
        gnss.feather:                altitude, latitude, longitude  (float64; NO index column)
        imu.feather:                 acceleration_x/y/z, compass, longitude_x/y/z
                                     (float64; NO index column)
        anomaly-observation.feather: anomaly (bool), tick (int64), description (object);
                                     optional extra columns (anomaly_obj_ids,
                                     anomaly_class_ids, meta) may appear and are ignored

    Sensor alignment: gnss/imu rows are aligned POSITIONALLY (row i <-> timestep i);
    no interpolation and no sampling-rate assumption is made. If an integer
    'tick'/'frame' column is present it is used as the explicit index instead.

    Label semantics: anomaly-observation.feather['anomaly'] is the per-timestep
    ground truth and may be False for every tick of a scenario in the anomaly
    split (verified for change-weather/scenario-1). The scenario-level category
    comes from the directory structure (anomaly_type) and is kept strictly
    independent of the per-timestep label.

Scientific constraints honored by this module
---------------------------------------------
1.  READ-ONLY. No downloads, no training, no artifact writes, no modification of
    cognix/adapters/carla/dataset.py, features.py, agents.py, pipeline, thresholds,
    or artifacts/synthetic/.
2.  The compact statistics computed here (camera variance/brightness, GNSS drift,
    IMU jerk, ...) are a COMPATIBILITY BRIDGE to the frozen synthetic contract.
    They are NOT genuine learned perception and NOT genuine uncertainty estimation.
3.  The official training split contains ONLY normal scenarios. The binary
    P(ACT/safe) GAT CANNOT simply be supervised-trained on the real training split.
    Training methodology is a separate future decision; this module only reads data.
4.  Fail loudly on absent/corrupt/misaligned data. Never silently zero-fill
    missing modalities: absent modalities are declared in the scenario manifest
    and their frame fields are None (explicit absence), never zeros.
5.  Strict split isolation: train / test_normal / test_anomaly are separate
    iteration spaces; no code path mixes them.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from cognix.adapters.carla.dataset import CarlAnomalyFrame

logger = logging.getLogger(__name__)

# Defaults chosen from the official docs (6 CARLA towns).
# NOTE: no sampling frequency is documented officially; row-count equality does
# not establish one, and no sampling-rate assumption is made anywhere here.
EXPECTED_TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD")

# Ticks are 6-digit zero-padded (e.g. 000000.jpg, 000000.feather).
TICK_GLOB = "[0-9]" * 6


class CarlAnomalySplit(str, Enum):
    """Strict split identifier. Values map to real directory trees."""

    TRAIN = "train"
    TEST_NORMAL = "test_normal"
    TEST_ANOMALY = "test_anomaly"

    @property
    def dir_components(self) -> tuple[str, ...]:
        """Directory components under the dataset root for this split."""
        if self is CarlAnomalySplit.TRAIN:
            return ("train",)
        if self is CarlAnomalySplit.TEST_NORMAL:
            return ("test", "normal")
        return ("test", "anomaly")


class Modality(str, Enum):
    """Modalities consumed by the six COGNIX agents (see features.py)."""

    RGB_FRONT = "rgb_front"
    DEPTH_FRONT = "depth_front"
    LIDAR = "lidar"
    SEGMENTATION_FRONT = "segmentation_front"
    GNSS = "gnss"
    IMU = "imu"
    TIMESTEP_LABELS = "timestep_labels"


# Modality -> directory name inside a scenario.
_MODALITY_DIRS = {
    Modality.RGB_FRONT: "rgb-front",
    Modality.DEPTH_FRONT: "depth-front",
    Modality.LIDAR: "pointclouds",
    Modality.SEGMENTATION_FRONT: "segmentation-front",
}

# Modality -> feather file name inside a scenario.
_MODALITY_FEATHER = {
    Modality.GNSS: "gnss.feather",
    Modality.IMU: "imu.feather",
    Modality.TIMESTEP_LABELS: "anomaly-observation.feather",
}


class CarlAnomalyLoaderError(RuntimeError):
    """Raised on absent, corrupt, or misaligned CarlAnomaly data. Fail loud."""


@dataclass(frozen=True)
class ScenarioManifest:
    """
    Explicit per-scenario modality manifest.

    Attributes
    ----------
    available_modalities : frozenset[Modality]
        Modalities physically present for this scenario. Absent modalities are
        declared here and must NOT be silently zero-filled by callers.
    n_rgb_ticks / n_depth_ticks / n_lidar_ticks / n_seg_ticks : int | None
        Per-modality frame counts. None when the modality directory is absent.
    n_feather_rows : dict[str, int]
        Row counts of present feather tables (gnss/imu/timestep labels).
    n_ticks : int
        Synchronized scenario length. Requires rgb_front present.
    """
    scenario_id: str
    split: CarlAnomalySplit
    town: str
    anomaly_type: Optional[str]  # None conceptually for train/test_normal; "NORMAL" by contract
    path: Path
    available_modalities: frozenset
    n_rgb_ticks: Optional[int]
    n_depth_ticks: Optional[int]
    n_lidar_ticks: Optional[int]
    n_seg_ticks: Optional[int]
    n_feather_rows: dict
    n_ticks: int

    def require(self, *modalities: Modality) -> None:
        """Raise if any requested modality is not available."""
        missing = [m.value for m in modalities if m not in self.available_modalities]
        if missing:
            raise CarlAnomalyLoaderError(
                f"Scenario '{self.scenario_id}' is missing modalities {missing}. "
                "Download the corresponding CarlAnomaly part or exclude this scenario. "
                "Zero-filling missing modalities is not permitted."
            )


@dataclass(frozen=True)
class TimestepLabel:
    """One synchronized timestep with its ground-truth anomaly label."""

    scenario_id: str
    tick: int
    label: int  # 0 = normal/safe, 1 = anomaly/hazard
    anomaly_type: str  # "NORMAL" for train / test_normal


def _count_frames(directory: Path) -> Optional[int]:
    """Count 6-digit frame files in a modality directory (None if absent)."""
    if not directory.is_dir():
        return None
    return sum(1 for p in directory.glob(f"{TICK_GLOB}.*") if p.is_file())


def _read_feather(path: Path):
    """
    Read a feather file via pandas, with a fail-loud, actionable error if the
    arrow backend is unavailable (pyarrow is required by the real dataset).
    """
    try:
        import pandas as pd

        return pd.read_feather(path)
    except ImportError as exc:
        raise CarlAnomalyLoaderError(
            f"Cannot read {path.name}: pandas could not import a feather backend. "
            "Install pyarrow into the project Python environment "
            "(e.g. `py -3.12 -m pip install pyarrow`)."
        ) from exc


def _feather_row_count(path: Path) -> Optional[int]:
    """Row count of a feather table (None if the file is absent)."""
    if not path.is_file():
        return None
    return int(len(_read_feather(path)))


def _require_feather_columns(df, required: set, path: Path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise CarlAnomalyLoaderError(
            f"{path.name}: missing required column(s) {sorted(missing)}; "
            f"found columns {list(df.columns)}."
        )


def _frame_statistic(path: Path, modality: Modality) -> np.ndarray:
    """
    Compute the compact compatibility-bridge statistic for one frame of one
    modality, matching the input contract of cognix/adapters/carla/features.py:

        rgb_front          -> [variance, brightness]   (rgb_to_hazard_prob)
        depth_front        -> [min_depth]              (depth_to_obstacle_score)
        lidar              -> [density]                (lidar_to_density_score)
        segmentation_front -> [entropy]                (seg_to_complexity_score)

    NOTE (scientific): these are simple image/point-cloud statistics, NOT learned
    perception. They only preserve array shapes so the frozen agents remain usable.
    """
    import PIL.Image

    if modality is Modality.RGB_FRONT:
        arr = np.asarray(PIL.Image.open(path), dtype=np.float64) / 255.0
        return np.array([arr.var(), arr.mean()], dtype=np.float64)
    if modality is Modality.DEPTH_FRONT:
        arr = np.asarray(PIL.Image.open(path), dtype=np.float64)
        # Official CarlAnomaly depth PNGs are 2-D grayscale uint8 (PIL mode 'L'),
        # verified against the official site example image. The dataset's metric
        # depth convention is NOT documented, so this statistic is explicitly a
        # NORMALIZED DEPTH INTENSITY in [0, 1] (value/255), NOT metric meters.
        if arr.ndim == 3:
            arr = arr[..., 0]  # defensive: channel 0 if a 3-channel array appears
        min_depth = float(arr.min()) / 255.0
        return np.array([min_depth], dtype=np.float64)
    if modality is Modality.LIDAR:
        df = _read_feather(path)
        _require_feather_columns(df, {"x", "y", "z"}, path)
        # Density proxy: normalized point count (bridge statistic only).
        density = float(len(df)) / 10000.0
        return np.array([min(density, 0.99)], dtype=np.float64)
    if modality is Modality.SEGMENTATION_FRONT:
        arr = np.asarray(PIL.Image.open(path), dtype=np.float64)
        # Per-pixel classes are encoded in the R channel (official docs).
        counts = np.bincount(arr[..., 0].astype(np.int64).ravel())
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        return np.array([entropy], dtype=np.float64)
    raise CarlAnomalyLoaderError(f"No statistic defined for modality {modality}")


class CarlAnomalyLoader:
    """
    Read-only, validation-first loader for the real CarlAnomaly dataset.

    Parameters
    ----------
    root : Path | str | None
        Dataset root (the directory that directly contains ``train/`` and
        ``test/``). If None, the ``CARLANOMALY_ROOT`` environment variable is
        used. Absent root raises CarlAnomalyLoaderError — no silent fallback.
    strict_layout : bool
        If True (default), unknown town directories under a split raise an
        error. Pass False to accept custom (e.g. fixture) layouts.
    """

    def __init__(self, root: Path | str | None = None, strict_layout: bool = True):
        if root is None:
            root = os.environ.get("CARLANOMALY_ROOT")
        if not root:
            raise CarlAnomalyLoaderError(
                "CarlAnomaly root not provided and CARLANOMALY_ROOT is not set. "
                "Pass root=<path to the directory containing train/ and test/> "
                "after downloading an official dataset part."
            )
        self.root = Path(root)
        if not self.root.is_dir():
            raise CarlAnomalyLoaderError(f"CarlAnomaly root does not exist: {self.root}")
        self.strict_layout = strict_layout

    # ------------------------------------------------------------------
    # Split / scenario discovery
    # ------------------------------------------------------------------

    def split_dir(self, split: CarlAnomalySplit) -> Path:
        d = self.root.joinpath(*split.dir_components)
        if not d.is_dir():
            raise CarlAnomalyLoaderError(
                f"Split '{split.value}' directory not found: {d}. "
                "The corresponding archive part/split has not been extracted."
            )
        return d

    def list_towns(self, split: CarlAnomalySplit) -> list:
        towns = sorted(d.name for d in self.split_dir(split).iterdir() if d.is_dir())
        if not towns:
            raise CarlAnomalyLoaderError(f"No towns found under {self.split_dir(split)}")
        if self.strict_layout:
            unknown = [t for t in towns if t not in EXPECTED_TOWNS]
            if unknown:
                raise CarlAnomalyLoaderError(
                    f"Unexpected town directories {unknown} under {self.split_dir(split)}; "
                    f"expected a subset of {list(EXPECTED_TOWNS)}. "
                    "Pass strict_layout=False to accept custom layouts."
                )
        return towns

    def list_scenarios(self, split: CarlAnomalySplit, town: Optional[str] = None) -> list:
        """
        List scenario IDs in the official nested layout.

        IDs are '<town>/<scenario-dir>' for train and test_normal, and
        '<town>/<anomaly-type>/<scenario-dir>' for test_anomaly. The anomaly
        type for a scenario is derived from the directory structure.
        """
        base = self.split_dir(split)
        towns = [town] if town is not None else self.list_towns(split)
        scenarios: list = []
        for t in towns:
            town_dir = base / t
            if not town_dir.is_dir():
                raise CarlAnomalyLoaderError(f"Town directory not found: {town_dir}")
            if split is CarlAnomalySplit.TEST_ANOMALY:
                for type_dir in sorted(p for p in town_dir.iterdir() if p.is_dir()):
                    for scen_dir in sorted(p for p in type_dir.iterdir() if p.is_dir()):
                        scenarios.append(f"{t}/{type_dir.name}/{scen_dir.name}")
            else:
                for scen_dir in sorted(p for p in town_dir.iterdir() if p.is_dir()):
                    scenarios.append(f"{t}/{scen_dir.name}")
        if not scenarios:
            raise CarlAnomalyLoaderError(f"No scenarios found under {base} (towns={towns}).")
        return scenarios

    def scenario_path(self, split: CarlAnomalySplit, scenario_id: str) -> Path:
        """Resolve a scenario ID to its directory, verifying all components exist."""
        parts = scenario_id.split("/")
        expected_parts = 3 if split is CarlAnomalySplit.TEST_ANOMALY else 2
        if len(parts) != expected_parts:
            raise CarlAnomalyLoaderError(
                f"Malformed scenario ID '{scenario_id}' for split '{split.value}': "
                f"expected {expected_parts} path components."
            )
        path = self.split_dir(split).joinpath(*parts)
        if not path.is_dir():
            raise CarlAnomalyLoaderError(f"Scenario directory not found: {path}")
        return path

    @staticmethod
    def anomaly_type_of(split: CarlAnomalySplit, scenario_id: str) -> str:
        """Anomaly type from directory structure; 'NORMAL' for non-anomaly splits."""
        if split is not CarlAnomalySplit.TEST_ANOMALY:
            return "NORMAL"
        return scenario_id.split("/")[1]

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def build_manifest(self, split: CarlAnomalySplit, scenario_id: str) -> ScenarioManifest:
        """
        Build the explicit modality manifest for one scenario.

        Raises CarlAnomalyLoaderError if the scenario directory is unreadable,
        has no rgb-front frames, or a present feather table is corrupt.
        """
        path = self.scenario_path(split, scenario_id)

        n_rgb = _count_frames(path / "rgb-front")
        if not n_rgb:
            raise CarlAnomalyLoaderError(
                f"Scenario '{scenario_id}' has no rgb-front frames; "
                "the Base part is required for synchronization."
            )

        n_depth = _count_frames(path / "depth-front")
        n_lidar = _count_frames(path / "pointclouds")
        n_seg = _count_frames(path / "segmentation-front")

        n_feather: dict = {}
        for mod, fname in _MODALITY_FEATHER.items():
            n = _feather_row_count(path / fname)
            if n is not None:
                n_feather[mod.value] = n

        available = {Modality.RGB_FRONT}
        if n_depth:
            available.add(Modality.DEPTH_FRONT)
        if n_lidar:
            available.add(Modality.LIDAR)
        if n_seg:
            available.add(Modality.SEGMENTATION_FRONT)
        if Modality.GNSS.value in n_feather:
            available.add(Modality.GNSS)
        if Modality.IMU.value in n_feather:
            available.add(Modality.IMU)
        if Modality.TIMESTEP_LABELS.value in n_feather:
            available.add(Modality.TIMESTEP_LABELS)

        return ScenarioManifest(
            scenario_id=scenario_id,
            split=split,
            town=scenario_id.split("/")[0],
            anomaly_type=self.anomaly_type_of(split, scenario_id),
            path=path,
            available_modalities=frozenset(available),
            n_rgb_ticks=n_rgb,
            n_depth_ticks=n_depth,
            n_lidar_ticks=n_lidar,
            n_seg_ticks=n_seg,
            n_feather_rows=n_feather,
            n_ticks=n_rgb,
        )

    # ------------------------------------------------------------------
    # Timestep labels (ground truth)
    # ------------------------------------------------------------------

    def load_timestep_labels(self, split: CarlAnomalySplit, scenario_id: str) -> list:
        """
        Read timestep-level anomaly ground truth from anomaly-observation.feather.

        Required columns: only 'tick' and 'anomaly'. Optional columns observed in
        real data ('description') or documented as optional ('anomaly_obj_ids',
        'anomaly_class_ids', 'meta') are allowed and ignored.

        Labels are the official per-timestep booleans (which may be False for
        every tick of a scenario in the anomaly split); the scenario-level
        anomaly category is carried separately via TimestepLabel.anomaly_type.

        Fail-loud checks: file present, required columns (tick, anomaly), tick
        strictly increasing and unique, labels boolean or 0/1.
        """
        manifest = self.build_manifest(split, scenario_id)
        manifest.require(Modality.TIMESTEP_LABELS)

        path = manifest.path / "anomaly-observation.feather"
        df = _read_feather(path)
        _require_feather_columns(df, {"tick", "anomaly"}, path)

        ticks = df["tick"].to_numpy(dtype=np.int64)
        if len(ticks) == 0:
            raise CarlAnomalyLoaderError(f"{path}: timestep label table is empty.")
        diffs = np.diff(ticks)
        if np.any(diffs <= 0):
            raise CarlAnomalyLoaderError(
                f"{path}: tick column is not strictly increasing "
                f"(first violation at index {int(np.argmax(diffs <= 0)) + 1})."
            )

        raw = df["anomaly"].to_numpy()
        if raw.dtype == bool:
            labels = raw.astype(np.int64)
        else:
            labels = raw.astype(np.int64)
            if not np.all(np.isin(labels, (0, 1))):
                raise CarlAnomalyLoaderError(
                    f"{path}: anomaly column must be boolean or 0/1, got values "
                    f"{np.unique(labels)[:5]}."
                )

        out = [
            TimestepLabel(
                scenario_id=scenario_id,
                tick=int(ticks[i]),
                label=int(labels[i]),
                anomaly_type=manifest.anomaly_type,
            )
            for i in range(len(ticks))
        ]

        # Validation-first: the label table must be synchronized with the
        # scenario frame count even when consumed directly (not via load_frame).
        self._check_synchronization(manifest, len(out), out[0].tick, out[-1].tick)
        return out

    def _check_synchronization(
        self,
        manifest: ScenarioManifest,
        n_labels: int,
        label_t0: int,
        label_t1: int,
    ) -> None:
        """Strict scenario/timestep synchronization checks (fail loud)."""
        n = manifest.n_ticks
        if n_labels != n:
            raise CarlAnomalyLoaderError(
                f"Scenario '{manifest.scenario_id}': timestep label count ({n_labels}) "
                f"does not match synchronized frame count ({n})."
            )
        if label_t0 != 0 or label_t1 != n - 1:
            raise CarlAnomalyLoaderError(
                f"Scenario '{manifest.scenario_id}': label tick range "
                f"[{label_t0}, {label_t1}] does not match frame range [0, {n - 1}]."
            )

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def load_frame(
        self,
        split: CarlAnomalySplit,
        scenario_id: str,
        tick: int,
        manifest: Optional[ScenarioManifest] = None,
        require_modalities: Optional[tuple] = None,
        labels: Optional[list] = None,
    ) -> CarlAnomalyFrame:
        """
        Load one synchronized observation as a CarlAnomalyFrame.

        Per-modality values are compact bridge statistics (see _frame_statistic
        for camera/depth/lidar/seg; GNSS -> per-tick position-jump magnitude;
        IMU -> per-tick jerk of the acceleration norm). Modalities present are
        computed from real data; ABSENT modalities yield None fields (explicit
        absence) and raise CarlAnomalyLoaderError when explicitly requested via
        require_modalities. Missing modalities are never zero-filled.

        Ground truth comes from anomaly-observation.feather (tick-aligned). The
        full label table is validated for synchronization with the frame count
        on every call (cheap for one scenario; pass `labels` to reuse).
        """
        manifest = manifest or self.build_manifest(split, scenario_id)

        if not (0 <= tick < manifest.n_ticks):
            raise CarlAnomalyLoaderError(
                f"Scenario '{scenario_id}': tick {tick} out of range [0, {manifest.n_ticks - 1}]."
            )

        if require_modalities:
            manifest.require(*require_modalities)

        if Modality.TIMESTEP_LABELS not in manifest.available_modalities:
            raise CarlAnomalyLoaderError(
                f"Scenario '{scenario_id}': anomaly-observation.feather is required to "
                "attach ground-truth labels and is missing. A frame without its "
                "timestep label cannot be validated and is not emitted."
            )

        tick_name = f"{tick:06d}"
        path = manifest.path
        avail = manifest.available_modalities

        # ── Per-modality statistics (present modalities only) ─────────
        rgb_stat = (
            _frame_statistic(path / "rgb-front" / f"{tick_name}.jpg", Modality.RGB_FRONT)
            if Modality.RGB_FRONT in avail else None
        )
        depth_stat = (
            _frame_statistic(path / "depth-front" / f"{tick_name}.png", Modality.DEPTH_FRONT)
            if Modality.DEPTH_FRONT in avail else None
        )
        lidar_stat = (
            _frame_statistic(path / "pointclouds" / f"{tick_name}.feather", Modality.LIDAR)
            if Modality.LIDAR in avail else None
        )
        seg_stat = (
            _frame_statistic(path / "segmentation-front" / f"{tick_name}.png", Modality.SEGMENTATION_FRONT)
            if Modality.SEGMENTATION_FRONT in avail else None
        )

        gnss_stat = None
        imu_stat = None
        if Modality.GNSS in avail:
            df = _read_feather(path / "gnss.feather")
            gnss_stat = np.array([self._gnss_jump_at(df, tick, path)], dtype=np.float64)
        if Modality.IMU in avail:
            df = _read_feather(path / "imu.feather")
            imu_stat = np.array([self._imu_jerk_at(df, tick, path)], dtype=np.float64)

        # ── Ground truth + synchronization ────────────────────────────
        if labels is None:
            labels = self.load_timestep_labels(split, scenario_id)
        label_map = {lb.tick: lb for lb in labels}
        if tick not in label_map:
            raise CarlAnomalyLoaderError(
                f"Scenario '{scenario_id}': tick {tick} has no timestep label."
            )
        self._check_synchronization(manifest, len(labels), labels[0].tick, labels[-1].tick)
        lb = label_map[tick]

        return CarlAnomalyFrame(
            rgb=rgb_stat.reshape(1, -1) if rgb_stat is not None else None,
            depth=depth_stat.reshape(1, -1) if depth_stat is not None else None,
            lidar=lidar_stat.reshape(1, -1) if lidar_stat is not None else None,
            gnss=gnss_stat.reshape(1, -1) if gnss_stat is not None else None,
            imu=imu_stat.reshape(1, -1) if imu_stat is not None else None,
            segmentation=seg_stat.reshape(1, -1) if seg_stat is not None else None,
            label=lb.label,
            anomaly_type=lb.anomaly_type,
            timestep=tick,
        )

    # ------------------------------------------------------------------
    # Bridge statistics for feather-based ego-motion sensors
    # ------------------------------------------------------------------

    # Integer index columns recognized in sensor feather tables. Real gnss/imu
    # files carry NEITHER (verified); positional alignment is the verified path.
    _INDEX_COLUMNS = ("tick", "frame")

    # Verified real imu.feather acceleration columns. Jerk uses ONLY these;
    # compass and longitude_* (orientation / angular-rate) columns never enter
    # the acceleration magnitude.
    _REAL_ACCEL_COLUMNS = ("acceleration_x", "acceleration_y", "acceleration_z")

    @classmethod
    def _sensor_index(cls, df, path: Path):
        """
        Return (name, values) of the integer index column ('tick' or 'frame')
        if present, else (None, None). Validated strictly increasing when present.
        """
        for col in cls._INDEX_COLUMNS:
            if col in df.columns:
                idx = df[col].to_numpy(dtype=np.int64)
                if len(idx) > 1 and np.any(np.diff(idx) <= 0):
                    raise CarlAnomalyLoaderError(
                        f"{path.name}: index column '{col}' is not strictly increasing."
                    )
                return col, idx
        return None, None

    @classmethod
    def _sensor_values(cls, df, path: Path, sensor_name: str):
        """
        Extract the per-timestep numeric measurement matrix from a feather table.

        Alignment policy (no interpolation, no sampling-rate assumption):
          * If an integer 'tick'/'frame' column exists, it is the explicit index
            (validated strictly increasing) and excluded from the values.
          * Otherwise rows align POSITIONALLY: row i <-> timestep i (verified
            against the official Base-test archive: real gnss.feather has only
            altitude/latitude/longitude and real imu.feather only
            acceleration_*/compass/longitude_* — no index column at all).
        Non-numeric extra columns (e.g. 'description', 'meta') are ignored
        explicitly, not silently mixed into the measurement.
        """
        index_col, index = cls._sensor_index(df, path)
        value_cols = [
            c for c in df.columns
            if c not in cls._INDEX_COLUMNS and np.issubdtype(df[c].dtype, np.number)
        ]
        if not value_cols:
            raise CarlAnomalyLoaderError(
                f"{path.name}: no numeric measurement columns found for {sensor_name}."
            )
        return index, df[value_cols].to_numpy(dtype=np.float64)

    @staticmethod
    def _row_for_timestep(index, n_rows: int, tick: int, path: Path) -> int:
        """Map a timestep to its row position; fail loud, never interpolate."""
        if index is None:
            row = tick  # positional alignment: row i <-> timestep i
        else:
            row = int(np.searchsorted(index, tick))
            if row >= n_rows or int(index[row]) != tick:
                raise CarlAnomalyLoaderError(
                    f"{path.name}: timestep {tick} not present in the sensor "
                    "index column; interpolation is not permitted."
                )
        if not (0 <= row < n_rows):
            raise CarlAnomalyLoaderError(
                f"{path.name}: timestep {tick} is beyond the {n_rows} available "
                "sensor rows."
            )
        return row

    @classmethod
    def _gnss_jump_at(cls, df, tick: int, path: Path) -> float:
        """
        Compatibility-bridge GNSS drift statistic: per-timestep position-jump
        magnitude (Euclidean norm of the step-over-step displacement across all
        numeric columns — altitude/latitude/longitude in real data). This is
        NOT a Kalman-filtered drift estimate; it is a simple bridge statistic.
        """
        index, pos = cls._sensor_values(df, path, "GNSS")
        row = cls._row_for_timestep(index, len(pos), tick, path)
        jumps = (
            np.linalg.norm(np.diff(pos, axis=0), axis=1)
            if len(pos) > 1
            else np.zeros(1)
        )
        jumps = np.concatenate([[0.0], jumps])  # element i corresponds to row i
        return float(jumps[row])

    @classmethod
    def _imu_jerk_at(cls, df, tick: int, path: Path) -> float:
        """
        Compatibility-bridge IMU statistic: per-timestep jerk (absolute discrete
        derivative of the acceleration-norm series). NOT a full jerk analysis.

        The acceleration magnitude uses ONLY the verified acceleration columns
        acceleration_x/y/z (real imu.feather schema):
            a_mag = sqrt(ax^2 + ay^2 + az^2)
        compass and longitude_* (orientation / angular-rate) columns NEVER enter
        the magnitude. Fails loud if the acceleration triple is missing or
        incomplete rather than silently mixing unrelated numeric columns.
        """
        missing = [c for c in cls._REAL_ACCEL_COLUMNS if c not in df.columns]
        if missing:
            raise CarlAnomalyLoaderError(
                f"{path.name}: IMU jerk requires acceleration columns "
                f"{list(cls._REAL_ACCEL_COLUMNS)}; missing {missing}."
            )
        index, _ = cls._sensor_values(df, path, "IMU")
        accel = df[list(cls._REAL_ACCEL_COLUMNS)].to_numpy(dtype=np.float64)
        row = cls._row_for_timestep(index, len(accel), tick, path)
        a_mag = np.linalg.norm(accel, axis=1)
        jerk = np.abs(np.diff(a_mag)) if len(a_mag) > 1 else np.zeros(1)
        jerk = np.concatenate([[0.0], jerk])  # element i corresponds to row i
        return float(jerk[row])

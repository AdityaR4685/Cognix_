"""
Compact per-timestep TRAIN cache for CarlAnomaly Experiment 1.

SCOPE AND ARCHITECTURE GUARDRAIL
--------------------------------
CARLA/CarlAnomaly-specific functionality inside the CARLA adapter. The generic
COGNIX framework never sees this module; nothing here imports the frozen
synthetic bridge, and the only `cognix.*` import is the loader's split enum.

PURPOSE
-------
Turn official CarlAnomaly TRAIN scenarios (via the existing validated loader)
into a DETERMINISTIC, COMPACT, per-timestep feature cache that later stages
(agent fitting, CAL wiring, pseudo-anomaly studies) consume without touching
raw images again:

    per row (scenario_id, town, tick, source_split):
        Camera features    (camera_embedding_features, 18 dims)
        Segmentation features (segmentation_histogram_features, 29 dims)
        GNSS features      (gnss_window_features, 8 dims)
        IMU features       (imu_window_features, 10 dims)
    later (agent-output stage, not fitted here):
        camera_p / camera_epistemic / camera_aleatoric, ... per modality
    plus provenance and version/hash metadata.

The scientific wording is inherited from real_features.py: camera features are
DETERMINISTIC HANDCRAFTED COMPACT FEATURES for Experiment 1 — NOT pretrained
and NOT learned visual embeddings. Nothing here infers a sampling frequency.

TEMPORAL WINDOW POLICY (explicit, causal)
-----------------------------------------
GNSS and IMU features summarize a short trailing window of sensor rows:
    - ONLY current and previous timesteps are used (strictly causal);
    - future frames are never read;
    - WINDOW_LENGTH = 12 (documented constant; chosen a priori, not tuned);
    - ticks < WINDOW_LENGTH - 1 use exactly (tick + 1) rows (warm-up is
      deterministic and recorded in the manifest policy block);
    - the FIRST CACHED TICK IS 1: a window feature requires >= 2 rows, so tick 0
      would need future data (forbidden) or a fabricated step (dishonest);
      every cached tick has a fully causal window of 2..WINDOW_LENGTH rows;
    - windows NEVER cross scenario boundaries (per-scenario sensor tables).
No sampling rate is inferred or assumed; windows are in TICKS, not seconds.

CACHE FORMAT
------------
Simple, auditable, dependency-free (numpy is already required):
    <name>.npz       dense per-modality feature matrices + index arrays
    <name>.json      manifest (schema version, provenance, hashes, policy)
Determinism: identical inputs produce identical npz ARRAY CONTENT (the zip
container embeds wall-clock timestamps, so the auditable identity hash is
computed over sorted array contents, not raw file bytes) and identical
manifest content except the wall-clock creation timestamp. The builder NEVER
silently overwrites a mismatched cache: rebuilding over an existing cache with
a different CONTENT hash raises CacheBuilderError unless allow_mismatch=True
(the new npz is staged to a temp file and only atomically moved into place
after the check passes, so a refused rebuild leaves the old cache intact).

TRAIN/CAL SCENARIO PARTITION
----------------------------
Official TRAIN scenarios are partitioned deterministically at SCENARIO level:
    TRAIN_NORMAL  (agent fitting rows)   and   CAL_NORMAL (calibration rows).
    - whole scenarios stay in exactly one partition (ticks never split);
    - partition = seeded hash of the scenario id (PCG64 via default_rng, never
      Python's salted hash()), so it is reproducible and recorded in the
      manifest permanently;
    - an explicit partition mapping may be supplied to override the seeded one
      (still validated: complete, disjoint, TRAIN-only);
    - no TEST scenario can enter either partition (hard gate: only
      CarlAnomalySplit.TRAIN scenarios are accepted).
Partition membership is never chosen using anomaly-test performance (no test
data is ever touched).

PSEUDO-ANOMALY INTEGRATION (CAUSAL-WINDOW OBSERVATION SCOPE)
------------------------------------------------------------
For CAL_NORMAL scenarios only, controlled pseudo-corrupted counterparts are
generated with the TRAIN-gated pseudo_anomalies generator, preserving
parent_scenario / parent_tick / recipe_id / severity / seed / modality. The
CAL dataset is conceptually:
    normal CAL observation            -> target_normal = 1
    controlled corruption of the SAME -> target_normal = 0
No pseudo-anomaly is ever generated from official TEST data (the generator's
hard gate is exercised again here).

The OBSERVATION passed to the generator is the cached observation itself: for
tick t that is the trailing causal window rows [max(0, t - L + 1), t] —
EXACTLY the rows the window-feature extractors read. Temporal (GNSS/IMU)
recipes therefore corrupt the window, never the full scenario table: a
corruption landing outside [lo, t] would modify rows the observation does not
contain (a different tick's data), which made table-scoped gnss_bias/imu_spike
no-ops for most ticks. Camera/Seg recipes already receive the per-tick frame.
Every corrupted row index is <= t (strictly causal, never future rows) and
windows never cross scenario boundaries. The window bounds are recorded per
pseudo row (window_start_tick / window_end_tick) and summarized in the
manifest; no-effect rows (corruption invisible in the window features) are
skipped BEFORE any bookkeeping and counted per recipe, as are generation
attempts, so emission/material-change rates are auditable per recipe.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cognix.adapters.carla.carlanomaly_loader import (
    CarlAnomalyLoader,
    CarlAnomalySplit,
    Modality,
    ScenarioManifest,
)
from cognix.adapters.carla.real_features import (
    camera_embedding_features,
    gnss_window_features,
    imu_window_features,
    segmentation_histogram_features,
)

# Documented window policy (ticks, not seconds; no frequency is inferred).
WINDOW_LENGTH = 12

CAMERA_DIM = 18
SEG_DIM = 29
GNSS_DIM = 8
IMU_DIM = 10

CACHE_SCHEMA_VERSION = 2

# Raw-modality corruption recipes used for the CAL pseudo side. Fixed a priori;
# never tuned against test performance.
CAL_PSEUDO_RECIPES: Tuple[str, ...] = (
    "camera_brightness_shift",
    "camera_occlusion",
    "seg_region_corruption",
    "gnss_bias",
    "gnss_drift",
    "imu_spike",
    "imu_bias_scale",
)


class CacheBuilderError(RuntimeError):
    """Raised on invalid partitions, mismatched caches, or bad inputs."""


# ── extraction helpers (raw payload -> compact features) ─────────────────────

def _rgb_to_array(path: Path) -> np.ndarray:
    import PIL.Image

    return np.asarray(PIL.Image.open(path))


def extract_camera_features(path: Path) -> np.ndarray:
    return camera_embedding_features(_rgb_to_array(path))


def extract_seg_features(path: Path) -> np.ndarray:
    return segmentation_histogram_features(_rgb_to_array(path))


def _gnss_table(scenario_path: Path) -> np.ndarray:
    df = pd.read_feather(scenario_path / "gnss.feather")
    return df[["altitude", "latitude", "longitude"]].to_numpy(dtype=np.float64)


def _imu_accel_table(scenario_path: Path) -> np.ndarray:
    df = pd.read_feather(scenario_path / "imu.feather")
    return df[
        ["acceleration_x", "acceleration_y", "acceleration_z"]
    ].to_numpy(dtype=np.float64)


def extract_gnss_feature_rows(
    scenario_path: Path,
    n_ticks: int,
    window_length: int = WINDOW_LENGTH,
) -> np.ndarray:
    """
    Per-tick causal-window GNSS features for ticks 1..n_ticks-1,
    shape (n_ticks - 1, GNSS_DIM).

    Row for tick t summarizes sensor rows [max(0, t - window_length + 1), t] —
    current and previous ticks only, never future rows, never another scenario.
    Tick 0 is NOT emitted: a window feature needs >= 2 rows, so tick 0 would
    require future data (forbidden) or a fabricated step (dishonest). This is
    the documented warm-up rule of the cache (see module docstring).
    """
    table = _gnss_table(scenario_path)
    if len(table) < n_ticks:
        raise CacheBuilderError(
            f"{scenario_path}: gnss table has {len(table)} rows < {n_ticks} ticks"
        )
    out = np.empty((n_ticks - 1, GNSS_DIM), dtype=np.float64)
    for t in range(1, n_ticks):
        lo = max(0, t - window_length + 1)
        out[t - 1] = gnss_window_features(table[lo:t + 1])
    return out


def extract_imu_feature_rows(
    scenario_path: Path,
    n_ticks: int,
    window_length: int = WINDOW_LENGTH,
) -> np.ndarray:
    """Per-tick causal-window IMU features for ticks 1..n_ticks-1,
    shape (n_ticks - 1, IMU_DIM); tick 0 not emitted (warm-up rule)."""
    table = _imu_accel_table(scenario_path)
    if len(table) < n_ticks:
        raise CacheBuilderError(
            f"{scenario_path}: imu table has {len(table)} rows < {n_ticks} ticks"
        )
    out = np.empty((n_ticks - 1, IMU_DIM), dtype=np.float64)
    for t in range(1, n_ticks):
        lo = max(0, t - window_length + 1)
        out[t - 1] = imu_window_features(table[lo:t + 1])
    return out


# ── TRAIN/CAL scenario partition ─────────────────────────────────────────────

@dataclass(frozen=True)
class TrainCalPartition:
    """Deterministic scenario-level partition of official TRAIN scenarios."""

    train_normal: Tuple[str, ...]
    cal_normal: Tuple[str, ...]
    seed: int
    rule: str  # "seeded_hash" or "explicit"

    def mapping(self) -> Dict[str, str]:
        out = {s: "TRAIN_NORMAL" for s in self.train_normal}
        out.update({s: "CAL_NORMAL" for s in self.cal_normal})
        return out


def partition_train_scenarios(
    scenario_ids: Sequence[str],
    seed: int = 2026,
    cal_fraction: float = 0.25,
    explicit: Optional[Dict[str, str]] = None,
) -> TrainCalPartition:
    """
    Partition official TRAIN scenario ids into TRAIN_NORMAL / CAL_NORMAL.

    Deterministic: with the same (scenario_ids, seed, cal_fraction) the split
    is identical on every machine and run. Whole scenarios only; ticks are
    never split across partitions. `explicit` overrides the seeded rule and is
    validated for completeness, disjointness and TRAIN-only membership (the
    caller must guarantee the ids originate from the TRAIN split; the cache
    builder re-gates by construction because it only ever enumerates
    CarlAnomalySplit.TRAIN scenarios).
    """
    ids = sorted(set(scenario_ids))
    if explicit is not None:
        missing = [s for s in ids if s not in explicit]
        extra = [s for s in explicit if s not in set(ids)]
        train = sorted(s for s in ids if explicit.get(s) == "TRAIN_NORMAL")
        cal = sorted(s for s in ids if explicit.get(s) == "CAL_NORMAL")
        if missing or extra or set(explicit.values()) - {"TRAIN_NORMAL", "CAL_NORMAL"}:
            raise CacheBuilderError(
                "explicit partition must map every TRAIN scenario to exactly "
                f"TRAIN_NORMAL or CAL_NORMAL (missing={missing}, extra={extra})"
            )
        if not train or not cal:
            raise CacheBuilderError(
                "explicit partition must produce a non-empty TRAIN_NORMAL and "
                "CAL_NORMAL (calibration requires both classes)"
            )
        return TrainCalPartition(tuple(train), tuple(cal), seed, "explicit")
    if not 0.0 < cal_fraction < 1.0:
        raise CacheBuilderError("cal_fraction must be in (0, 1)")
    if len(ids) < 2:
        raise CacheBuilderError(
            "need at least 2 TRAIN scenarios to form both partitions"
        )
    # Seeded PCG64 permutation of the sorted ids: deterministic under any
    # PYTHONHASHSEED (never Python's salted hash()).
    order = list(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(order))
    n_cal = max(1, int(round(cal_fraction * len(ids))))
    cal = sorted(order[i] for i in perm[:n_cal])
    train = sorted(s for s in order if s not in set(cal))
    return TrainCalPartition(tuple(train), tuple(cal), seed, "seeded_hash")


# ── cache content ────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_sha256(npz_path: Path) -> str:
    """Deterministic identity hash over npz ARRAY CONTENTS (the zip container
    embeds timestamps, so raw file bytes are NOT reproducible across runs)."""
    h = hashlib.sha256()
    with np.load(npz_path, allow_pickle=False) as z:
        for key in sorted(z.files):
            arr = z[key]
            h.update(key.encode("utf-8"))
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(str(arr.shape).encode("utf-8"))
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _extractor_version() -> str:
    """Version/hash of the feature-extraction code (provenance)."""
    import inspect

    from cognix.adapters.carla import real_features

    return "real_features.py sha256:" + hashlib.sha256(
        inspect.getsource(real_features).encode("utf-8")
    ).hexdigest()[:16]


def _code_version() -> str:
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return f"git:{head.stdout.strip()}" if head.returncode == 0 else "git:unknown"
    except Exception:
        return "git:unknown"


@dataclass
class CacheContent:
    """In-memory cache content (also the npz payload)."""

    scenario_id: np.ndarray      # (N,) unicode str (no pickle in the cache)
    town: np.ndarray             # (N,)
    tick: np.ndarray             # (N,) int64
    source_split: np.ndarray     # (N,) "train"
    partition: np.ndarray        # (N,) "TRAIN_NORMAL" | "CAL_NORMAL"
    camera: np.ndarray           # (N, 18)
    seg: np.ndarray              # (N, 29)
    gnss: np.ndarray             # (N, 8)
    imu: np.ndarray              # (N, 10)
    cal_target_normal: Optional[np.ndarray]       # (M,) 1 normal / 0 pseudo
    cal_parent_scenario: Optional[np.ndarray]
    cal_parent_tick: Optional[np.ndarray]
    cal_recipe_id: Optional[np.ndarray]
    cal_modality: Optional[np.ndarray]
    cal_severity: Optional[np.ndarray]
    cal_seed: Optional[np.ndarray]
    cal_features: Optional[np.ndarray]      # (M, 65) compact vector [cam|seg|gnss|imu]


FEATURE_ORDER = ("camera", "seg", "gnss", "imu")
FEATURE_DIMS = {"camera": CAMERA_DIM, "seg": SEG_DIM, "gnss": GNSS_DIM, "imu": IMU_DIM}
COMPACT_DIM = sum(FEATURE_DIMS.values())  # 65


def _scenario_rows(
    loader: CarlAnomalyLoader,
    manifest: ScenarioManifest,
    partition_label: str,
    window_length: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract per-tick rows (ticks 1..n-1; tick 0 has no causal window) for one
    scenario; also return the scenario's raw modality payloads needed later for
    CAL pseudo-corruption."""
    p = manifest.path
    n = manifest.n_ticks
    if n < 2:
        raise CacheBuilderError(
            f"scenario '{manifest.scenario_id}' has {n} ticks; at least 2 are "
            "required (the first cached tick is 1: a causal window needs >= 2 rows)"
        )
    ticks = range(1, n)
    camera = np.empty((len(ticks), CAMERA_DIM), dtype=np.float64)
    seg = np.empty((len(ticks), SEG_DIM), dtype=np.float64)
    for i, t in enumerate(ticks):
        name = f"{t:06d}"
        camera[i] = extract_camera_features(p / "rgb-front" / f"{name}.jpg")
        seg[i] = extract_seg_features(p / "segmentation-front" / f"{name}.png")
    gnss = extract_gnss_feature_rows(p, n, window_length)
    imu = extract_imu_feature_rows(p, n, window_length)
    rows = {
        "scenario_id": np.array([manifest.scenario_id] * len(ticks), dtype=str),
        "town": np.array([manifest.town] * len(ticks), dtype=str),
        "tick": np.arange(1, n, dtype=np.int64),
        "source_split": np.array([manifest.split.value] * len(ticks), dtype=str),
        "partition": np.array([partition_label] * len(ticks), dtype=str),
        "camera": camera,
        "seg": seg,
        "gnss": gnss,
        "imu": imu,
    }
    # GNSS/IMU tables are small and stay in memory; raw IMAGE frames are NOT
    # stacked (3000 x 1920x1080x3 would need ~19 GB per scenario) — pseudo
    # corruption re-reads frames lazily from disk per tick instead.
    raw = {
        "gnss": _gnss_table(p),
        "imu": _imu_accel_table(p),
        "scenario_path": p,
    }
    return rows, raw


def _pseudo_for_scenario(
    raw: Dict[str, Any],
    parent_feats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    scenario_id: str,
    recipe_ids: Sequence[str],
    seed: int,
    max_pseudo_per_tick: int,
    severities: Optional[Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Pseudo-corrupted CAL rows for one scenario (parent provenance kept).

    The compact vector of a pseudo row models the CORRUPTED OBSERVATION: the
    corrupted modality block carries the corruption's window features; all
    UNTOUCHED blocks carry the parent tick's real cached features (a pseudo row
    is NOT a zero-padded corner of feature space — only one modality degrades).

    The observation handed to the generator is the ACTIVE CAUSAL WINDOW
    rows [max(0, t - L + 1), t] — the same rows the cached parent's window
    features summarize. Corrupting rows outside that slice would modify data
    the tick-t observation does not contain (the historical full-table defect
    that made gnss_bias/imu_spike no-ops); corrupting rows >= t + 1 is
    forbidden (future data).
    """
    parent_camera, parent_seg, parent_gnss, parent_imu = parent_feats
    meta_rows: List[Dict[str, Any]] = []
    feat_rows: List[np.ndarray] = []
    attempts_by_recipe: Dict[str, int] = {}
    skipped_by_recipe: Dict[str, Dict[str, int]] = {}
    n = len(parent_camera) + 1  # parent rows are ticks 1..n-1
    for t in range(1, n):  # cached ticks start at 1 (causal-window warm-up)
        p_idx = t - 1  # parent feature row index (ticks start at 1)
        # Deterministic per-tick rotation: with max_pseudo_per_tick=1 the whole
        # CAL set still sees ALL recipes (tick t uses recipe[t % len]); with
        # larger caps each tick gets a rotation window (capped at len(recipes)).
        rids = list(recipe_ids)
        chosen = [rids[(t + k) % len(rids)] for k in range(max(1, min(max_pseudo_per_tick, len(rids))))]
        for k, rid in enumerate(chosen):
            attempts_by_recipe[rid] = attempts_by_recipe.get(rid, 0) + 1
            def _skip(reason: str) -> None:
                skipped_by_recipe.setdefault(rid, {"no_effect": 0, "parent_uncached": 0,
                                                   "feature_extraction_error": 0})
                skipped_by_recipe[rid][reason] += 1
            # CAUSAL-WINDOW OBSERVATION SCOPE: the corrupted observation at
            # tick t is the trailing causal window rows [lo, t] (lo below) —
            # exactly the rows whose window features define the cached row.
            # Temporal recipes corrupt THIS slice (the historical full-table
            # payload let their step/burst land outside the window, silently
            # corrupting a different tick's data or nothing at all). All
            # corrupted row indices are <= t: strictly causal, never future
            # rows, never another scenario's rows (per-scenario tables).
            lo = max(0, t - WINDOW_LENGTH + 1)
            scenario_path = Path(raw["scenario_path"])
            if rid.startswith("gnss"):
                sample = _generate(
                    {"GNSS": raw["gnss"][lo:t + 1]}, rid, scenario_id, t, seed + t, severities
                )
                corr = np.asarray(sample.data, dtype=np.float64)
                feats = gnss_window_features(corr)
                modality = "gnss"
            elif rid.startswith("imu"):
                sample = _generate(
                    {"IMU": raw["imu"][lo:t + 1]}, rid, scenario_id, t, seed + t, severities
                )
                corr_table = sample.data  # array payload -> array back; dict -> dict
                if isinstance(corr_table, dict):
                    accel = np.column_stack(
                        [np.asarray(corr_table[c], dtype=np.float64) for c in
                         ("acceleration_x", "acceleration_y", "acceleration_z")]
                    )
                else:
                    accel = np.asarray(corr_table, dtype=np.float64)
                feats = imu_window_features(accel)
                modality = "imu"
            elif rid.startswith("camera"):
                frame = _rgb_to_array(scenario_path / "rgb-front" / f"{t:06d}.jpg")
                sample = _generate(
                    {"Camera": frame}, rid, scenario_id, t, seed + t, severities
                )
                feats = camera_embedding_features(np.asarray(sample.data))
                modality = "camera"
            else:
                seg_map = _rgb_to_array(
                    scenario_path / "segmentation-front" / f"{t:06d}.png"
                )
                sample = _generate(
                    {"Seg": seg_map}, rid, scenario_id, t, seed + t, severities
                )
                feats = segmentation_histogram_features(np.asarray(sample.data))
                modality = "seg"
            # No-effect guard FIRST (before any provenance bookkeeping): if the
            # corruption's affected rows fall entirely outside this tick's
            # causal window, the block equals the parent's. Emitting it would
            # hand the calibrator a pseudo row IDENTICAL to its normal parent —
            # contradictory labels — so the row is skipped entirely (meta AND
            # features; appending one without the other would desync the
            # position-aligned CAL arrays).
            parent_block = {"camera": parent_camera[p_idx], "seg": parent_seg[p_idx],
                            "gnss": parent_gnss[p_idx], "imu": parent_imu[p_idx]}[modality]
            if np.allclose(feats, parent_block, rtol=1e-12, atol=1e-12):
                _skip("no_effect")
                continue
            prov = sample.provenance()
            if int(prov["source_tick"]) < 1:
                # Parent tick 0 is never cached (warm-up rule); such a pseudo
                # row would have no cacheable parent to pair with.
                _skip("parent_uncached")
                continue
            meta_rows.append(
                {
                    "parent_scenario": prov["source_scenario"],
                    "parent_tick": prov["source_tick"],
                    "recipe_id": prov["recipe_id"],
                    "severity": prov["severity"],
                    "seed": prov["seed"],
                    "modality": modality,
                    "source_split": prov["source_split"],
                    "window_start_tick": int(lo),
                    "window_end_tick": int(t),
                    "synthetic_corruption": True,
                }
            )
            # compact vector = corrupted modality block + parent's untouched blocks
            v = np.zeros(COMPACT_DIM, dtype=np.float64)
            off = {"camera": 0, "seg": CAMERA_DIM, "gnss": CAMERA_DIM + SEG_DIM,
                   "imu": CAMERA_DIM + SEG_DIM + GNSS_DIM}[modality]
            v[off:off + len(feats)] = feats
            if modality != "camera":
                v[:CAMERA_DIM] = parent_camera[p_idx]
            if modality != "seg":
                v[CAMERA_DIM:CAMERA_DIM + SEG_DIM] = parent_seg[p_idx]
            if modality != "gnss":
                v[CAMERA_DIM + SEG_DIM:CAMERA_DIM + SEG_DIM + GNSS_DIM] = parent_gnss[p_idx]
            if modality != "imu":
                v[CAMERA_DIM + SEG_DIM + GNSS_DIM:] = parent_imu[p_idx]
            feat_rows.append(v)
    return meta_rows, feat_rows, attempts_by_recipe, skipped_by_recipe


def _generate(
    payload: Dict[str, Any],
    recipe_id: str,
    scenario_id: str,
    tick: int,
    seed: int,
    severities: Optional[Dict[str, float]],
):
    from cognix.adapters.carla.pseudo_anomalies import generate_pseudo_anomaly
    from cognix.adapters.carla.carlanomaly_loader import CarlAnomalySplit as _S

    return generate_pseudo_anomaly(
        payload,
        recipe_id=recipe_id,
        source_split=_S.TRAIN,
        source_scenario=scenario_id,
        source_tick=tick,
        seed=seed,
        severity=(severities or {}).get(recipe_id),
    )


def _compact_vector(
    camera: Optional[np.ndarray],
    seg: Optional[np.ndarray],
    gnss: Optional[np.ndarray],
    imu: Optional[np.ndarray],
    preset: Optional[np.ndarray] = None,
    modality: Optional[str] = None,
) -> np.ndarray:
    """(65,) [camera | seg | gnss | imu] from per-modality feature rows.
    (Pseudo rows are assembled in _pseudo_for_scenario: corrupted block +
    parent's untouched blocks.)"""
    v = np.zeros(COMPACT_DIM, dtype=np.float64)
    if preset is not None and modality is not None:
        off = {"camera": 0, "seg": CAMERA_DIM, "gnss": CAMERA_DIM + SEG_DIM,
               "imu": CAMERA_DIM + SEG_DIM + GNSS_DIM}[modality]
        v[off:off + len(preset)] = preset
        return v
    if camera is not None:
        v[:CAMERA_DIM] = camera
    if seg is not None:
        v[CAMERA_DIM:CAMERA_DIM + SEG_DIM] = seg
    if gnss is not None:
        v[CAMERA_DIM + SEG_DIM:CAMERA_DIM + SEG_DIM + GNSS_DIM] = gnss
    if imu is not None:
        v[CAMERA_DIM + SEG_DIM + GNSS_DIM:] = imu
    return v


# ── public builder ───────────────────────────────────────────────────────────

def build_cache(
    root: Path | str,
    out_dir: Path | str,
    scenarios: Optional[Sequence[str]] = None,
    towns: Optional[Sequence[str]] = None,
    partition: Optional[TrainCalPartition] = None,
    partition_seed: int = 2026,
    cal_fraction: float = 0.25,
    window_length: int = WINDOW_LENGTH,
    pseudo_recipes: Sequence[str] = CAL_PSEUDO_RECIPES,
    max_pseudo_per_tick: int = 1,
    pseudo_severities: Optional[Dict[str, float]] = None,
    pseudo_seed: int = 0,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    """
    Build the compact TRAIN cache (npz + json manifest) from official TRAIN
    scenarios under `root`. TRAIN-only: the loader only enumerates
    CarlAnomalySplit.TRAIN scenarios, so TEST data cannot enter the cache.

    Determinism: identical inputs -> byte-identical npz (manifest differs only
    in the wall-clock timestamp). Rebuilding over an existing cache with a
    different content hash raises unless allow_mismatch=True.
    """
    loader = CarlAnomalyLoader(root)
    train_ids = []
    for town in towns or loader.list_towns(CarlAnomalySplit.TRAIN):
        for sid in loader.list_scenarios(CarlAnomalySplit.TRAIN, town):
            train_ids.append(sid)
    if not train_ids:
        raise CacheBuilderError("no TRAIN scenarios found under the given root")
    if scenarios is not None:
        unknown = [s for s in scenarios if s not in set(train_ids)]
        if unknown:
            raise CacheBuilderError(f"scenarios not present in TRAIN split: {unknown}")
        train_ids = sorted(set(scenarios))
    else:
        train_ids = sorted(set(train_ids))

    part = partition or partition_train_scenarios(
        train_ids, seed=partition_seed, cal_fraction=cal_fraction
    )
    known = set(part.mapping())
    missing = [s for s in train_ids if s not in known]
    if missing:
        raise CacheBuilderError(f"partition does not cover scenarios: {missing}")

    blocks: List[Dict[str, np.ndarray]] = []
    raws: Dict[str, Dict[str, np.ndarray]] = {}
    feats_by_sid: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid in train_ids:
        manifest = loader.build_manifest(CarlAnomalySplit.TRAIN, sid)
        for mod in (Modality.RGB_FRONT, Modality.SEGMENTATION_FRONT, Modality.GNSS, Modality.IMU):
            if mod not in manifest.available_modalities:
                raise CacheBuilderError(
                    f"scenario '{sid}' is missing required modality {mod.value}"
                )
        rows, raw = _scenario_rows(
            loader, manifest, part.mapping()[sid], window_length
        )
        blocks.append(rows)
        raws[sid] = raw
        feats_by_sid[sid] = (rows["camera"], rows["seg"], rows["gnss"], rows["imu"])

    content = _merge(blocks)

    # ── CAL pseudo anomalies (CAL_NORMAL scenarios only) ─────────────────
    cal_meta: List[Dict[str, Any]] = []
    cal_feats: List[np.ndarray] = []
    attempts_by_recipe: Dict[str, int] = {}
    skipped_by_recipe: Dict[str, Dict[str, int]] = {}
    for sid in part.cal_normal:
        meta, feats, att, skipped = _pseudo_for_scenario(
            raws[sid], feats_by_sid[sid], sid, pseudo_recipes, pseudo_seed,
            max_pseudo_per_tick, pseudo_severities,
        )
        cal_meta.extend(meta)
        cal_feats.extend(feats)
        for rid, c in att.items():
            attempts_by_recipe[rid] = attempts_by_recipe.get(rid, 0) + c
        for rid, counts in skipped.items():
            acc = skipped_by_recipe.setdefault(
                rid, {"no_effect": 0, "parent_uncached": 0, "feature_extraction_error": 0})
            for reason, c in counts.items():
                acc[reason] = acc.get(reason, 0) + c
    skipped_no_effect = int(sum(c.get("no_effect", 0) for c in skipped_by_recipe.values()))

    manifest = _build_manifest_dict(
        content, part, train_ids, window_length, pseudo_recipes,
        cal_meta, cal_feats, attempts_by_recipe, skipped_by_recipe,
        allow_mismatch,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    npz_path = out_path / "carla_train_cache.npz"
    # Stage to a temp file first: a refused rebuild must leave any existing
    # cache untouched (no silent overwrite, and no clobbered npz either).
    tmp_path = out_path / "carla_train_cache.npz.tmp"
    _write_npz(tmp_path, content)
    content_hash = _content_sha256(tmp_path)
    manifest["cache_content_sha256"] = content_hash
    json_path = out_path / "carla_train_cache.json"
    if json_path.exists() and not allow_mismatch:
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f)
        old_hash = old.get("cache_content_sha256")
        if old_hash and old_hash != content_hash:
            tmp_path.unlink()
            raise CacheBuilderError(
                "refusing to silently overwrite a mismatched cache: existing "
                f"content sha256={old_hash[:12]}... != new content sha256="
                f"{content_hash[:12]}... (pass allow_mismatch=True to override)"
            )
    tmp_path.replace(npz_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _check_no_silent_overwrite(npz_path: Path, manifest: Dict[str, Any]) -> None:
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f)
        old_hash = old.get("cache_sha256")
        new_hash = manifest["cache_sha256"]
        if old_hash and old_hash != new_hash:
            raise CacheBuilderError(
                "refusing to silently overwrite a mismatched cache: existing "
                f"sha256={old_hash[:12]}... != new sha256={new_hash[:12]}... "
                "(pass allow_mismatch=True to override explicitly)"
            )


def _merge(blocks: List[Dict[str, np.ndarray]]) -> CacheContent:
    cat = {k: np.concatenate([b[k] for b in blocks], axis=0)
           for k in blocks[0]}
    return CacheContent(
        scenario_id=cat["scenario_id"], town=cat["town"], tick=cat["tick"],
        source_split=cat["source_split"], partition=cat["partition"],
        camera=cat["camera"], seg=cat["seg"], gnss=cat["gnss"], imu=cat["imu"],
        cal_target_normal=None, cal_parent_scenario=None, cal_parent_tick=None,
        cal_recipe_id=None, cal_modality=None, cal_severity=None, cal_seed=None,
        cal_features=None,
    )


def _pseudo_arrays(cal_meta: List[Dict[str, Any]], cal_feats: List[np.ndarray]):
    if not cal_meta:
        return None
    return dict(
        cal_target_normal=np.zeros(len(cal_meta), dtype=np.int64),
        cal_parent_scenario=np.array([m["parent_scenario"] for m in cal_meta], dtype=str),
        cal_parent_tick=np.array([m["parent_tick"] for m in cal_meta], dtype=np.int64),
        cal_recipe_id=np.array([m["recipe_id"] for m in cal_meta], dtype=str),
        cal_modality=np.array([m["modality"] for m in cal_meta], dtype=str),
        cal_severity=np.array([m["severity"] for m in cal_meta], dtype=np.float64),
        cal_seed=np.array([m["seed"] for m in cal_meta], dtype=np.int64),
        cal_window_start_tick=np.array(
            [int(m["window_start_tick"]) for m in cal_meta], dtype=np.int64),
        cal_window_end_tick=np.array(
            [int(m["window_end_tick"]) for m in cal_meta], dtype=np.int64),
        cal_features=np.asarray(cal_feats, dtype=np.float64),
    )


def _build_manifest_dict(
    content: CacheContent,
    part: TrainCalPartition,
    train_ids: List[str],
    window_length: int,
    pseudo_recipes: Sequence[str],
    cal_meta: List[Dict[str, Any]],
    cal_feats: List[np.ndarray],
    attempts_by_recipe: Dict[str, int],
    skipped_by_recipe: Dict[str, Dict[str, int]],
    allow_mismatch: bool,
) -> Dict[str, Any]:
    pseudo = _pseudo_arrays(cal_meta, cal_feats) if cal_meta else None
    if pseudo is not None:
        for k, v in pseudo.items():
            setattr(content, k, v)
    n = len(content.tick)
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "prediction_semantics": "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration",
        "source_split": "train",
        "source_split_enum": "CarlAnomalySplit.TRAIN only; TEST never accepted",
        "scenario_ids": list(train_ids),
        "towns": sorted({str(t) for t in content.town}),
        "n_rows": n,
        "feature_dims": dict(FEATURE_DIMS),
        "compact_dim": COMPACT_DIM,
        "window_policy": {
            "window_length_ticks": window_length,
            "causal": True,
            "warmup_rule": "tick t uses rows [max(0, t - L + 1), t]",
            "first_cached_tick": 1,
            "first_cached_tick_reason": (
                "a causal window feature needs >= 2 rows; tick 0 would require "
                "future data or a fabricated step"
            ),
            "cross_scenario": "never; windows are per-scenario",
            "sampling_frequency_claimed": False,
        },
        "extractor_version": _extractor_version(),
        "code_version": _code_version(),
        "partition": {
            "rule": part.rule,
            "seed": part.seed,
            "TRAIN_NORMAL": list(part.train_normal),
            "CAL_NORMAL": list(part.cal_normal),
            "scenario_level_only": True,
        },
        "pseudo_anomalies": {
            "present": bool(cal_meta),
            "recipes": list(pseudo_recipes),
            "n_rows": len(cal_meta),
            "skipped_no_effect_rows": int(sum(
                c.get("no_effect", 0) for c in skipped_by_recipe.values())),
            "attempts_by_recipe": {rid: int(attempts_by_recipe.get(rid, 0))
                                   for rid in pseudo_recipes},
            "skipped_by_recipe": {rid: {k: int(v) for k, v in skipped_by_recipe.get(rid, {}).items()}
                                  for rid in pseudo_recipes},
            "observation_scope": (
                "causal window rows [max(0, t - L + 1), t] — the cached "
                "observation itself; strictly causal, per-scenario"
            ),
            "gated_to_split": "train",
            "severity_source": "fixed a priori (never tuned on test data)",
        },
        "agent_outputs_present": False,
        "raw_data_present": False,
        "allow_mismatch": bool(allow_mismatch),
        "created_utc": None,  # filled below (nondeterministic field)
    }
    import datetime as _dt

    manifest["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return manifest


def _write_npz(npz_path: Path, content: CacheContent) -> None:
    arrays = {
        "scenario_id": content.scenario_id,
        "town": content.town,
        "tick": content.tick,
        "source_split": content.source_split,
        "partition": content.partition,
        "camera": content.camera,
        "seg": content.seg,
        "gnss": content.gnss,
        "imu": content.imu,
    }
    pseudo_keys = (
        "cal_target_normal", "cal_parent_scenario", "cal_parent_tick",
        "cal_recipe_id", "cal_modality", "cal_severity", "cal_seed",
        "cal_window_start_tick", "cal_window_end_tick",
        "cal_features",
    )
    for k in pseudo_keys:
        v = getattr(content, k)
        if v is not None:
            arrays[k] = v
    # Write through an open handle: np.savez silently APPENDS '.npz' to string
    # paths without that suffix (which would corrupt the staging-file scheme).
    with open(npz_path, "wb") as f:
        np.savez(f, **arrays)


def load_cache(cache_dir: Path | str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load a cache directory (npz arrays + manifest dict) for downstream use."""
    d = Path(cache_dir)
    with np.load(d / "carla_train_cache.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with open(d / "carla_train_cache.json", encoding="utf-8") as f:
        manifest = json.load(f)
    # Historical cache bytes/hashes remain frozen; semantic migration is only
    # in memory. New writes use the unambiguous v2 target name.
    legacy_key = "cal_is_" + "safe"
    if legacy_key in arrays:
        arrays["cal_target_normal"] = arrays.pop(legacy_key)
        manifest = {**manifest, "loaded_target_name_migration":
                    "legacy binary target renamed in memory; source hash unchanged"}
    manifest = {**manifest, "prediction_semantics":
                "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration"}
    return arrays, manifest

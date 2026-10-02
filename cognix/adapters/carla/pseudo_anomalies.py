"""
TRAIN-only pseudo-anomaly generation for CarlAnomaly Experiment 1.

SCOPE AND ARCHITECTURE GUARDRAIL
--------------------------------
This module is CARLA/CarlAnomaly-specific functionality inside the CARLA
adapter. It does NOT touch the generic COGNIX core (no GAT, fusion, conformal,
decision, or Shapley changes), does NOT import the frozen synthetic bridge, and
imports `cognix` lazily inside functions so that `import cognix` never pulls
adapter modules.

SCIENTIFIC INTERPRETATION (NOT the real anomaly distribution)
------------------------------------------------------------
Pseudo-anomalies are CONTROLLED, SYNTHETIC, TRAIN-only corruption examples.
They are NOT replicas, samples, or simulations of the real CarlAnomaly anomaly
distribution, and they make no claim about how real anomalies look. They exist
to supply the pseudo side (target_normal = 0) for

    - normality-score -> P(target=normal) under TRAIN-derived clean-vs-pseudo calibration calibration (ScoreCalibrator), and
    - later GAT/fusion behavior studies under CONTROLLED degradation,

because the official training split contains only normal scenarios. The real
CarlAnomaly test anomalies remain EXTERNAL held-out evaluation and are never
consumed here. Severity levels are fixed a priori by this module and must
never be tuned against official test data or test performance.

HARD SPLIT GATE
---------------
Every public generation entry point requires explicit `source_split`
provenance. Only `CarlAnomalySplit.TRAIN` is accepted; `TEST_NORMAL`,
`TEST_ANOMALY`, and unknown/missing provenance fail loudly. Forgetting to
specify a split is itself an error — there is no default split.

PURITY / DETERMINISM CONTRACT
-----------------------------
    - Pure functions: source arrays are never mutated (fresh outputs only).
    - Deterministic for identical (input, recipe, seed); each sample's
      randomness comes from an explicit numpy PCG64 generator seeded by the
      integer seed. Python's salted hash() is never used.
    - Shape and schema preserved unless a recipe explicitly models missing
      data; all outputs finite and valid for their modality.
    - Every generated sample carries full provenance metadata
      (recipe_id, modality, seed, severity, source_split, source_scenario,
      source_tick, synthetic_corruption = True).
    - The recipe registry is an explicit, auditable tuple.

Raw path: raw TRAIN observation -> pseudo corruption (here) ->
real_features.py extractors -> real agent (normality.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from cognix.adapters.carla.carlanomaly_loader import CarlAnomalySplit
from cognix.adapters.carla.real_features import (
    IMU_ACCEL_COLUMNS,
    local_meter_offsets_to_degrees,
)


class PseudoAnomalyError(RuntimeError):
    """Raised on split-gate violations, bad recipes, or invalid inputs."""


# ── split gate ───────────────────────────────────────────────────────────────

def require_train_split(source_split: Any) -> CarlAnomalySplit:
    """
    Hard gate: only official TRAIN-derived observations may enter pseudo-anomaly
    generation. TEST_NORMAL / TEST_ANOMALY / unknown / missing provenance fail
    loudly. No default split exists — omission is itself a failure.
    """
    if source_split is None:
        raise PseudoAnomalyError(
            "pseudo-anomaly generation requires explicit source_split "
            "provenance; missing provenance is rejected (no default split)."
        )
    if isinstance(source_split, str):
        try:
            source_split = CarlAnomalySplit(source_split)
        except ValueError:
            raise PseudoAnomalyError(
                f"unknown source_split '{source_split}'; expected one of "
                f"{[s.value for s in CarlAnomalySplit]}."
            )
    if not isinstance(source_split, CarlAnomalySplit):
        raise PseudoAnomalyError(
            f"source_split must be a CarlAnomalySplit, got {type(source_split)!r}."
        )
    if source_split is not CarlAnomalySplit.TRAIN:
        raise PseudoAnomalyError(
            f"pseudo-anomalies may only be generated from TRAIN-derived data; "
            f"got source_split={source_split.value}. Test splits are held-out "
            "evaluation data and must never enter the generator."
        )
    return source_split


# ── provenance record ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PseudoAnomalySample:
    """A corrupted observation plus its full provenance."""

    data: Any                      # corrupted raw modality payload (never aliased to source)
    recipe_id: str
    modality: str                  # camera | seg | gnss | imu
    seed: int
    severity: float
    source_split: str              # always "train"
    source_scenario: str
    source_tick: Optional[int]
    synthetic_corruption: bool     # always True

    def provenance(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "modality": self.modality,
            "seed": self.seed,
            "severity": self.severity,
            "source_split": self.source_split,
            "source_scenario": self.source_scenario,
            "source_tick": self.source_tick,
            "synthetic_corruption": self.synthetic_corruption,
        }


# ── shared recipe helpers ────────────────────────────────────────────────────

def _as_observation_dict(observation: Any) -> Dict[str, Any]:
    """Normalize a raw TRAIN observation into a {modality_key: payload} dict.

    Accepted: dict payloads (keys Camera/Seg/GNSS/IMU) or an already-extracted
    single-modality payload passed with an explicit modality.
    """
    if isinstance(observation, dict):
        return observation
    raise PseudoAnomalyError(
        "pseudo-anomaly recipes expect a dict observation keyed by modality "
        "(Camera/Seg/GNSS/IMU); refusing to guess the payload type."
    )


def _validated_float(value: Any, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise PseudoAnomalyError(f"{name} must be a finite value >= 0, got {value!r}")
    return v


# ── camera recipes (raw uint8 (H, W, 3) RGB frames) ─────────────────────────

def _camera_rect_darken(
    observation: Any,
    severity: float,
    rng: Optional[np.random.Generator],
    recipe_id: str,
) -> PseudoAnomalySample:
    """Shared pure worker: darken a deterministic rectangle covering ~severity
    of the frame area (severity=1.0 -> the entire frame)."""
    data = _as_observation_dict(observation)
    frame = np.asarray(data["Camera"])
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise PseudoAnomalyError(
            f"camera recipes expect (H, W, 3) uint8 RGB, got {frame.shape}"
        )
    sev = _validated_float(severity, "severity")
    if not 0.0 < sev <= 1.0:
        raise PseudoAnomalyError(f"severity must be in (0, 1], got {sev}")
    h, w = frame.shape[:2]
    area_target = sev * h * w
    rh = min(h, max(1, int(round(np.sqrt(area_target * h / w)))))
    rw = min(w, max(1, int(round(np.sqrt(area_target * w / h)))))
    gen = rng if rng is not None else np.random.default_rng(0)
    top = int(gen.integers(0, h - rh + 1))
    left = int(gen.integers(0, w - rw + 1))
    out = frame.copy()
    out[top:top + rh, left:left + rw] = 0
    return PseudoAnomalySample(
        data=out,
        recipe_id=recipe_id,
        modality="camera",
        seed=0,
        severity=sev,
        source_split="train",
        source_scenario=str(data.get("scenario_id", "unknown")),
        source_tick=data.get("tick"),
        synthetic_corruption=True,
    )


def camera_brightness_shift(
    observation: Any,
    severity: float = 0.35,
    rng: Optional[np.random.Generator] = None,
    direction: str = "down",
) -> PseudoAnomalySample:
    """Exposure corruption: multiply pixel intensities by (1 -/+ severity).

    Deterministic; severity in [0, 1] (1 = full blackout for direction='down').
    Pure: the source array is never modified."""
    gate = observation.get("__split__") if isinstance(observation, dict) else None
    data = _as_observation_dict(observation)
    frame = np.asarray(data["Camera"])
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise PseudoAnomalyError(
            f"camera recipes expect (H, W, 3) uint8 RGB, got {frame.shape}"
        )
    sev = _validated_float(severity, "severity")
    if direction == "down":
        if sev > 1.0:
            raise PseudoAnomalyError(
                f"severity > 1 for direction='down' would invert the image; got {sev}"
            )
        factor = 1.0 - sev
    elif direction == "up":
        factor = 1.0 + sev
    else:
        raise PseudoAnomalyError(f"direction must be 'down' or 'up', got {direction!r}")
    out = np.clip(frame.astype(np.float64) * factor, 0.0, 255.0).astype(frame.dtype)
    return PseudoAnomalySample(
        data=out,
        recipe_id="camera_brightness_shift",
        modality="camera",
        seed=0,
        severity=sev,
        source_split=gate or "train",
        source_scenario=str(data.get("scenario_id", "unknown")),
        source_tick=data.get("tick"),
        synthetic_corruption=True,
    )


def camera_occlusion(
    observation: Any,
    severity: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> PseudoAnomalySample:
    """Rectangular occlusion covering ~severity of the frame area. The rectangle
    position is deterministic given (shape, severity, rng state)."""
    return _camera_rect_darken(observation, severity, rng, "camera_occlusion")


def camera_blackout(
    observation: Any,
    severity: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> PseudoAnomalySample:
    """Full-frame blackout / severe degradation (severity = fraction darkened;
    1.0 = entire frame black)."""
    return _camera_rect_darken(observation, severity, rng, "camera_blackout")


# ── segmentation recipes (class maps; only SOURCE-OBSERVED class ids) ────────

def seg_region_corruption(
    observation: Any,
    severity: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> PseudoAnomalySample:
    """
    Controlled region/class corruption: a deterministic rectangle's labels are
    replaced by a class id ALREADY PRESENT in the source observation (never an
    invented semantic id). severity = fraction of pixels relabeled.
    """
    gate = observation.get("__split__") if isinstance(observation, dict) else None
    data = _as_observation_dict(observation)
    seg = np.asarray(data["Seg"])
    classes = seg[..., 0] if seg.ndim == 3 else seg
    present = np.unique(classes)
    if present.size == 0:
        raise PseudoAnomalyError("segmentation corruption requires a non-empty class map")
    sev = _validated_float(severity, "severity")
    if not 0.0 < sev <= 1.0:
        raise PseudoAnomalyError(f"severity must be in (0, 1], got {sev}")
    h, w = classes.shape
    area_target = sev * h * w
    rh = max(1, int(round(np.sqrt(area_target * h / w))))
    rw = max(1, int(round(np.sqrt(area_target * w / h))))
    gen = rng if rng is not None else np.random.default_rng(0)
    top = int(gen.integers(0, h - rh + 1))
    left = int(gen.integers(0, w - rw + 1))
    # Replacement id: a class already present in the SOURCE map (auditable,
    # no invented semantics). Choose deterministically via the seeded generator.
    replacement = present[int(gen.integers(0, present.size))]
    out_classes = classes.copy()
    out_classes[top:top + rh, left:left + rw] = replacement
    if seg.ndim == 3:
        out = seg.copy()          # non-class channels preserved byte-identical
        out[..., 0] = out_classes
    else:
        out = out_classes
    return PseudoAnomalySample(
        data=out,
        recipe_id="seg_region_corruption",
        modality="seg",
        seed=0,
        severity=sev,
        source_split=gate or "train",
        source_scenario=str(data.get("scenario_id", "unknown")),
        source_tick=data.get("tick"),
        synthetic_corruption=True,
    )


# ── GNSS recipes (schema: altitude, latitude, longitude rows) ────────────────

def gnss_bias(
    observation: Any,
    severity: float = 25.0,
    rng: Optional[np.random.Generator] = None,
) -> PseudoAnomalySample:
    """Half-window position STEP-bias (GPS-jump style failure): rows before the
    window midpoint are untouched; rows from the midpoint on are offset by a
    deterministic (north=severity, east=0.6*severity, alt=severity/5) vector in
    METERS converted through the SAME local small-area approximation as
    gnss_window_features. A constant whole-window offset would be invisible to
    translation-invariant trajectory features, so the bias enters as a
    midpoint discontinuity — the observable failure mode."""
    gate = observation.get("__split__") if isinstance(observation, dict) else None
    data = _as_observation_dict(observation)
    rows = np.asarray(data["GNSS"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] != 3:
        raise PseudoAnomalyError(
            f"gnss recipes expect (n>=2, 3) [altitude, latitude, longitude], got {rows.shape}"
        )
    sev = _validated_float(severity, "severity")
    lat_ref = float(rows[:, 1].mean())
    dlat, dlon = local_meter_offsets_to_degrees(sev, 0.6 * sev, lat_ref)
    out = rows.copy()
    mid = rows.shape[0] // 2
    out[mid:, 1] += dlat
    out[mid:, 2] += dlon
    out[mid:, 0] += sev / 5.0
    return PseudoAnomalySample(
        data=out,
        recipe_id="gnss_bias",
        modality="gnss",
        seed=0,
        severity=sev,
        source_split=gate or "train",
        source_scenario=str(data.get("scenario_id", "unknown")),
        source_tick=data.get("tick"),
        synthetic_corruption=True,
    )


def gnss_drift(
    observation: Any,
    severity: float = 50.0,
    rng: Optional[np.random.Generator] = None,
) -> PseudoAnomalySample:
    """Gradual linear drift ramping to a total `severity` meters displacement
    over the window (unit-correct: degrees via the documented approximation)."""
    gate = observation.get("__split__") if isinstance(observation, dict) else None
    data = _as_observation_dict(observation)
    rows = np.asarray(data["GNSS"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] != 3:
        raise PseudoAnomalyError(
            f"gnss recipes expect (n>=2, 3) [altitude, latitude, longitude], got {rows.shape}"
        )
    sev = _validated_float(severity, "severity")
    n = rows.shape[0]
    frac = np.linspace(0.0, 1.0, n)
    lat_ref = float(rows[:, 1].mean())
    dlat_total, dlon_total = local_meter_offsets_to_degrees(sev, 0.4 * sev, lat_ref)
    out = rows.copy()
    out[:, 1] += dlat_total * frac
    out[:, 2] += dlon_total * frac
    out[:, 0] += (sev / 5.0) * frac
    return PseudoAnomalySample(
        data=out,
        recipe_id="gnss_drift",
        modality="gnss",
        seed=0,
        severity=sev,
        source_split=gate or "train",
        source_scenario=str(data.get("scenario_id", "unknown")),
        source_tick=data.get("tick"),
        synthetic_corruption=True,
    )


# ── IMU recipes (acceleration columns only; orientation preserved) ───────────

def _imu_accel_matrix(table: Any) -> np.ndarray:
    """Extract the (n, 3) acceleration matrix from either an array payload or a
    per-column dict payload keyed by the verified IMU column names."""
    if isinstance(table, dict):
        cols = []
        for c in IMU_ACCEL_COLUMNS:
            if c not in table:
                raise PseudoAnomalyError(
                    f"IMU dict payload is missing verified acceleration column '{c}'"
                )
            cols.append(np.asarray(table[c], dtype=np.float64).ravel())
        if len({len(c) for c in cols}) != 1:
            raise PseudoAnomalyError("IMU acceleration columns have unequal lengths")
        return np.column_stack(cols)
    return np.asarray(table, dtype=np.float64)


def imu_spike(
    observation: Any,
    severity: float = 15.0,
    rng: Optional[np.random.Generator] = None,
) -> PseudoAnomalySample:
    """Transient acceleration spike: a short burst of magnitude `severity`
    added to ONE axis at a seeded position. Orientation columns (compass,
    longitude_*) are preserved untouched."""
    gate = observation.get("__split__") if isinstance(observation, dict) else None
    data = _as_observation_dict(observation)
    table = data["IMU"]
    accel = _imu_accel_matrix(table)
    if accel.ndim != 2 or accel.shape[0] < 2 or accel.shape[1] != 3:
        raise PseudoAnomalyError(
            f"imu recipes expect (n>=2, 3) acceleration columns, got {accel.shape}"
        )
    sev = _validated_float(severity, "severity")
    gen = rng if rng is not None else np.random.default_rng(0)
    n = accel.shape[0]
    burst = max(1, min(3, n - 1))
    start = int(gen.integers(0, n - burst))
    axis = int(gen.integers(0, 3))
    out_accel = accel.copy()
    out_accel[start:start + burst, axis] += sev
    out = _rebuild_imu_table(table, out_accel)
    return PseudoAnomalySample(
        data=out,
        recipe_id="imu_spike",
        modality="imu",
        seed=0,
        severity=sev,
        source_split=gate or "train",
        source_scenario=str(data.get("scenario_id", "unknown")),
        source_tick=data.get("tick"),
        synthetic_corruption=True,
    )


def imu_bias_scale(
    observation: Any,
    severity: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> PseudoAnomalySample:
    """Acceleration bias + scale corruption: accel -> (1 + severity) * accel +
    severity (constant offset on each acceleration axis). Orientation columns
    are preserved untouched."""
    gate = observation.get("__split__") if isinstance(observation, dict) else None
    data = _as_observation_dict(observation)
    table = data["IMU"]
    accel = _imu_accel_matrix(table)
    if accel.ndim != 2 or accel.shape[0] < 2 or accel.shape[1] != 3:
        raise PseudoAnomalyError(
            f"imu recipes expect (n>=2, 3) acceleration columns, got {accel.shape}"
        )
    sev = _validated_float(severity, "severity")
    out_accel = (1.0 + sev) * accel + sev
    out = _rebuild_imu_table(table, out_accel)
    return PseudoAnomalySample(
        data=out,
        recipe_id="imu_bias_scale",
        modality="imu",
        seed=0,
        severity=sev,
        source_split=gate or "train",
        source_scenario=str(data.get("scenario_id", "unknown")),
        source_tick=data.get("tick"),
        synthetic_corruption=True,
    )


def _rebuild_imu_table(table: Any, corrupted_accel: np.ndarray) -> Any:
    """Return the IMU payload with corrupted acceleration columns and ALL other
    columns (e.g. compass, longitude_*) preserved by value (copy, not alias)."""
    if isinstance(table, dict):
        out = {}
        for i, col in enumerate(IMU_ACCEL_COLUMNS):
            out[col] = corrupted_accel[:, i]
        for col, val in table.items():
            if col not in out:
                out[col] = np.array(val, copy=True)
        return out
    return corrupted_accel.copy()


# ── auditable recipe registry ────────────────────────────────────────────────

RECIPES: Tuple[Tuple[str, str, Callable], ...] = (
    ("camera_brightness_shift", "camera", camera_brightness_shift),
    ("camera_occlusion", "camera", camera_occlusion),
    ("camera_blackout", "camera", camera_blackout),
    ("seg_region_corruption", "seg", seg_region_corruption),
    ("gnss_bias", "gnss", gnss_bias),
    ("gnss_drift", "gnss", gnss_drift),
    ("imu_spike", "imu", imu_spike),
    ("imu_bias_scale", "imu", imu_bias_scale),
)

_RECIPE_BY_ID = {rid: fn for rid, _mod, fn in RECIPES}


def recipe_registry() -> Tuple[Tuple[str, str], ...]:
    """Auditable (recipe_id, modality) listing — the complete recipe set."""
    return tuple((rid, mod) for rid, mod, _fn in RECIPES)


# ── single gated entry point ────────────────────────────────────────────────

def generate_pseudo_anomaly(
    observation: Any,
    recipe_id: str,
    source_split: Any,
    source_scenario: str = "unknown",
    source_tick: Optional[int] = None,
    seed: int = 0,
    severity: Optional[float] = None,
) -> PseudoAnomalySample:
    """
    THE public generation entry point. `source_split` is mandatory and must be
    CarlAnomalySplit.TRAIN (the hard gate rejects everything else).

    `observation` is a dict with one modality payload keyed 'Camera', 'Seg',
    'GNSS' or 'IMU'. Recipes are pure: the input is never mutated; determinism
    is a function of (observation, recipe_id, seed, severity).
    """
    split = require_train_split(source_split)  # HARD GATE — first thing, always
    if recipe_id not in _RECIPE_BY_ID:
        raise PseudoAnomalyError(
            f"unknown recipe_id '{recipe_id}'; available: {sorted(_RECIPE_BY_ID)}"
        )
    if not isinstance(observation, dict):
        raise PseudoAnomalyError("observation must be a modality-keyed dict")
    if isinstance(observation.get("__split__"), str):
        try:
            observed_split = CarlAnomalySplit(observation["__split__"])
        except ValueError:
            raise PseudoAnomalyError(
                f"observation carries unknown __split__ provenance "
                f"'{observation['__split__']}'"
            )
        if observed_split is not split:
            raise PseudoAnomalyError(
                f"observation carries __split__='{observed_split.value}' but "
                f"source_split='{split.value}'; provenance mismatch."
            )
    if observation.get("__split__") is None and "__split__" in observation:
        raise PseudoAnomalyError("observation carries missing __split__ provenance")
    rng = np.random.default_rng(int(seed))  # PCG64 — explicit, never hash()
    recipe = _RECIPE_BY_ID[recipe_id]
    payload = dict(observation)
    payload.setdefault("scenario_id", source_scenario)
    if source_tick is not None:
        payload.setdefault("tick", source_tick)
    if severity is not None:
        # severity travels through the recipe's keyword (all recipes accept it).
        sample = recipe(payload, severity=severity, rng=rng)
    else:
        sample = recipe(payload, rng=rng)
    # Rebuild provenance with the CALLER-AUTHORITATIVE values (recipes only see
    # payloads; the entry point owns the record).
    return PseudoAnomalySample(
        data=sample.data,
        recipe_id=sample.recipe_id,
        modality=sample.modality,
        seed=int(seed),
        severity=sample.severity,
        source_split=split.value,
        source_scenario=source_scenario,
        source_tick=source_tick,
        synthetic_corruption=True,
    )


# ── CAL pair construction (calibration integration scope) ───────────────────

def build_calibration_pairs(
    normal_observations,
    source_split: Any,
    recipe_ids,
    seed: int = 0,
    severities: Optional[Dict[str, float]] = None,
):
    """
    Construct CAL examples in the Stage-1 calibration format:

        normal TRAIN-derived observation              -> target_normal = 1
        pseudo-corrupted version of the SAME observation -> target_normal = 0

    Parent-sample provenance is preserved (each pseudo row records the scenario
    and tick of the normal observation it was derived from). Split separation
    stays explicit: the gate rejects any non-TRAIN provenance. Official TEST
    labels are never read; no artifact is fitted here.

    Returns (scores_ready_observations, target_normal, provenance_rows) where
    observations is a list of raw modality dicts (normal first, then corrupted)
    in row-aligned order with target_normal.
    """
    split = require_train_split(source_split)
    rows = []
    observations = []
    labels = []
    for i, obs in enumerate(normal_observations):
        observations.append(dict(obs))
        labels.append(1.0)
        rows.append({
            "kind": "normal",
            "source_scenario": str(obs.get("scenario_id", f"obs-{i}")),
            "source_tick": obs.get("tick"),
            "recipe_id": None,
            "seed": None,
            "severity": None,
            "source_split": split.value,
        })
        for rid in recipe_ids:
            corrupted = generate_pseudo_anomaly(
                obs,
                recipe_id=rid,
                source_split=split,
                source_scenario=str(obs.get("scenario_id", f"obs-{i}")),
                source_tick=obs.get("tick"),
                seed=seed + i,
                severity=(severities or {}).get(rid),
            )
            observations.append({
                "Camera": corrupted.data if rid.startswith("camera") else obs.get("Camera"),
                "Seg": corrupted.data if rid.startswith("seg") else obs.get("Seg"),
                "GNSS": corrupted.data if rid.startswith("gnss") else obs.get("GNSS"),
                "IMU": corrupted.data if rid.startswith("imu") else obs.get("IMU"),
                "scenario_id": str(obs.get("scenario_id", f"obs-{i}")),
                "tick": obs.get("tick"),
            })
            labels.append(0.0)
            prov = corrupted.provenance()
            prov["kind"] = "pseudo_anomaly"
            rows.append(prov)
    return observations, np.asarray(labels, dtype=np.float64), rows

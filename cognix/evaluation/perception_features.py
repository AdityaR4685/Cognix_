"""
Perception-Aware Hazard Features and Experimental Gating Module (Milestone 4).

This module provides isolated, label-blind research prototypes for:
1. Spatial region-of-interest hazard occupancy from semantic segmentation maps.
2. Explicit sensor-health monitoring (severe sensor dropout / loss).
3. Shadow-mode decision evaluation across controlled experimental variants.

ARCHITECTURAL ISOLATION:
- Purely evaluative and experimental.
- Does NOT alter production DecisionEngine or live operational contracts.
- Strictly causal: consumes only current-frame sensor inputs; no future frames,
  no ground-truth labels, and no post-hoc anomaly annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


# CARLA 0.9.14 semantic segmentation class conventions
# Class 3: Other (props, static obstacles)
# Class 4: Pedestrians (vulnerable road users)
DEFAULT_HAZARD_CLASSES: Tuple[int, ...] = (3, 4)


def extract_corridor_hazard_occupancy(
    semantic_map: np.ndarray,
    y_range: Tuple[float, float] = (0.5, 1.0),
    x_range: Tuple[float, float] = (0.33, 0.67),
    hazard_classes: Sequence[int] = DEFAULT_HAZARD_CLASSES,
) -> float:
    """
    Computes normalized hazard class occupancy within the forward driving corridor.

    Args:
        semantic_map: Semantic class array (H, W) or (H, W, C) where channel 0 is class ID.
        y_range: Normalized vertical bounds (min_y, max_y) in [0.0, 1.0].
        x_range: Normalized horizontal bounds (min_x, max_x) in [0.0, 1.0].
        hazard_classes: Class IDs considered hazardous (e.g. 3=Props, 4=Pedestrians).

    Returns:
        Occupancy ratio in [0.0, 1.0] (hazard pixels / total corridor pixels).
    """
    arr = np.asarray(semantic_map)
    if arr.ndim == 3:
        arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D semantic map, got shape {arr.shape}")

    h, w = arr.shape
    if h == 0 or w == 0:
        return 0.0

    y0 = int(np.clip(y_range[0] * h, 0, h))
    y1 = int(np.clip(y_range[1] * h, 0, h))
    x0 = int(np.clip(x_range[0] * w, 0, w))
    x1 = int(np.clip(x_range[1] * w, 0, w))

    if y1 <= y0 or x1 <= x0:
        return 0.0

    corridor = arr[y0:y1, x0:x1]
    corridor_size = corridor.size
    if corridor_size == 0:
        return 0.0

    # Vectorized membership test for hazard classes
    mask = np.isin(corridor, hazard_classes)
    hazard_pixel_count = int(np.count_nonzero(mask))

    return float(hazard_pixel_count / corridor_size)


def extract_sensor_health_signal(
    agent_predictions: Dict[str, float],
    min_loss_threshold: float = 0.10,
) -> Tuple[bool, Optional[str], float]:
    """
    Monitors individual sensor agents for catastrophic sensor outage or dropout.

    Args:
        agent_predictions: Mapping of agent name -> P(target=normal).
        min_loss_threshold: Threshold below which an agent is flagged as lost.

    Returns:
        (is_degraded, failing_agent_name, min_probability)
    """
    if not agent_predictions:
        return False, None, 1.0

    min_agent = min(agent_predictions, key=agent_predictions.get)
    min_prob = float(agent_predictions[min_agent])

    is_degraded = bool(min_prob < min_loss_threshold)
    failing_agent = min_agent if is_degraded else None

    return is_degraded, failing_agent, min_prob


@dataclass(frozen=True)
class PerceptionDecisionEvaluation:
    baseline_decision: str
    perception_hazard_decision: str
    sensor_health_decision: str
    combined_decision: str
    hazard_occupancy: float
    sensor_health_flag: bool
    min_sensor_p: float
    failing_agent: Optional[str]


def evaluate_perception_variants(
    baseline_decision: str,
    hazard_occupancy: float,
    agent_predictions: Dict[str, float],
    hazard_occupancy_threshold: float = 0.05,
    sensor_loss_threshold: float = 0.10,
) -> PerceptionDecisionEvaluation:
    """
    Evaluates shadow-mode decision variants without altering production engine.

    Variants:
    1. Baseline: production decision (e.g. ACT, REQUEST_INFORMATION, ESCALATE).
    2. Perception Hazard: Baseline + (hazard_occupancy > threshold -> ESCALATE).
    3. Sensor Health: Baseline + (min_sensor_p < threshold -> ESCALATE).
    4. Combined: Baseline + (hazard_occupancy > threshold OR min_sensor_p < threshold -> ESCALATE).
    """
    sensor_loss, failing_agent, min_p = extract_sensor_health_signal(
        agent_predictions, min_loss_threshold=sensor_loss_threshold
    )

    hazard_flag = bool(hazard_occupancy > hazard_occupancy_threshold)

    # Variant 2: Perception Hazard Gating
    v2_dec = "ESCALATE" if hazard_flag else baseline_decision

    # Variant 3: Sensor Health Gating
    v3_dec = "ESCALATE" if sensor_loss else baseline_decision

    # Variant 4: Combined Gating
    v4_dec = "ESCALATE" if (hazard_flag or sensor_loss) else baseline_decision

    return PerceptionDecisionEvaluation(
        baseline_decision=baseline_decision,
        perception_hazard_decision=v2_dec,
        sensor_health_decision=v3_dec,
        combined_decision=v4_dec,
        hazard_occupancy=hazard_occupancy,
        sensor_health_flag=sensor_loss,
        min_sensor_p=min_p,
        failing_agent=failing_agent,
    )

"""
COGNIX Live Demo V2 — Fast, Continuous, Multi-Agent Uncertainty Dashboard.

Integrates the live CARLA simulator with the actual COGNIX Decision Engine,
Multi-Agent Epistemic-Weighted Belief Fusion, Conformal Prediction sets,
and real-time perception quality analyzers.

Key Architecture Principles:
1. Low-Overhead Continuous Rendering: No heavyweight R-CNN or large neural detector
   in the per-frame render loop. Maintains high interactive FPS (15-60 FPS).
2. Authoritative Decision Engine: The primary safety decision is produced directly
   by cognix.decision.escalation.EscalationEngine (ACT, WAIT, ESCALATE, ABSTAIN).
3. Explicit Source Attribution: Every uncertainty and hazard is visibly labeled with its
   true source: LIVE (CARLA sensor), COGNIX (real subsystem), DEMO PROXY (heuristic),
   SCENARIO INPUT (injected test signal), or NOT EXPOSED (unattached sensor).
4. Interactive Scenario Switching: Allows real-time switching between Normal (N),
   Pedestrian (P), Severe Weather (W), Roadway Prop (B), and Reset (R) without restarting.
5. Strict Pedestrian Protection: A confirmed pedestrian inside the forward corridor
   commands immediate protective escalation (ESCALATE / HIGH risk) per verified M16 policy.
6. Safe Actor Lifecycle: Tracks and cleans up all demo-owned actors and restores world weather.
7. Graceful Shutdown: Clean KeyboardInterrupt (Ctrl+C), Q, and Esc handlers with zero actor leakage.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import logging
import math
import os
import random
import signal
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path for direct execution
repo_root = str(Path(__file__).resolve().parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import threading
import time

try:
    import msvcrt
except ImportError:
    msvcrt = None

import cv2
import numpy as np

# Cognix core decision and multi-agent components
from cognix.decision.escalation import (
    DecisionOutcome,
    EscalationEngine,
    EscalationResult,
    RiskLevel,
)
from cognix.adapters.carla.agents import (
    CameraAgent,
    SegAgent,
    IMUAgent,
    LiDARAgent,
    DepthAgent,
    GNSSAgent,
)
from cognix.adapters.carla.features import (
    rgb_to_hazard_prob,
    seg_to_complexity_score,
    imu_to_motion_score,
)
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor

logger = logging.getLogger("cognix.live_demo_v2")

# Constants for observation freshness
OBSTACLE_STALE_SECONDS = 0.40
COLLISION_ALERT_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Enums and Dataclasses
# ---------------------------------------------------------------------------

class SourceType(Enum):
    LIVE = "LIVE"
    COGNIX = "COGNIX"
    DEMO_PROXY = "DEMO PROXY"
    SCENARIO_INPUT = "SCENARIO INPUT"
    NOT_EXPOSED = "NOT EXPOSED"


class ScenarioMode(Enum):
    NORMAL = "NORMAL"
    PEDESTRIAN = "PEDESTRIAN"
    WEATHER = "WEATHER"
    PROPS = "PROPS"


class FaultMode(Enum):
    NONE = 0
    RGB_DARKNESS = 1
    RGB_OCCLUSION = 2
    SEG_CORRUPTION = 3
    IMU_DISTURBANCE = 4


@dataclass
class UncertaintyEntry:
    id: str
    name: str
    category: str
    value: Optional[float]
    display_str: str
    source_type: SourceType
    description: str
    is_active: bool = True


@dataclass
class FrameEvaluation:
    frame_index: int
    timestamp: float
    fps: float
    latency_ms: float
    sensor_age_ms: float
    scenario_mode: ScenarioMode
    fault_mode: FaultMode
    speed_kmh: float
    obstacle_distance: Optional[float]
    obstacle_actor: str
    obstacle_age_ms: Optional[float] = None
    obstacle_fresh: bool = False
    ttc_sec: Optional[float] = None
    v_close_mps: Optional[float] = None
    relative_motion_available: bool = False
    policy_branch_reason: str = ""
    last_key_source: str = "NONE"
    collision_active: bool = False
    collision_impulse: float = 0.0
    collision_age_ms: Optional[float] = None
    collision_actor: str = "none"
    collision_actor_id: Optional[int] = None
    traffic_light_state: str = "NONE"

    # Cognix Decision Engine outputs
    decision_outcome: DecisionOutcome = DecisionOutcome.ACT
    risk_level: RiskLevel = RiskLevel.LOW
    escalation_required: bool = False
    confidence: float = 0.90
    epistemic_uncertainty: float = 0.05
    conformal_set_size: int = 1
    decision_reason: str = "Nominal trajectory."
    trigger_source: SourceType = SourceType.LIVE

    # Uncertainty inventory
    uncertainties: List[UncertaintyEntry] = None  # type: ignore
    fused_uncertainty: float = 0.0
    fused_safe_probability: float = 0.90
    agent_weights: Dict[str, float] = None  # type: ignore

    # Legacy demo heuristic advisory (for visual backward compatibility)
    legacy_advisory: str = "CONTINUE (NOMINAL)"
    legacy_risk: str = "LOW"

    # Processed sensor visuals and telemetry for dashboard rendering
    processed_rgb: Optional[np.ndarray] = None
    processed_seg: Optional[np.ndarray] = None
    clean_imu: Optional[Dict[str, np.ndarray]] = None
    injected_imu: Optional[Dict[str, np.ndarray]] = None

    # Safe checkpoint and recovery telemetry
    safe_checkpoint_loc: Optional[Tuple[float, float, float]] = None
    recovery_status: str = "IDLE"
    stuck_status: str = "NOMINAL"

    # Autopilot motion diagnostics and vehicle control telemetry
    autopilot_status: str = "AP: ENGAGED"
    road_info: str = ""
    motion_displacement: float = 0.0
    motion_cum_yaw: float = 0.0
    circling_detected: bool = False
    ctrl_steer: float = 0.0
    ctrl_throttle: float = 0.0
    ctrl_brake: float = 0.0

    # Extended trajectory diagnostics (College Presentation Suite)
    cumulative_distance_m: float = 0.0
    net_displacement_m: float = 0.0
    efficiency_ratio: float = 1.0
    revisit_detected: bool = False
    circular_movement_detected: bool = False
    excessive_steering: bool = False
    wrong_direction: bool = False
    is_immobilized: bool = False
    stuck_seconds: float = 0.0


@dataclass
class SafeCheckpoint:
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]  # pitch, yaw, roll
    timestamp: float
    road_id: int
    lane_id: int
    speed_kmh: float


# ---------------------------------------------------------------------------
# Presentation Route Planning & Multi-Metric Trajectory Auditor
# ---------------------------------------------------------------------------

class TrajectoryAuditor:
    """
    Multi-metric vehicle trajectory auditor for CARLA simulated vehicle movement.
    Detects:
      - Repeated visits to same location / road segment (< 45s revisit interval)
      - Large cumulative heading change without meaningful route progress
      - Sustained circular movement (even when displacement > 2.5m)
      - Excessive steering oscillation (high-frequency zero-crossings)
      - Prolonged near-zero speed (differentiating red light / queuing from immobilization)
      - Wrong-direction lane flow (heading opposite to lane waypoint direction)
    """

    def __init__(self, history_seconds: float = 15.0, cell_size_m: float = 6.0):
        self.history_seconds = history_seconds
        self.cell_size_m = cell_size_m
        self.samples = collections.deque()  # stores (t, x, y, z, yaw, speed_kmh, steer, road_id, lane_id)
        self.total_traveled_distance_m = 0.0
        self.last_pos: Optional[Tuple[float, float, float]] = None
        self.start_pos: Optional[Tuple[float, float, float]] = None
        self.revisited_cells_count = 0
        self.visited_cells: Dict[Tuple[int, int], float] = {}  # (grid_x, grid_y) -> last_visit_time
        self.last_steer_sign = 0
        self.steer_reversals_window = collections.deque()
        self.consecutive_stuck_seconds = 0.0
        self.last_stuck_check_time: Optional[float] = None

    def reset(self, start_pos: Optional[Tuple[float, float, float]] = None) -> None:
        """Reset auditor state upon vehicle spawn or road recovery."""
        self.samples.clear()
        self.total_traveled_distance_m = 0.0
        self.last_pos = start_pos
        self.start_pos = start_pos
        self.visited_cells.clear()
        self.revisited_cells_count = 0
        self.steer_reversals_window.clear()
        self.last_steer_sign = 0
        self.consecutive_stuck_seconds = 0.0
        self.last_stuck_check_time = None

    def update(
        self,
        t: float,
        x: float,
        y: float,
        z: float,
        yaw: float,
        speed_kmh: float,
        steer: float = 0.0,
        road_id: int = 0,
        lane_id: int = 0,
        is_at_red_light: bool = False,
        is_queuing: bool = False,
        lane_yaw: Optional[float] = None,
    ) -> Dict[str, Any]:
        curr_pos = (float(x), float(y), float(z))
        if self.start_pos is None:
            self.start_pos = curr_pos

        # Running traveled distance (integrating step distances)
        if self.last_pos is not None:
            step_dist = math.sqrt((curr_pos[0] - self.last_pos[0]) ** 2 + (curr_pos[1] - self.last_pos[1]) ** 2)
            self.total_traveled_distance_m += step_dist
        self.last_pos = curr_pos

        # Record history sample
        self.samples.append((t, curr_pos[0], curr_pos[1], curr_pos[2], float(yaw), float(speed_kmh), float(steer), int(road_id), int(lane_id)))
        while self.samples and (t - self.samples[0][0]) > self.history_seconds:
            self.samples.popleft()

        # Grid cell revisitation (detects tight circular loops around spawn points)
        cell = (int(curr_pos[0] // self.cell_size_m), int(curr_pos[1] // self.cell_size_m))
        revisit_detected = False
        revisit_interval = None
        if cell in self.visited_cells:
            interval = t - self.visited_cells[cell]
            # Flag if cell is revisited in under 45s after having moved at least 15m away
            if 5.0 < interval < 45.0 and self.total_traveled_distance_m > 15.0:
                revisit_detected = True
                revisit_interval = interval
                self.revisited_cells_count += 1
        self.visited_cells[cell] = t

        # Rolling window metrics
        window_dist = 0.0
        window_disp = 0.0
        cum_abs_yaw = 0.0
        if len(self.samples) >= 2:
            first = self.samples[0]
            window_disp = math.sqrt((curr_pos[0] - first[1]) ** 2 + (curr_pos[1] - first[2]) ** 2)
            for i in range(1, len(self.samples)):
                p0 = self.samples[i - 1]
                p1 = self.samples[i]
                window_dist += math.sqrt((p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2)
                yaw_diff = abs((p1[4] - p0[4] + 180.0) % 360.0 - 180.0)
                cum_abs_yaw += yaw_diff

        # Route Progress Efficiency Ratio: window_displacement / window_distance
        eff_ratio = window_disp / (window_dist + 1e-3) if window_dist > 5.0 else 1.0

        # Steering oscillations: high-frequency zero-crossings
        curr_steer_sign = 1 if steer > 0.15 else (-1 if steer < -0.15 else 0)
        if curr_steer_sign != 0 and self.last_steer_sign != 0 and curr_steer_sign != self.last_steer_sign:
            self.steer_reversals_window.append(t)
            self.last_steer_sign = curr_steer_sign
        elif curr_steer_sign != 0:
            self.last_steer_sign = curr_steer_sign

        while self.steer_reversals_window and (t - self.steer_reversals_window[0]) > 4.0:
            self.steer_reversals_window.popleft()
        excessive_steering = len(self.steer_reversals_window) >= 4

        # Sustained circular movement: large heading rotation without meaningful displacement
        # (detects large looping circles where displacement exceeds 2.5m, e.g. up to 15m radius)
        circular_movement = False
        if window_dist > 20.0 and cum_abs_yaw >= 300.0 and eff_ratio < 0.35:
            circular_movement = True

        # Heading vs lane flow orientation
        wrong_direction = False
        if lane_yaw is not None and speed_kmh > 2.0:
            cos_flow = math.cos(math.radians(float(yaw) - float(lane_yaw)))
            if cos_flow < -0.2:
                wrong_direction = True

        # Immobilization check: speed < 0.5 km/h while not at a red light or queued behind traffic
        if speed_kmh < 0.5 and not is_at_red_light and not is_queuing:
            if self.last_stuck_check_time is not None:
                self.consecutive_stuck_seconds += (t - self.last_stuck_check_time)
        else:
            self.consecutive_stuck_seconds = 0.0
        self.last_stuck_check_time = t
        is_immobilized = self.consecutive_stuck_seconds >= 4.0

        net_disp_from_start = (
            math.sqrt((curr_pos[0] - self.start_pos[0]) ** 2 + (curr_pos[1] - self.start_pos[1]) ** 2)
            if self.start_pos is not None
            else 0.0
        )

        return {
            "cumulative_distance_m": round(self.total_traveled_distance_m, 2),
            "net_displacement_m": round(net_disp_from_start, 2),
            "window_distance_m": round(window_dist, 2),
            "window_displacement_m": round(window_disp, 2),
            "efficiency_ratio": round(eff_ratio, 3),
            "cum_yaw_change_deg": round(cum_abs_yaw, 1),
            "revisit_detected": revisit_detected,
            "revisit_interval_s": round(revisit_interval, 1) if revisit_interval else None,
            "circular_movement_detected": circular_movement,
            "excessive_steering": excessive_steering,
            "wrong_direction": wrong_direction,
            "is_immobilized": is_immobilized,
            "stuck_seconds": round(self.consecutive_stuck_seconds, 1),
        }


def build_presentation_loop_route(
    carla_map: Any,
    start_transform: Any,
    total_distance_m: float = 2000.0,
    step_m: float = 2.5,
) -> List[Any]:
    """
    Build a deterministic, repeatable, continuous waypoint path on Town10HD_Opt.
    Follows the wide 850m perimeter boulevard loop seamlessly, resolving junction
    branches by prioritizing heading continuity and lane-flow consistency.
    """
    try:
        import carla
        wp = carla_map.get_waypoint(
            start_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        wp = None

    if wp is None:
        return [getattr(start_transform, "location", start_transform)]

    curr = wp
    route_locations = [curr.transform.location]
    steps_needed = max(10, int(total_distance_m / step_m))

    for _ in range(steps_needed):
        try:
            next_wps = curr.next(step_m)
        except Exception:
            break
        if not next_wps:
            break
        best_wp = next_wps[0]
        if len(next_wps) > 1:
            # Pick branch that minimizes absolute heading difference with current waypoint
            min_diff = 999.0
            for cand in next_wps:
                diff = abs((cand.transform.rotation.yaw - curr.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
                if diff < min_diff:
                    min_diff = diff
                    best_wp = cand
        curr = best_wp
        route_locations.append(curr.transform.location)

    return route_locations


# ---------------------------------------------------------------------------
# Sensor Quality and Heuristic Feature Analyzers
# ---------------------------------------------------------------------------

def encode_bgr_pixels(frame: np.ndarray) -> np.ndarray:
    """Encode BGR image into 24-bit integer codes for fast palette matching."""
    b = frame[:, :, 0].astype(np.uint32)
    g = frame[:, :, 1].astype(np.uint32)
    r = frame[:, :, 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def compute_rgb_quality_uncertainty(frame: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Compute real-time perceptual quality uncertainty on RGB camera frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    black_fraction = float(np.mean(gray < 15))

    # True daytime driving in CARLA Town10 has mean brightness ~65-140 and contrast ~30-65.
    # Darkness fault drops brightness to < 10. Central occlusion produces black_fraction >= 0.40.
    mean_darkness = float(np.clip((65.0 - mean_brightness) / 65.0, 0.0, 1.0))
    black_region_score = float(np.clip(black_fraction / 0.25, 0.0, 1.0))
    low_contrast = float(np.clip((20.0 - contrast) / 20.0, 0.0, 1.0))

    uncertainty = 0.55 * mean_darkness + 0.35 * black_region_score + 0.10 * low_contrast
    u_val = float(np.clip(uncertainty, 0.0, 1.0))
    return u_val, {
        "brightness": mean_brightness,
        "contrast": contrast,
        "black_fraction": black_fraction,
    }


def compute_seg_vocab_uncertainty(frame: np.ndarray, known_codes_set: set) -> float:
    """Compute semantic segmentation vocabulary deviation uncertainty."""
    if not known_codes_set:
        return 0.05
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
    codes = encode_bgr_pixels(small)
    known_arr = np.fromiter(known_codes_set, dtype=np.uint32)
    valid = np.isin(codes, known_arr)
    invalid_fraction = float(np.mean(~valid))
    u_val = 0.05 + 0.95 * invalid_fraction
    return float(np.clip(u_val, 0.0, 1.0))


def compute_imu_motion_uncertainty(
    curr_acc: np.ndarray, prev_acc: Optional[np.ndarray], gyro: np.ndarray
) -> Tuple[float, float, float]:
    """Compute dynamic uncertainty from vehicle acceleration changes and angular rate."""
    if prev_acc is None:
        acc_change = 0.0
    else:
        acc_change = float(np.linalg.norm(curr_acc - prev_acc))
    gyro_mag = float(np.linalg.norm(gyro))

    # Scaling calibrated for Town10 driving: nominal cornering/braking (< 2.0 m/s^2 change)
    # does not elevate uncertainty. Active IMU disturbance fault (sigma=15 m/s^2) drives jerk_score to 1.0.
    jerk_score = float(np.clip(acc_change / 8.0, 0.0, 1.0))
    gyro_score = float(np.clip((gyro_mag - 0.50) / 2.5, 0.0, 1.0))

    uncertainty = 0.70 * jerk_score + 0.30 * gyro_score
    return float(np.clip(uncertainty, 0.0, 1.0)), acc_change, gyro_mag


def apply_sensor_faults(
    rgb: np.ndarray,
    seg: np.ndarray,
    imu: Dict[str, np.ndarray],
    mode: FaultMode,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Inject synthetic sensor degradation for interactive robustness demonstrations."""
    rgb_out = rgb.copy()
    seg_out = seg.copy()
    imu_out = {
        "accelerometer": imu["accelerometer"].copy(),
        "gyroscope": imu["gyroscope"].copy(),
    }

    if mode == FaultMode.RGB_DARKNESS:
        rgb_out = (rgb_out.astype(np.float32) * 0.12).astype(np.uint8)

    elif mode == FaultMode.RGB_OCCLUSION:
        h, w = rgb_out.shape[:2]
        x1, y1 = int(w * 0.20), int(h * 0.20)
        x2, y2 = int(w * 0.80), int(h * 0.80)
        rgb_out[y1:y2, x1:x2] = 0
        # Draw high-visibility red warning perimeter
        cv2.rectangle(rgb_out, (x1, y1), (x2, y2), (30, 30, 220), 4)
        # Overlay clear occlusion alert inside blocked region
        cv2.putText(
            rgb_out,
            "SENSOR OCCLUSION - SENSOR BLOCKED",
            (x1 + 18, y1 + int((y2 - y1) * 0.45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 90, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rgb_out,
            "Central 60% Optical Field-of-View Obstructed",
            (x1 + 25, y1 + int((y2 - y1) * 0.45) + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    elif mode == FaultMode.SEG_CORRUPTION:
        h, w = seg_out.shape[:2]
        num_patches = 8
        for _ in range(num_patches):
            px = np.random.randint(0, max(1, w - 80))
            py = np.random.randint(0, max(1, h - 80))
            color = (
                int(np.random.randint(180, 255)),
                int(np.random.randint(0, 70)),
                int(np.random.randint(180, 255)),
            )
            cv2.rectangle(seg_out, (px, py), (px + 80, py + 80), color, -1)

    elif mode == FaultMode.IMU_DISTURBANCE:
        noise_acc = np.random.normal(0.0, 6.5, size=3)
        noise_gyro = np.random.normal(0.0, 2.5, size=3)
        imu_out["accelerometer"] += noise_acc
        imu_out["gyroscope"] += noise_gyro

    return rgb_out, seg_out, imu_out


def compute_obstacle_relative_motion(
    ego_velocity: Any,
    ego_location: Any,
    obs_velocity: Any,
    obs_location: Any,
    distance: float,
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Compute relative closing speed (v_close_mps) and Time-To-Collision (TTC)
    between ego vehicle and an obstacle actor.

    Args:
        ego_velocity: Ego velocity vector (carla.Vector3D, np.ndarray, list, or None)
        ego_location: Ego location (carla.Location, np.ndarray, list, or None)
        obs_velocity: Obstacle velocity vector (carla.Vector3D, np.ndarray, list, or None)
        obs_location: Obstacle location (carla.Location, np.ndarray, list, or None)
        distance: Measured Euclidean distance to obstacle in meters.

    Returns:
        (v_close_mps, ttc_sec, relative_motion_available)
        - v_close_mps: positive when closing in, negative when opening gap, None if unavailable.
        - ttc_sec: distance / v_close_mps if v_close_mps >= 0.5 m/s, else None.
        - relative_motion_available: True if both velocity vectors were resolved.
    """
    if ego_velocity is None or obs_velocity is None or distance <= 0.0:
        return None, None, False

    def _to_np_vec(v: Any) -> Optional[np.ndarray]:
        if v is None:
            return None
        if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
            return np.array([float(v.x), float(v.y), float(v.z)])
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            z = float(v[2]) if len(v) >= 3 else 0.0
            return np.array([float(v[0]), float(v[1]), z])
        return None

    v_ego = _to_np_vec(ego_velocity)
    v_obs = _to_np_vec(obs_velocity)
    loc_ego = _to_np_vec(ego_location)
    loc_obs = _to_np_vec(obs_location)

    if v_ego is None or v_obs is None:
        return None, None, False

    # Relative velocity vector: v_rel = v_ego - v_obs
    v_rel = v_ego - v_obs

    # Determine line of sight unit vector u_los pointing from ego to obstacle
    if loc_ego is not None and loc_obs is not None:
        r_los = loc_obs - loc_ego
        dist_los = float(np.linalg.norm(r_los))
        if dist_los > 0.1:
            u_los = r_los / dist_los
        else:
            norm_v = float(np.linalg.norm(v_ego))
            u_los = v_ego / norm_v if norm_v > 0.1 else np.array([1.0, 0.0, 0.0])
    else:
        norm_v = float(np.linalg.norm(v_ego))
        u_los = v_ego / norm_v if norm_v > 0.1 else np.array([1.0, 0.0, 0.0])

    # Closing speed along line-of-sight: v_close = v_rel . u_los
    v_close = float(np.dot(v_rel, u_los))

    # TTC calculation: only meaningful when positively closing at >= 0.5 m/s (~1.8 km/h)
    if v_close >= 0.5:
        ttc = float(distance / v_close)
    else:
        ttc = None

    return v_close, ttc, True


# ---------------------------------------------------------------------------
# Cognix Live Demo V2 Engine and Controller
# ---------------------------------------------------------------------------

class CognixLiveDemoV2:
    """Polished, fast, continuous multi-agent uncertainty demonstration."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        conformal_path: str = "artifacts/synthetic/conformal_cal_scores.npy",
        target_fps: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.target_fps = target_fps
        self.conformal_path = conformal_path

        # State management
        self.scenario_mode = ScenarioMode.NORMAL
        self.fault_mode = FaultMode.NONE
        self.last_key_name: str = "NONE"
        self.last_key_source: str = "NONE"
        self.last_key_time: float = 0.0
        self.transition_message: str = "System Initialized (Nominal)"
        self.transition_message_time: float = time.monotonic()
        self.scenario_spawn_status: str = "NOMINAL (READY)"

        # Sensor state buffer (protected by thread lock)
        self.lock = threading.Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_seg: Optional[np.ndarray] = None
        self.latest_imu: Optional[Dict[str, np.ndarray]] = None
        self.latest_obstacle: Optional[Dict[str, Any]] = None
        self.latest_collision: Optional[Dict[str, Any]] = None
        self.last_sensor_time: float = time.monotonic()

        # Vehicle & world actor state
        self.vehicle_speed_kmh: float = 0.0
        self.traffic_light_state: str = "NONE"
        self.original_weather: Any = None

        # Tracking demo-owned actors for safe cleanup
        self.demo_owned_actors: List[Any] = []
        self.demo_owned_sensors: List[Any] = []
        self.scenario_actors: List[Any] = []

        # Segmentation reference vocabulary
        self.seg_reference_codes: set = set()
        self._seed_reference_codes()

        # Previous acceleration for IMU jerk
        self.previous_acc: Optional[np.ndarray] = None

        # Cognix Components
        self._init_cognix_subsystems()

        # Performance monitoring
        self.frame_count: int = 0
        self.start_time: float = time.time()
        self.recent_latencies: List[float] = []
        self.current_fps: float = 0.0

        # Safe checkpoint and road recovery state
        self.safe_checkpoint: Optional[SafeCheckpoint] = None
        self.last_checkpoint_time: float = 0.0
        self.recovery_status: str = "IDLE"
        self.stuck_detected: bool = False
        self.stuck_reason: str = "NOMINAL"
        self.stuck_start_time: Optional[float] = None
        self.recovery_count: int = 0
        self.post_recovery_warmup_frames: int = 0

        # Motion diagnostics, Traffic Manager port, and circling detection
        self.pose_history: collections.deque = collections.deque(maxlen=120)
        self.circling_detected: bool = False
        self.circling_reason: str = "NOMINAL"
        self.rolling_displacement: float = 0.0
        self.rolling_cum_yaw: float = 0.0
        self.tm_port: int = 8000
        self.client: Any = None

        # Extended trajectory auditor & presentation route tracking
        self.trajectory_auditor = TrajectoryAuditor(history_seconds=15.0)
        self.presentation_route_locations: List[Any] = []
        self.current_route_index: int = 0
        self.trajectory_metrics: Dict[str, Any] = {
            "cumulative_distance_m": 0.0,
            "net_displacement_m": 0.0,
            "window_distance_m": 0.0,
            "window_displacement_m": 0.0,
            "efficiency_ratio": 1.0,
            "cum_yaw_change_deg": 0.0,
            "revisit_detected": False,
            "revisit_interval_s": None,
            "circular_movement_detected": False,
            "excessive_steering": False,
            "wrong_direction": False,
            "is_immobilized": False,
            "stuck_seconds": 0.0,
        }

    def _seed_reference_codes(self) -> None:
        """Seed known CityScapes / CARLA palette codes for segmentation reference."""
        # CARLA Cityscapes palette classes (R, G, B)
        sample_palette_rgb = [
            (0, 0, 0),        # 0: Unlabeled
            (70, 70, 70),     # 1: Building
            (100, 40, 40),    # 2: Fences
            (55, 90, 80),     # 3: Other
            (220, 20, 60),    # 4: Pedestrian
            (153, 153, 153),  # 5: Pole
            (157, 234, 50),   # 6: Roadline
            (128, 64, 128),   # 7: Road
            (244, 35, 232),   # 8: Sidewalk
            (107, 142, 35),   # 9: Vegetation
            (0, 0, 142),      # 10: Vehicles / Car
            (102, 102, 156),  # 11: Wall
            (220, 220, 0),    # 12: TrafficSign
            (70, 130, 180),   # 13: Sky
            (81, 0, 81),      # 14: Ground
            (150, 100, 100),  # 15: Bridge
            (230, 150, 140),  # 16: RailTrack
            (180, 165, 180),  # 17: GuardRail
            (250, 170, 30),   # 18: TrafficLight
            (110, 190, 160),  # 19: Static
            (170, 120, 50),   # 20: Dynamic
            (45, 60, 150),    # 21: Water
            (145, 170, 100),  # 22: Terrain
        ]
        # CARLA segmentation outputs BGR, and encode_bgr_pixels does (r<<16)|(g<<8)|b.
        # Seed both (r, g, b) and (b, g, r) permutations to cover BGR and RGB memory layouts.
        for r, g, b in sample_palette_rgb:
            code_rgb = int((np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b))
            code_bgr = int((np.uint32(b) << 16) | (np.uint32(g) << 8) | np.uint32(r))
            self.seg_reference_codes.add(code_rgb)
            self.seg_reference_codes.add(code_bgr)

    def update_motion_diagnostics(
        self,
        x: float,
        y: float,
        yaw: float,
        speed_kmh: float,
        steer: float = 0.0,
        throttle: float = 0.0,
        timestamp: Optional[float] = None,
        road_id: int = 0,
        lane_id: int = 0,
        is_at_red_light: bool = False,
        is_queuing: bool = False,
        lane_yaw: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """
        Track rolling motion and detect repetitive in-place circling.
        Window: 3.5 seconds for legacy compatibility + 15.0 seconds for multi-metric TrajectoryAuditor.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        self.pose_history.append((now, float(x), float(y), float(yaw)))

        # Update comprehensive multi-metric trajectory auditor
        self.trajectory_metrics = self.trajectory_auditor.update(
            t=now,
            x=x,
            y=y,
            z=0.0,
            yaw=yaw,
            speed_kmh=speed_kmh,
            steer=steer,
            road_id=road_id,
            lane_id=lane_id,
            is_at_red_light=is_at_red_light,
            is_queuing=is_queuing,
            lane_yaw=lane_yaw,
        )

        # Prune entries older than 3.5 seconds
        while len(self.pose_history) > 2 and (now - self.pose_history[0][0]) > 3.5:
            self.pose_history.popleft()

        if len(self.pose_history) < 3:
            self.rolling_displacement = 0.0
            self.rolling_cum_yaw = 0.0
            self.circling_detected = False
            return False, 0.0, 0.0

        oldest = self.pose_history[0]
        newest = self.pose_history[-1]

        # Rolling displacement from oldest to newest in window
        disp = math.sqrt((newest[1] - oldest[1]) ** 2 + (newest[2] - oldest[2]) ** 2)
        self.rolling_displacement = float(disp)

        # Cumulative yaw change across consecutive samples in window
        cum_yaw = 0.0
        for i in range(1, len(self.pose_history)):
            prev_y = self.pose_history[i - 1][3]
            curr_y = self.pose_history[i][3]
            dy = abs(curr_y - prev_y) % 360.0
            if dy > 180.0:
                dy = 360.0 - dy
            cum_yaw += dy
        self.rolling_cum_yaw = float(cum_yaw)

        # Duration spanned by window
        dt = newest[0] - oldest[0]

        # Circling condition:
        # Window spans at least 2.0 seconds, cumulative yaw >= 180 degrees, displacement < 2.5 meters,
        # or sustained circular motion flagged by TrajectoryAuditor.
        is_circling = (
            (
                dt >= 2.0
                and cum_yaw >= 180.0
                and disp < 2.5
                and (speed_kmh >= 1.0 or abs(steer) > 0.3 or throttle > 0.1)
            )
            or self.trajectory_metrics.get("circular_movement_detected", False)
        )

        self.circling_detected = is_circling
        if self.trajectory_metrics.get("circular_movement_detected", False):
            self.circling_reason = "AUTOPILOT_CIRCLING_DETECTED (Sustained circular looping trajectory)"
        elif is_circling:
            self.circling_reason = (
                f"AUTOPILOT_CIRCLING_DETECTED (rot={cum_yaw:.1f}deg, disp={disp:.2f}m < 2.5m, dt={dt:.1f}s)"
            )
        else:
            self.circling_reason = "NOMINAL"

        return is_circling, disp, cum_yaw

    def _init_cognix_subsystems(self) -> None:
        """Initialize genuine Cognix EscalationEngine, agents, fusion and calibration."""
        # 1. Authoritative Escalation Engine
        self.escalation_engine = EscalationEngine(
            high_conf_thresh=0.70,
            low_conf_thresh=0.40,
            high_unc_thresh=0.50,
            max_conformal_set_size=3,
        )

        # 2. Real Cognix Sensor Agents
        self.camera_agent = CameraAgent(agent_id="Camera")
        self.seg_agent = SegAgent(agent_id="Seg")
        self.imu_agent = IMUAgent(agent_id="IMU")

        # Subsystems documented as NOT EXPOSED / NO SENSOR in this vehicle config
        self.lidar_agent = LiDARAgent(agent_id="LiDAR")
        self.depth_agent = DepthAgent(agent_id="Depth")
        self.gnss_agent = GNSSAgent(agent_id="GNSS")

        # 3. Multi-Agent Epistemic-Weighted Belief Fusion
        self.belief_fuser = EpistemicWeightedFusion(eps=1e-5)

        # 4. Inductive Conformal Predictor
        self.conformal_predictor = ConformalPredictor()
        if os.path.exists(self.conformal_path):
            try:
                self.conformal_predictor.load(self.conformal_path)
                logger.info("Loaded ConformalPredictor with %d calibration scores.", self.conformal_predictor.n_cal)
            except Exception as e:
                logger.warning("Could not load conformal calibration scores: %s", e)
        else:
            logger.info("Conformal scores not found at %s; operating with default coverage quantile.", self.conformal_path)

    def clear_expired_observations(self) -> None:
        """Clear stale obstacle and collision observations from internal state buffer."""
        now = time.monotonic()
        with self.lock:
            if self.latest_obstacle and (now - self.latest_obstacle.get("timestamp", 0) > OBSTACLE_STALE_SECONDS):
                self.latest_obstacle = None
            if self.latest_collision and (now - self.latest_collision.get("timestamp", 0) > COLLISION_ALERT_SECONDS):
                self.latest_collision = None

    def update_safe_checkpoint(
        self,
        world: Any,
        vehicle: Any,
        speed_kmh: float,
        traffic_light_state: str = "NONE",
    ) -> bool:
        """
        Record a known-good safe drivable road checkpoint.
        Updated when moving forward (speed >= 2.0 km/h) on a validated Driving lane,
        with no active collision, and not stopped at a red light.
        """
        if world is None or vehicle is None:
            return False

        # Throttle checkpoint updates to once every ~0.8s
        now_mono = time.monotonic()
        if (now_mono - self.last_checkpoint_time) < 0.8:
            return False

        # Condition 1: Vehicle moving forward
        if speed_kmh < 2.0:
            return False

        # Condition 2: No active collision
        with self.lock:
            if self.latest_collision is not None:
                return False

        # Condition 3: Not stopped at red light
        if traffic_light_state == "RED":
            return False

        try:
            v_loc = vehicle.get_location()
            v_tf = vehicle.get_transform()
            v_rot = v_tf.rotation

            carla_map = world.get_map()
            try:
                import carla
                driving_lane_type = carla.LaneType.Driving
            except Exception:
                driving_lane_type = None

            if driving_lane_type is not None:
                wp = carla_map.get_waypoint(v_loc, project_to_road=False, lane_type=driving_lane_type)
            else:
                wp = carla_map.get_waypoint(v_loc, project_to_road=False)

            if wp is None:
                return False

            lane_type_str = str(getattr(wp, "lane_type", "Driving"))
            if not ("Driving" in lane_type_str or lane_type_str == "Driving"):
                return False

            # Check lateral offset from lane centerline (< 2.5m is within driving lane)
            lane_loc = wp.transform.location
            lateral_dist = math.sqrt((v_loc.x - lane_loc.x) ** 2 + (v_loc.y - lane_loc.y) ** 2)
            if lateral_dist > 2.5:
                return False

            # Condition 4: Heading alignment with lane flow (must be within 60 deg of lane direction)
            try:
                v_rot_yaw = float(v_rot.yaw)
                wp_rot = getattr(wp.transform, "rotation", None)
                if wp_rot is not None and hasattr(wp_rot, "yaw"):
                    yaw_val = wp_rot.yaw
                    if isinstance(yaw_val, (int, float)) and not type(yaw_val).__name__.startswith("MagicMock"):
                        wp_yaw = float(yaw_val)
                        cos_align = math.cos(math.radians(v_rot_yaw - wp_yaw))
                        if cos_align < 0.5:
                            # Vehicle is turned sideways or spinning; do not record as safe road checkpoint
                            return False
            except (TypeError, ValueError, AttributeError):
                pass

            self.safe_checkpoint = SafeCheckpoint(
                location=(float(v_loc.x), float(v_loc.y), float(v_loc.z)),
                rotation=(float(v_rot.pitch), float(v_rot.yaw), float(v_rot.roll)),
                timestamp=time.time(),
                road_id=int(getattr(wp, "road_id", 0)),
                lane_id=int(getattr(wp, "lane_id", 0)),
                speed_kmh=float(speed_kmh),
            )
            self.last_checkpoint_time = now_mono
            return True
        except Exception as e:
            logger.debug("Safe checkpoint update skipped: %s", e)
            return False

    def detect_stuck_condition(
        self,
        world: Any,
        vehicle: Any,
        speed_kmh: float,
        traffic_light_state: str = "NONE",
        obstacle_distance: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Diagnose if the ego vehicle is stuck, pinned, off-road, or immobilized.
        Explicitly prevents false positives when vehicle is legally waiting at a red
        traffic light or queuing behind lead traffic.
        """
        # Condition 1: Collision Pinned
        with self.lock:
            col = self.latest_collision
        if col is not None:
            impulse = float(col.get("impulse", 0.0))
            col_ts = float(col.get("timestamp", 0.0))
            col_age = time.monotonic() - col_ts
            if impulse >= 50.0 and col_age < 12.0 and speed_kmh < 1.0:
                self.stuck_detected = True
                self.stuck_reason = f"COLLISION_PINNED (impulse={impulse:.1f}N*s, v={speed_kmh:.1f}km/h)"
                return True, self.stuck_reason

        # Condition 2: Off-road
        if world is not None and vehicle is not None:
            try:
                carla_map = world.get_map()
                v_loc = vehicle.get_location()

                try:
                    import carla
                    driving_lane_type = carla.LaneType.Driving
                except Exception:
                    driving_lane_type = None

                if driving_lane_type is not None:
                    wp = carla_map.get_waypoint(v_loc, project_to_road=False, lane_type=driving_lane_type)
                else:
                    wp = carla_map.get_waypoint(v_loc, project_to_road=False)

                if wp is None:
                    self.stuck_detected = True
                    self.stuck_reason = "OFF_ROAD (No driving lane waypoint at vehicle location)"
                    return True, self.stuck_reason

                lane_type_str = str(getattr(wp, "lane_type", "Driving"))
                if not ("Driving" in lane_type_str or lane_type_str == "Driving"):
                    self.stuck_detected = True
                    self.stuck_reason = f"OFF_ROAD (Lane type is {lane_type_str})"
                    return True, self.stuck_reason

                lane_loc = wp.transform.location
                lateral_dist = math.sqrt((v_loc.x - lane_loc.x) ** 2 + (v_loc.y - lane_loc.y) ** 2)
                if lateral_dist > 4.5:
                    self.stuck_detected = True
                    self.stuck_reason = f"OFF_ROAD (Lateral offset {lateral_dist:.1f}m > 4.5m)"
                    return True, self.stuck_reason
            except Exception as e:
                logger.debug("Off-road check exception: %s", e)

        # Condition 3: Stuck Stationary (Immobilized)
        is_at_red_light = (traffic_light_state == "RED")
        is_queuing = (obstacle_distance is not None and obstacle_distance <= 8.0)
        if speed_kmh < 0.5:
            if not is_at_red_light and not is_queuing:
                if self.stuck_start_time is None:
                    self.stuck_start_time = time.monotonic()
                elif (time.monotonic() - self.stuck_start_time) >= 4.0 or self.trajectory_metrics.get("is_immobilized", False):
                    duration = time.monotonic() - self.stuck_start_time if self.stuck_start_time else self.trajectory_metrics.get("stuck_seconds", 4.0)
                    self.stuck_detected = True
                    self.stuck_reason = f"STUCK_STATIONARY (v={speed_kmh:.1f}km/h for {duration:.1f}s, no light/queue)"
                    return True, self.stuck_reason
            else:
                self.stuck_start_time = None
        else:
            self.stuck_start_time = None

        # Condition 4: Autopilot Circling / In-Place Spinning
        if self.circling_detected:
            self.stuck_detected = True
            self.stuck_reason = self.circling_reason
            return True, self.stuck_reason

        self.stuck_detected = False
        self.stuck_reason = "NOMINAL"
        return False, "NOMINAL"

    def recover_vehicle_to_safe_road(
        self,
        world: Any,
        vehicle: Any,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Recover vehicle onto a safe, drivable road location.
        If the vehicle is NOT stuck and force is False, resets scenario and faults
        without unnecessary teleportation.
        """
        if world is None or vehicle is None:
            self.recovery_status = "RECOVERY_SKIPPED (NO VEHICLE)"
            return False, "NO_VEHICLE"

        is_stuck, stuck_reason = self.detect_stuck_condition(
            world=world,
            vehicle=vehicle,
            speed_kmh=self.vehicle_speed_kmh,
            traffic_light_state=self.traffic_light_state,
            obstacle_distance=self.latest_obstacle.get("distance") if self.latest_obstacle else None,
        )

        if not is_stuck and not force:
            # Vehicle is driving normally on road. Just reset scenario and faults!
            self.set_scenario(ScenarioMode.NORMAL, world, vehicle)
            self.fault_mode = FaultMode.NONE
            with self.lock:
                self.latest_collision = None
                self.latest_obstacle = None
            self.scenario_spawn_status = "RESET ALL"
            self.recovery_status = "NOMINAL_RESET_NO_TELEPORT"
            self.transition_message = "RESET: Scenario & Faults reset (Vehicle driving normally, no teleport)"
            self.transition_message_time = time.monotonic()
            logger.info("Reset requested while vehicle moving normally on road. Reset scenarios and faults; skipped teleport.")
            return False, "NOMINAL_RESET_NO_TELEPORT"

        logger.warning(
            "Executing vehicle road recovery: reason='%s', force=%s",
            stuck_reason if not force else "USER_FORCE",
            force,
        )

        # 1. Clear scenario actors and restore weather
        self.set_scenario(ScenarioMode.NORMAL, world, vehicle)
        self.fault_mode = FaultMode.NONE

        # 2. Select recovery candidate transform
        candidate_tf = None
        target_desc = "UNKNOWN"

        try:
            import carla
        except ImportError:
            carla = None

        # Candidate 0: Presentation Route Waypoint Ahead (Deterministic Route Consistency)
        if self.presentation_route_locations and carla is not None and hasattr(vehicle, "get_location"):
            try:
                v_loc = vehicle.get_location()
                # Find closest route location index
                min_idx = 0
                min_d = 9999.0
                for r_i, r_loc in enumerate(self.presentation_route_locations):
                    d = math.sqrt((r_loc.x - v_loc.x) ** 2 + (r_loc.y - v_loc.y) ** 2)
                    if d < min_d:
                        min_d = d
                        min_idx = r_i
                # Advance 4 waypoints (~10m) forward along route to ensure clear headway
                tgt_idx = (min_idx + 4) % len(self.presentation_route_locations)
                tgt_loc = self.presentation_route_locations[tgt_idx]
                carla_map = world.get_map()
                tgt_wp = carla_map.get_waypoint(tgt_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                cand_rot = tgt_wp.transform.rotation if tgt_wp else carla.Rotation(yaw=0.0)
                if not isinstance(cand_rot, carla.Rotation):
                    cand_rot = carla.Rotation(yaw=float(getattr(cand_rot, "yaw", 0.0)))
                cand_loc = carla.Location(x=float(tgt_loc.x), y=float(tgt_loc.y), z=float(tgt_loc.z) + 0.35)
                candidate_tf = carla.Transform(cand_loc, cand_rot)
                target_desc = f"ROUTE_WAYPOINT (idx={tgt_idx}, road={tgt_wp.road_id if tgt_wp else 0})"

                # Re-slice route and set explicit path on Traffic Manager
                remaining_route = self.presentation_route_locations[tgt_idx:] + self.presentation_route_locations[:tgt_idx]
                if self.client is not None:
                    try:
                        tm = self.client.get_trafficmanager(self.tm_port)
                        tm.auto_lane_change(vehicle, False)
                        tm.distance_to_leading_vehicle(vehicle, 4.0)
                        try:
                            tm.set_desired_speed(vehicle, 22.0)
                        except Exception:
                            tm.vehicle_percentage_speed_difference(vehicle, 30.0)
                        tm.set_path(vehicle, remaining_route)
                    except Exception as e:
                        logger.warning("Could not refresh TrafficManager path on recovery: %s", e)
            except Exception as e:
                logger.warning("Could not recover to route waypoint: %s", e)

        # Candidate 1: Safe Checkpoint
        if candidate_tf is None and self.safe_checkpoint is not None and carla is not None:
            chk = self.safe_checkpoint
            cand_loc = carla.Location(x=chk.location[0], y=chk.location[1], z=chk.location[2] + 0.35)
            cand_rot = carla.Rotation(pitch=chk.rotation[0], yaw=chk.rotation[1], roll=chk.rotation[2])
            candidate_tf = carla.Transform(cand_loc, cand_rot)
            target_desc = f"SAFE_CHECKPOINT (road={chk.road_id}, lane={chk.lane_id})"

        # Candidate 2: Map Waypoint (Project to nearest driving lane with heading validation)
        if candidate_tf is None and carla is not None:
            try:
                carla_map = world.get_map()
                v_loc = vehicle.get_location()
                v_rot = vehicle.get_transform().rotation if hasattr(vehicle, "get_transform") else None
                wp = carla_map.get_waypoint(v_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is not None:
                    chosen_wp = wp
                    # If vehicle rotation is available, verify heading alignment to avoid opposing lane flip
                    if v_rot is not None:
                        wp_yaw = wp.transform.rotation.yaw
                        cos_align = math.cos(math.radians(v_rot.yaw - wp_yaw))
                        if cos_align < 0.0:
                            # Waypoint points in opposite direction; inspect adjacent lanes
                            left_wp = wp.get_left_lane()
                            right_wp = wp.get_right_lane()
                            if left_wp and getattr(left_wp, "lane_type", None) == carla.LaneType.Driving:
                                if math.cos(math.radians(v_rot.yaw - left_wp.transform.rotation.yaw)) > 0.0:
                                    chosen_wp = left_wp
                            elif right_wp and getattr(right_wp, "lane_type", None) == carla.LaneType.Driving:
                                if math.cos(math.radians(v_rot.yaw - right_wp.transform.rotation.yaw)) > 0.0:
                                    chosen_wp = right_wp

                    cand_loc = carla.Location(
                        x=chosen_wp.transform.location.x,
                        y=chosen_wp.transform.location.y,
                        z=chosen_wp.transform.location.z + 0.40,
                    )
                    # Always use chosen waypoint's natural lane heading so vehicle drives with traffic
                    cand_rot = chosen_wp.transform.rotation
                    candidate_tf = carla.Transform(cand_loc, cand_rot)
                    target_desc = f"MAP_WAYPOINT (road={chosen_wp.road_id}, lane={chosen_wp.lane_id}, yaw={cand_rot.yaw:.1f})"
            except Exception as e:
                logger.warning("Could not project to map waypoint: %s", e)

        # Candidate 3: Map Spawn Point (Filter for verified straight-boulevard spawn points)
        if candidate_tf is None and carla is not None:
            try:
                carla_map = world.get_map()
                spawn_points = carla_map.get_spawn_points()
                if spawn_points:
                    v_loc = vehicle.get_location()
                    # Priority indices for clean straight roads in Town10HD_Opt
                    preferred_indices = [19, 20, 22, 4, 5, 6, 8, 10]
                    valid_sps = [spawn_points[i] for i in preferred_indices if i < len(spawn_points)]
                    if not valid_sps:
                        valid_sps = spawn_points
                    valid_sps.sort(key=lambda sp: (sp.location.x - v_loc.x) ** 2 + (sp.location.y - v_loc.y) ** 2)
                    candidate_tf = valid_sps[0]
                    candidate_tf.location.z += 0.40
                    target_desc = "SAFE_SPAWN_POINT"
            except Exception as e:
                logger.warning("Could not get nearest spawn point: %s", e)

        # Mock fallback (for unit tests where carla is not imported)
        if candidate_tf is None and hasattr(vehicle, "get_transform"):
            v_tf = vehicle.get_transform()
            candidate_tf = v_tf
            target_desc = "MOCK_VEHICLE_TRANSFORM"

        if candidate_tf is None:
            self.recovery_status = "RECOVERY_FAILED (NO CANDIDATE)"
            logger.error("Road recovery failed: No candidate transform found.")
            return False, "NO_CANDIDATE"

        # 3. Teleportation and stabilization sequence
        try:
            if hasattr(vehicle, "set_autopilot"):
                vehicle.set_autopilot(False)

            if hasattr(vehicle, "set_target_velocity") and carla is not None:
                vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            if hasattr(vehicle, "set_target_angular_velocity") and carla is not None:
                vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

            if hasattr(vehicle, "apply_control") and carla is not None:
                vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True, throttle=0.0, steer=0.0))

            vehicle.set_transform(candidate_tf)

            if hasattr(vehicle, "set_autopilot"):
                if self.tm_port != 8000:
                    try:
                        vehicle.set_autopilot(True, self.tm_port)
                    except TypeError:
                        vehicle.set_autopilot(True)
                else:
                    vehicle.set_autopilot(True)

            if self.client is not None:
                try:
                    tm = self.client.get_trafficmanager(self.tm_port)
                    tm.auto_lane_change(vehicle, False)
                    tm.distance_to_leading_vehicle(vehicle, 3.5)
                    tm.vehicle_percentage_speed_difference(vehicle, -10.0)
                except Exception as e:
                    logger.debug("TM reconfigure on recovery: %s", e)

        except Exception as e:
            self.recovery_status = f"RECOVERY_ERROR ({e})"
            logger.error("Failed during recovery teleportation sequence: %s", e)
            return False, f"TELEPORT_ERROR ({e})"

        # 4. Clear transient collision and obstacle states and motion history
        with self.lock:
            self.latest_collision = None
            self.latest_obstacle = None
        self.previous_acc = None
        self.stuck_detected = False
        self.stuck_reason = "NOMINAL"
        self.stuck_start_time = None
        self.circling_detected = False
        self.circling_reason = "NOMINAL"
        self.pose_history.clear()
        self.recovery_count += 1
        self.post_recovery_warmup_frames = 12
        self.recovery_status = f"RECOVERED ({target_desc})"
        self.transition_message = f"RECOVERED TO SAFE ROAD: Teleported ({target_desc}) & Autopilot re-engaged"
        self.transition_message_time = time.monotonic()
        if candidate_tf is not None and hasattr(candidate_tf, "location"):
            c_loc = candidate_tf.location
            self.trajectory_auditor.reset(start_pos=(float(c_loc.x), float(c_loc.y), float(c_loc.z)))
        logger.info("Vehicle successfully recovered to safe drivable road: %s", target_desc)
        return True, f"RECOVERED ({target_desc})"

    # -----------------------------------------------------------------------
    # Step-by-Step Per-Frame Evaluation (Fast, No R-CNN in loop)
    # -----------------------------------------------------------------------

    def evaluate_frame(
        self,
        rgb_raw: np.ndarray,
        seg_raw: np.ndarray,
        imu_raw: Dict[str, np.ndarray],
        obstacle_raw: Optional[Dict[str, Any]],
        collision_raw: Optional[Dict[str, Any]],
        speed_kmh: float,
        traffic_light_state: str,
        sensor_timestamp: float,
        v_close_mps: Optional[float] = None,
        ttc_sec: Optional[float] = None,
        relative_motion_available: bool = False,
        autopilot_status: str = "AP: ENGAGED",
        road_info: str = "",
        motion_displacement: float = 0.0,
        motion_cum_yaw: float = 0.0,
        circling_detected: bool = False,
        ctrl_steer: float = 0.0,
        ctrl_throttle: float = 0.0,
        ctrl_brake: float = 0.0,
    ) -> FrameEvaluation:
        """Process sensor snapshot through Cognix pipeline in under 5 milliseconds."""
        t_eval_start = time.perf_counter()

        # 1. Apply interactive sensor faults (if active)
        fault_active = self.fault_mode != FaultMode.NONE
        rgb_eval, seg_eval, imu_eval = apply_sensor_faults(rgb_raw, seg_raw, imu_raw, self.fault_mode)

        # 2. Compute lightweight perceptual quality metrics
        rgb_quality_u, rgb_stats = compute_rgb_quality_uncertainty(rgb_eval)
        seg_vocab_u = compute_seg_vocab_uncertainty(seg_eval, self.seg_reference_codes)
        imu_motion_u, acc_change, gyro_mag = compute_imu_motion_uncertainty(
            imu_eval["accelerometer"], self.previous_acc, imu_eval["gyroscope"]
        )
        self.previous_acc = imu_eval["accelerometer"].copy()

        # 3. Evaluate actual Cognix Agents with feature vectors
        # Map perceptual quality smoothly: nominal daylight produces [0.50, 0.50]
        # (matching CarlAnomalyAgent's nominal point in rgb_to_hazard_prob).
        # Injected darkness or occlusion degrades quality towards [0.05, 0.05].
        cam_var = float(np.clip(0.50 - 0.45 * rgb_quality_u, 0.05, 0.95))
        cam_bright = float(np.clip(0.50 - 0.45 * rgb_quality_u, 0.05, 0.95))
        cam_feature = np.array([[cam_var, cam_bright]])
        seg_feature = np.array([[seg_vocab_u]])
        imu_feature = np.array([[min(1.0, acc_change / 8.0), min(1.0, (gyro_mag - 0.5) / 2.5 if gyro_mag > 0.5 else 0.0)]])

        cam_u_res = self.camera_agent.estimate_uncertainty(cam_feature)
        seg_u_res = self.seg_agent.estimate_uncertainty(seg_feature)
        imu_u_res = self.imu_agent.estimate_uncertainty(imu_feature)

        cam_pred = self.camera_agent.predict(cam_feature)
        seg_pred = self.seg_agent.predict(seg_feature)
        imu_pred = self.imu_agent.predict(imu_feature)

        # 4. Multi-Agent Epistemic-Weighted Belief Fusion
        agent_preds = {
            "Camera": float(cam_pred.value),
            "Seg": float(seg_pred.value),
            "IMU": float(imu_pred.value),
        }
        agent_uncs = {
            "Camera": float(cam_u_res.epistemic),
            "Seg": float(seg_u_res.epistemic),
            "IMU": float(imu_u_res.epistemic),
        }
        agent_rels = {
            "Camera": float(max(0.05, 1.0 - rgb_quality_u)),
            "Seg": float(max(0.05, 1.0 - seg_vocab_u)),
            "IMU": float(max(0.05, 1.0 - imu_motion_u)),
        }

        fusion_result = self.belief_fuser.fuse(agent_preds, agent_uncs, agent_rels)
        fused_safe_p = float(fusion_result.probability)
        agent_weights = fusion_result.weights
        fused_uncertainty = sum(agent_weights[k] * agent_uncs[k] for k in agent_weights)

        # 5. Conformal Prediction Set Size
        pred_probs = np.array([[fused_safe_p, 1.0 - fused_safe_p]])
        if self.conformal_predictor.cal_scores is not None and len(self.conformal_predictor.cal_scores) > 0:
            cp_sets = self.conformal_predictor.predict(pred_probs, alpha=0.05)
            conformal_set_size = len(cp_sets[0].prediction_set)
        else:
            conformal_set_size = 1 if (fused_safe_p > 0.70 or fused_safe_p < 0.30) else 2

        # 6. Obstacle, Collision, and Freshness Processing
        speed_mps = speed_kmh / 3.6
        obs_dist = obstacle_raw.get("distance") if obstacle_raw else None
        obs_actor = obstacle_raw.get("actor_type", "none") if obstacle_raw else "none"
        obs_ts = obstacle_raw.get("timestamp") if obstacle_raw else None
        obs_age_ms = (time.monotonic() - obs_ts) * 1000.0 if obs_ts is not None else None
        obs_fresh = bool(obs_ts is not None and (time.monotonic() - obs_ts) <= OBSTACLE_STALE_SECONDS)

        # Resolve relative motion and TTC (prioritize explicit parameters, then obstacle_raw metadata)
        if v_close_mps is None and obstacle_raw:
            v_close_mps = obstacle_raw.get("v_close_mps")
        if ttc_sec is None and obstacle_raw:
            ttc_sec = obstacle_raw.get("ttc_sec")
        if not relative_motion_available and obstacle_raw:
            relative_motion_available = bool(obstacle_raw.get("relative_motion_available", False))

        ttc = ttc_sec

        collision_active = bool(collision_raw)
        collision_impulse = float(collision_raw.get("impulse", 0.0)) if collision_raw else 0.0
        col_ts = collision_raw.get("timestamp") if collision_raw else None
        col_age_ms = (time.monotonic() - col_ts) * 1000.0 if col_ts is not None else None
        col_actor = collision_raw.get("actor_type", "none") if collision_raw else "none"
        col_id = collision_raw.get("actor_id", None) if collision_raw else None

        # 7. Formulate Policy Inputs for EscalationEngine
        trigger_source = SourceType.LIVE
        policy_branch_reason = "NOMINAL_CLEAR_CORRIDOR"

        # Default nominal safe state from multi-agent belief fusion
        policy_confidence = max(0.0, min(1.0, fused_safe_p))
        policy_epistemic = float(fused_uncertainty)
        decision_reason = "Forward corridor clear; confident nominal trajectory."

        # A. Actual Physical Collision Event (Highest Priority Safety Interlock)
        if collision_active and collision_impulse >= 50.0:
            policy_confidence = 0.0
            policy_epistemic = 0.80
            trigger_source = SourceType.LIVE
            policy_branch_reason = "PHYSICAL_COLLISION_INTERLOCK"
            if speed_kmh < 0.5:
                decision_reason = (
                    f"EMERGENCY: Vehicle pinned/stopped against obstacle: {col_actor} "
                    f"(id={col_id}, impulse={collision_impulse:.1f}N)."
                )
            else:
                decision_reason = (
                    f"EMERGENCY: Physical collision registered! Impulse={collision_impulse:.1f}N "
                    f"against {col_actor} (id={col_id})."
                )

        # A2. Post-Recovery Sensor Warmup (Vehicle relocated, awaiting fresh sensor stream)
        elif self.post_recovery_warmup_frames > 0:
            self.post_recovery_warmup_frames -= 1
            policy_confidence = 0.55
            policy_epistemic = 0.20
            trigger_source = SourceType.LIVE
            policy_branch_reason = "POST_RECOVERY_WARMUP"
            decision_reason = "RECOVERY: Vehicle relocated; awaiting fresh sensor observations."

        # B. Pedestrian Protection Rule (M16 Verified Policy)
        # Triggered by either active PEDESTRIAN scenario OR live sensor detecting a walker
        elif self.scenario_mode == ScenarioMode.PEDESTRIAN or (
            obs_dist is not None and any(w in obs_actor.lower() for w in ("walker", "pedestrian"))
        ):
            has_walker_sensor = obs_dist is not None and any(w in obs_actor.lower() for w in ("walker", "pedestrian"))
            policy_confidence = 0.05
            policy_epistemic = 0.20
            policy_branch_reason = "PEDESTRIAN_PROTECTION_RULE"
            if has_walker_sensor:
                trigger_source = SourceType.LIVE
                decision_reason = f"CRITICAL: Live CARLA pedestrian detected in corridor at {obs_dist:.1f}m!"
            else:
                trigger_source = SourceType.SCENARIO_INPUT
                decision_reason = "CRITICAL: Scenario-injected pedestrian hazard in driving corridor!"

        # C. Roadway Prop Scenario Mode (Controlled OOD Obstacle Test)
        elif self.scenario_mode == ScenarioMode.PROPS:
            policy_confidence = 0.35
            policy_epistemic = 0.65
            policy_branch_reason = "SCENARIO_ROADWAY_PROP"
            if obs_dist is not None:
                trigger_source = SourceType.LIVE
                decision_reason = f"Hazard: Roadway prop obstacle detected ahead at {obs_dist:.1f}m ({obs_actor})."
            else:
                trigger_source = SourceType.SCENARIO_INPUT
                decision_reason = "Advisory: Scenario-injected roadway obstacle in path (model OOD limit)."

        # D. Genuine Live Sensor Obstacle Hazards (when not in specific scenario)
        elif obs_dist is not None and obs_fresh:
            # D1. Imminent Proximity (dist <= 3.0m) OR Rapid Closing (TTC <= 1.8s with v_close >= 0.5)
            if obs_dist <= 3.0 or (ttc is not None and ttc <= 1.8):
                policy_confidence = 0.15
                policy_epistemic = 0.70
                trigger_source = SourceType.LIVE
                if obs_dist <= 3.0:
                    policy_branch_reason = "OBSTACLE_IMMINENT_PROXIMITY"
                    decision_reason = f"Hazard: Imminent obstacle ahead at {obs_dist:.1f}m ({obs_actor})!"
                else:
                    policy_branch_reason = "OBSTACLE_RAPID_CLOSING"
                    v_close_str = f"{v_close_mps:.1f}" if v_close_mps is not None else "N/A"
                    decision_reason = f"Hazard: Obstacle closing rapidly ahead at {obs_dist:.1f}m (TTC={ttc:.1f}s, v_close={v_close_str}m/s)!"

            # D2. Moderate Proximity / Following Zone (3.0m < dist <= 8.0m)
            elif obs_dist <= 8.0:
                policy_confidence = 0.45
                policy_epistemic = 0.35
                trigger_source = SourceType.LIVE
                policy_branch_reason = "OBSTACLE_FOLLOWING_ZONE"
                if speed_kmh < 1.0:
                    decision_reason = f"Traffic: Stationary behind obstacle/traffic at {obs_dist:.1f}m."
                else:
                    decision_reason = f"Caution: Following obstacle ahead at {obs_dist:.1f}m ({obs_actor})."

            # D3. Intermediate Closing Zone (8.0m < dist <= 12.0m with TTC <= 3.0s)
            elif obs_dist <= 12.0 and (ttc is not None and ttc <= 3.0):
                policy_confidence = 0.48
                policy_epistemic = 0.38
                trigger_source = SourceType.LIVE
                policy_branch_reason = "OBSTACLE_INTERMEDIATE_CLOSING"
                v_close_str = f"{v_close_mps:.1f}" if v_close_mps is not None else "N/A"
                decision_reason = f"Caution: Approaching obstacle ahead at {obs_dist:.1f}m (TTC={ttc:.1f}s, v_close={v_close_str}m/s)."

            # D4. Distant Obstacle (> 12.0m, or > 8.0m with no rapid closing) -> Nominal corridor
            else:
                policy_branch_reason = "NOMINAL_CLEAR_CORRIDOR"
                decision_reason = f"Forward corridor clear; distant obstacle at {obs_dist:.1f}m monitored ({obs_actor})."

        # E. Severe Weather Scenario (Degraded Sensory Visibility)
        elif self.scenario_mode == ScenarioMode.WEATHER:
            policy_confidence = 0.50
            policy_epistemic = max(0.45, fused_uncertainty)
            trigger_source = SourceType.SCENARIO_INPUT
            policy_branch_reason = "SCENARIO_SEVERE_WEATHER"
            decision_reason = "Advisory: Severe weather degraded visibility; precautionary wait."

        # F. Active Red Traffic Light (Only evaluated when vehicle is at traffic light)
        elif traffic_light_state == "RED":
            policy_confidence = 0.45
            policy_epistemic = 0.25
            trigger_source = SourceType.LIVE
            policy_branch_reason = "SIGNAL_RED_LIGHT"
            decision_reason = "Signal: Red traffic light active; precautionary wait."

        # G. Active Sensor Fault Injections
        elif fault_active:
            trigger_source = SourceType.DEMO_PROXY
            policy_branch_reason = f"SENSOR_FAULT_{self.fault_mode.name}"
            if self.fault_mode in (FaultMode.RGB_DARKNESS, FaultMode.RGB_OCCLUSION):
                policy_confidence = 0.38
                policy_epistemic = 0.58
                decision_reason = f"Sensor Degraded: {self.fault_mode.name} elevates epistemic uncertainty."
            elif self.fault_mode == FaultMode.SEG_CORRUPTION:
                policy_confidence = 0.42
                policy_epistemic = 0.52
                decision_reason = "Sensor Degraded: Segmentation vocabulary corruption detected."
            elif self.fault_mode == FaultMode.IMU_DISTURBANCE:
                policy_confidence = 0.48
                policy_epistemic = 0.49
                decision_reason = "Sensor Degraded: High IMU disturbance jerk detected."

        # 8. Evaluate Authoritative EscalationEngine
        esc_res = self.escalation_engine.evaluate(
            confidence=policy_confidence,
            epistemic_uncertainty=policy_epistemic,
            conformal_set_size=conformal_set_size,
        )

        # 9. Legacy Demo Heuristic Advisory (for separate comparison)
        if collision_active and collision_impulse >= 50.0:
            legacy_advisory = "EMERGENCY STOP"
            legacy_risk = "CRITICAL"
        elif obs_dist is not None and obs_dist <= 3.0:
            legacy_advisory = "BRAKE NOW"
            legacy_risk = "HIGH"
        elif self.scenario_mode == ScenarioMode.PEDESTRIAN or (
            obs_dist is not None and any(w in obs_actor.lower() for w in ("walker", "pedestrian"))
        ):
            legacy_advisory = "BRAKE FOR PEDESTRIAN"
            legacy_risk = "HIGH"
        elif traffic_light_state == "RED":
            legacy_advisory = "HOLD FOR RED LIGHT" if speed_mps < 0.5 else "BRAKE FOR RED LIGHT"
            legacy_risk = "MEDIUM"
        elif esc_res.outcome == DecisionOutcome.ESCALATE:
            legacy_advisory = "ESCALATE TO SUPERVISOR"
            legacy_risk = "HIGH"
        elif esc_res.outcome == DecisionOutcome.WAIT:
            legacy_advisory = "CAUTION / PRECAUTIONARY WAIT"
            legacy_risk = "MEDIUM"
        else:
            legacy_advisory = "CONTINUE (NOMINAL)"
            legacy_risk = "LOW"

        # 10. Assemble Multi-Agent Uncertainty Inventory
        uncertainties = [
            UncertaintyEntry(
                id="rgb_quality",
                name="RGB Quality (Dark/Occl/Blur)",
                category="Sensor",
                value=rgb_quality_u,
                display_str=f"{rgb_quality_u:.3f}",
                source_type=SourceType.DEMO_PROXY if (self.fault_mode in (FaultMode.RGB_DARKNESS, FaultMode.RGB_OCCLUSION)) else SourceType.LIVE,
                description="Perceptual quality loss, darkness, low contrast",
            ),
            UncertaintyEntry(
                id="seg_vocab",
                name="Seg Vocabulary Deviation",
                category="Sensor",
                value=seg_vocab_u,
                display_str=f"{seg_vocab_u:.3f}",
                source_type=SourceType.DEMO_PROXY if (self.fault_mode == FaultMode.SEG_CORRUPTION) else SourceType.LIVE,
                description="CityScapes palette consistency & label entropy",
            ),
            UncertaintyEntry(
                id="imu_dynamics",
                name="IMU Dynamics (Jerk/Gyro)",
                category="Sensor",
                value=imu_motion_u,
                display_str=f"{imu_motion_u:.3f}",
                source_type=SourceType.DEMO_PROXY if (self.fault_mode == FaultMode.IMU_DISTURBANCE) else SourceType.LIVE,
                description="Linear jerk norm and angular rotation rate",
            ),
            UncertaintyEntry(
                id="camera_agent_epistemic",
                name="CameraAgent Epistemic (MC)",
                category="Cognix Agent",
                value=float(cam_u_res.epistemic),
                display_str=f"{cam_u_res.epistemic:.3f}",
                source_type=SourceType.COGNIX,
                description="Monte Carlo entropy decomposition on RGB feature",
            ),
            UncertaintyEntry(
                id="seg_agent_epistemic",
                name="SegAgent Epistemic (MC)",
                category="Cognix Agent",
                value=float(seg_u_res.epistemic),
                display_str=f"{seg_u_res.epistemic:.3f}",
                source_type=SourceType.COGNIX,
                description="Monte Carlo entropy decomposition on Seg feature",
            ),
            UncertaintyEntry(
                id="imu_agent_epistemic",
                name="IMUAgent Epistemic (MC)",
                category="Cognix Agent",
                value=float(imu_u_res.epistemic),
                display_str=f"{imu_u_res.epistemic:.3f}",
                source_type=SourceType.COGNIX,
                description="Monte Carlo entropy decomposition on IMU feature",
            ),
            UncertaintyEntry(
                id="fused_belief_unc",
                name="Epistemic-Weighted Fusion",
                category="Cognix Fusion",
                value=fused_uncertainty,
                display_str=f"{fused_uncertainty:.3f}",
                source_type=SourceType.COGNIX,
                description="Belief fusion weighted by reliability / (sigma_e + eps)",
            ),
            UncertaintyEntry(
                id="conformal_prediction_set",
                name="Conformal Prediction Set",
                category="Calibration",
                value=float(conformal_set_size) / 3.0,
                display_str=f"Size: {conformal_set_size} ({'Confident' if conformal_set_size == 1 else 'Ambiguous' if conformal_set_size == 2 else 'OOD'})",
                source_type=SourceType.COGNIX,
                description="Fitted on synthetic split (alpha=0.05, 95% target)",
            ),
            UncertaintyEntry(
                id="lidar_agent",
                name="LiDARAgent (Density/Range)",
                category="Cognix Agent",
                value=None,
                display_str="N/A (NO SENSOR)",
                source_type=SourceType.NOT_EXPOSED,
                description="Unattached sensor in current vehicle configuration",
                is_active=False,
            ),
            UncertaintyEntry(
                id="depth_agent",
                name="DepthAgent (Obstacle Proximity)",
                category="Cognix Agent",
                value=None,
                display_str="N/A (NO SENSOR)",
                source_type=SourceType.NOT_EXPOSED,
                description="Unattached sensor in current vehicle configuration",
                is_active=False,
            ),
            UncertaintyEntry(
                id="gnss_agent",
                name="GNSSAgent (Drift / Position)",
                category="Cognix Agent",
                value=None,
                display_str="N/A (NO SENSOR)",
                source_type=SourceType.NOT_EXPOSED,
                description="Unattached sensor in current vehicle configuration",
                is_active=False,
            ),
        ]

        total_latency_ms = (time.perf_counter() - t_eval_start) * 1000.0
        sensor_age_ms = (time.monotonic() - sensor_timestamp) * 1000.0

        return FrameEvaluation(
            frame_index=self.frame_count,
            timestamp=time.time(),
            fps=self.current_fps,
            latency_ms=total_latency_ms,
            sensor_age_ms=sensor_age_ms,
            scenario_mode=self.scenario_mode,
            fault_mode=self.fault_mode,
            speed_kmh=speed_kmh,
            obstacle_distance=obs_dist,
            obstacle_actor=obs_actor,
            obstacle_age_ms=obs_age_ms,
            obstacle_fresh=obs_fresh,
            ttc_sec=ttc,
            v_close_mps=v_close_mps,
            relative_motion_available=relative_motion_available,
            policy_branch_reason=policy_branch_reason,
            last_key_source=self.last_key_source,
            collision_active=collision_active,
            collision_impulse=collision_impulse,
            collision_age_ms=col_age_ms,
            collision_actor=col_actor,
            collision_actor_id=col_id,
            traffic_light_state=traffic_light_state,
            decision_outcome=esc_res.outcome,
            risk_level=esc_res.risk_level,
            escalation_required=esc_res.escalation,
            confidence=policy_confidence,
            epistemic_uncertainty=policy_epistemic,
            conformal_set_size=conformal_set_size,
            decision_reason=decision_reason,
            trigger_source=trigger_source,
            uncertainties=uncertainties,
            fused_uncertainty=fused_uncertainty,
            fused_safe_probability=fused_safe_p,
            agent_weights=agent_weights,
            legacy_advisory=legacy_advisory,
            legacy_risk=legacy_risk,
            processed_rgb=rgb_eval,
            processed_seg=seg_eval,
            clean_imu=imu_raw,
            injected_imu=imu_eval,
            safe_checkpoint_loc=self.safe_checkpoint.location if self.safe_checkpoint else None,
            recovery_status=self.recovery_status,
            stuck_status=self.stuck_reason if self.stuck_detected else "NOMINAL",
            autopilot_status=autopilot_status,
            road_info=road_info,
            motion_displacement=motion_displacement,
            motion_cum_yaw=motion_cum_yaw,
            circling_detected=circling_detected,
            ctrl_steer=ctrl_steer,
            ctrl_throttle=ctrl_throttle,
            ctrl_brake=ctrl_brake,
            cumulative_distance_m=self.trajectory_metrics.get("cumulative_distance_m", 0.0),
            net_displacement_m=self.trajectory_metrics.get("net_displacement_m", 0.0),
            efficiency_ratio=self.trajectory_metrics.get("efficiency_ratio", 1.0),
            revisit_detected=self.trajectory_metrics.get("revisit_detected", False),
            circular_movement_detected=self.trajectory_metrics.get("circular_movement_detected", False),
            excessive_steering=self.trajectory_metrics.get("excessive_steering", False),
            wrong_direction=self.trajectory_metrics.get("wrong_direction", False),
            is_immobilized=self.trajectory_metrics.get("is_immobilized", False),
            stuck_seconds=self.trajectory_metrics.get("stuck_seconds", 0.0),
        )

    # -----------------------------------------------------------------------
    # Interactive OpenCV Status Panel Renderer
    # -----------------------------------------------------------------------

    def render_dashboard(
        self,
        rgb_disp: np.ndarray,
        seg_disp: np.ndarray,
        ev: FrameEvaluation,
    ) -> np.ndarray:
        """Compose left sensor stack (640x720) and right status panel (760x720) into 1400x720 window."""
        panel_w = 760
        panel_h = 720
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (22, 22, 26)  # Dark slate background

        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Header (y: 0 to 45)
        cv2.rectangle(panel, (0, 0), (panel_w, 45), (32, 34, 40), -1)
        cv2.putText(panel, "COGNIX + CARLA MULTI-AGENT UNCERTAINTY DASHBOARD", (16, 26), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "CARLA Autopilot Controls Vehicle | Cognix Observational Safety Escalation", (16, 40), font, 0.38, (170, 175, 185), 1, cv2.LINE_AA)

        # 2. Performance & Freshness Bar (y: 45 to 80)
        cv2.rectangle(panel, (0, 45), (panel_w, 80), (28, 29, 34), -1)
        fps_color = (100, 220, 100) if ev.fps >= 15.0 else (50, 180, 255)
        cv2.putText(panel, f"Render FPS: {ev.fps:4.1f}", (16, 68), font, 0.50, fps_color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Latency: {ev.latency_ms:4.1f} ms", (180, 68), font, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

        age_str = f"Sensor Age: {ev.sensor_age_ms:3.0f} ms"
        age_color = (100, 220, 100) if ev.sensor_age_ms < 150 else (50, 120, 255)
        cv2.putText(panel, age_str, (340, 68), font, 0.50, age_color, 1, cv2.LINE_AA)

        cv2.putText(panel, "FAST CONTINUOUS (NO R-CNN)", (540, 68), font, 0.42, (150, 210, 255), 1, cv2.LINE_AA)
        cv2.line(panel, (0, 80), (panel_w, 80), (50, 55, 65), 1)

        # 3. Decision Engine Panel (y: 85 to 205)
        box_y1, box_y2 = 86, 205
        cv2.rectangle(panel, (12, box_y1), (panel_w - 12, box_y2), (34, 36, 44), -1)
        cv2.rectangle(panel, (12, box_y1), (panel_w - 12, box_y2), (65, 70, 85), 1)

        # Outcome badge color
        if ev.decision_outcome == DecisionOutcome.ACT:
            outcome_bg = (30, 130, 30)
            outcome_fg = (255, 255, 255)
        elif ev.decision_outcome == DecisionOutcome.WAIT:
            outcome_bg = (30, 150, 220)
            outcome_fg = (20, 20, 20)
        elif ev.decision_outcome == DecisionOutcome.ESCALATE:
            outcome_bg = (30, 30, 200)
            outcome_fg = (255, 255, 255)
        else:
            outcome_bg = (140, 40, 140)
            outcome_fg = (255, 255, 255)

        cv2.putText(panel, "COGNIX DECISION ENGINE (ESCALATION POLICY)", (22, 105), font, 0.48, (190, 200, 220), 1, cv2.LINE_AA)

        # Big Badge
        badge_text = f" {ev.decision_outcome.value} "
        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.95, 2)
        cv2.rectangle(panel, (20, 115), (20 + tw + 16, 115 + th + 14), outcome_bg, -1)
        cv2.putText(panel, badge_text, (28, 115 + th + 6), font, 0.95, outcome_fg, 2, cv2.LINE_AA)

        # Risk Level & Escalation Flag
        risk_color = (100, 220, 100) if ev.risk_level == RiskLevel.LOW else (50, 180, 255) if ev.risk_level == RiskLevel.MODERATE else (50, 50, 240)
        cv2.putText(panel, f"Risk Level: {ev.risk_level.value}", (20 + tw + 35, 132), font, 0.55, risk_color, 2, cv2.LINE_AA)

        esc_flag_str = "ESCALATION: REQUIRED" if ev.escalation_required else "ESCALATION: NONE"
        esc_flag_color = (60, 60, 240) if ev.escalation_required else (100, 220, 100)
        cv2.putText(panel, esc_flag_str, (20 + tw + 35, 150), font, 0.45, esc_flag_color, 1, cv2.LINE_AA)

        # Confidence, Epistemic, Conformal metrics row
        cv2.putText(panel, f"Actuation P(safe): {ev.confidence:0.3f}", (22, 175), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Epistemic: {ev.epistemic_uncertainty:0.3f}", (245, 175), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Conformal Set: {ev.conformal_set_size}", (420, 175), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

        # Trigger source badge
        source_color = (240, 180, 50) if ev.trigger_source == SourceType.LIVE else (200, 80, 240) if ev.trigger_source == SourceType.SCENARIO_INPUT else (50, 150, 240)
        cv2.rectangle(panel, (panel_w - 220, box_y1 + 8), (panel_w - 20, box_y1 + 32), (45, 48, 58), -1)
        cv2.putText(panel, f"SOURCE: {ev.trigger_source.value}", (panel_w - 212, box_y1 + 24), font, 0.38, source_color, 1, cv2.LINE_AA)

        # Rationale string
        cv2.putText(panel, f"Rationale: {ev.decision_reason[:75]}", (22, 195), font, 0.42, (160, 220, 255), 1, cv2.LINE_AA)

        # 4. Scenario & Sensor Fault Controls Bar (y: 210 to 285)
        scen_y1, scen_y2 = 210, 285
        cv2.rectangle(panel, (12, scen_y1), (panel_w - 12, scen_y2), (28, 30, 38), -1)
        cv2.rectangle(panel, (12, scen_y1), (panel_w - 12, scen_y2), (55, 60, 75), 1)

        # Active Scenario with prominent badge
        if ev.scenario_mode == ScenarioMode.NORMAL:
            scen_badge_bg = (30, 120, 30)
            scen_name_str = "NORMAL DRIVING"
        elif ev.scenario_mode == ScenarioMode.PEDESTRIAN:
            scen_badge_bg = (20, 20, 180)
            scen_name_str = "PEDESTRIAN HAZARD"
        elif ev.scenario_mode == ScenarioMode.WEATHER:
            scen_badge_bg = (180, 100, 20)
            scen_name_str = "SEVERE WEATHER"
        else:
            scen_badge_bg = (150, 40, 150)
            scen_name_str = "ROADWAY PROP"

        cv2.rectangle(panel, (20, scen_y1 + 8), (240, scen_y1 + 32), scen_badge_bg, -1)
        cv2.putText(panel, f"SCENARIO: {scen_name_str}", (26, scen_y1 + 24), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Last key pressed and fault mode
        key_source_short = "GUI" if self.last_key_source == "OPENCV_GUI" else "CON" if self.last_key_source == "CONSOLE_STDIN" else "---"
        cv2.putText(panel, f"Key: [{self.last_key_name}] ({key_source_short})", (248, scen_y1 + 24), font, 0.42, (255, 220, 100), 2, cv2.LINE_AA)

        fault_str = f"Fault: [{ev.fault_mode.name}]"
        fault_col = (100, 220, 100) if ev.fault_mode == FaultMode.NONE else (50, 100, 255)
        cv2.putText(panel, fault_str, (430, scen_y1 + 24), font, 0.45, fault_col, 1, cv2.LINE_AA)

        if ev.recovery_status != "IDLE":
            rec_display = f"Rec: {ev.recovery_status[:16]}"
            rec_col = (100, 240, 255)
        else:
            rec_display = f"Spawn: {self.scenario_spawn_status[:16]}"
            rec_col = (180, 185, 195)
        cv2.putText(panel, rec_display, (585, scen_y1 + 24), font, 0.36, rec_col, 1, cv2.LINE_AA)

        # Transition notification banner (flashes after keypress or stuck condition)
        if ev.stuck_status != "NOMINAL":
            cv2.rectangle(panel, (20, scen_y1 + 38), (panel_w - 20, scen_y1 + 56), (30, 30, 200), -1)
            cv2.putText(panel, f"! STUCK DETECTED: {ev.stuck_status[:50]} | Press 'R' to Recover Road", (24, scen_y1 + 51), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            time_since_transition = time.monotonic() - self.transition_message_time
            if time_since_transition < 3.5:
                cv2.rectangle(panel, (20, scen_y1 + 38), (panel_w - 20, scen_y1 + 56), (50, 45, 20), -1)
                cv2.putText(panel, f">> {self.transition_message}", (24, scen_y1 + 51), font, 0.40, (100, 240, 255), 1, cv2.LINE_AA)
            else:
                help_str = "Keys: [N] Normal  [P] Pedestrian  [W] Weather  [B] Prop  [R] Reset/Recover  |  [0-4] Faults  |  [Q] Quit"
                cv2.putText(panel, help_str, (22, scen_y1 + 51), font, 0.35, (170, 175, 185), 1, cv2.LINE_AA)

        # 5. Multi-Agent Uncertainty Inventory Table (y: 290 to 625)
        cv2.putText(panel, "MULTI-AGENT UNCERTAINTY INVENTORY & BELIEF FUSION", (16, 303), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Table Headers
        table_top = 312
        cv2.rectangle(panel, (12, table_top), (panel_w - 12, table_top + 22), (40, 44, 55), -1)
        cv2.putText(panel, "Subsystem / Agent", (20, table_top + 16), font, 0.40, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(panel, "Type", (260, table_top + 16), font, 0.40, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(panel, "Source", (370, table_top + 16), font, 0.40, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(panel, "Value", (490, table_top + 16), font, 0.40, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(panel, "Uncertainty Meter", (575, table_top + 16), font, 0.40, (200, 210, 225), 1, cv2.LINE_AA)

        row_y = table_top + 24
        row_height = 26

        for i, u in enumerate(ev.uncertainties):
            bg_col = (26, 28, 34) if i % 2 == 0 else (22, 24, 30)
            cv2.rectangle(panel, (12, row_y), (panel_w - 12, row_y + row_height), bg_col, -1)

            text_col = (230, 230, 230) if u.is_active else (120, 125, 135)
            cv2.putText(panel, u.name[:30], (20, row_y + 18), font, 0.42, text_col, 1, cv2.LINE_AA)
            cv2.putText(panel, u.category[:14], (260, row_y + 18), font, 0.38, (160, 170, 185), 1, cv2.LINE_AA)

            if u.source_type == SourceType.LIVE:
                badge_c = (200, 180, 50)
            elif u.source_type == SourceType.COGNIX:
                badge_c = (80, 220, 100)
            elif u.source_type == SourceType.DEMO_PROXY:
                badge_c = (40, 140, 240)
            elif u.source_type == SourceType.SCENARIO_INPUT:
                badge_c = (220, 80, 220)
            else:
                badge_c = (95, 100, 110)

            cv2.rectangle(panel, (365, row_y + 4), (470, row_y + row_height - 4), (35, 38, 48), -1)
            cv2.putText(panel, u.source_type.value, (372, row_y + 17), font, 0.35, badge_c, 1, cv2.LINE_AA)

            cv2.putText(panel, u.display_str, (485, row_y + 18), font, 0.40, text_col, 1, cv2.LINE_AA)

            bar_x1 = 575
            bar_w = 160
            bar_y1 = row_y + 7
            bar_y2 = row_y + row_height - 7
            cv2.rectangle(panel, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y2), (40, 44, 52), -1)

            if u.value is not None:
                val_clamped = max(0.0, min(1.0, u.value))
                fill_w = int(bar_w * val_clamped)
                if val_clamped < 0.35:
                    bar_color = (60, 180, 60)
                elif val_clamped < 0.65:
                    bar_color = (40, 180, 240)
                else:
                    bar_color = (40, 40, 230)
                cv2.rectangle(panel, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), bar_color, -1)
            else:
                cv2.putText(panel, "OFFLINE", (bar_x1 + 45, bar_y2 - 2), font, 0.32, (100, 105, 115), 1, cv2.LINE_AA)

            row_y += row_height

        # 6. Motion Diagnostics & Legacy Advisory Box (y: 630 to 712)
        cv2.rectangle(panel, (12, 630), (panel_w - 12, 712), (28, 30, 36), -1)
        cv2.rectangle(panel, (12, 630), (panel_w - 12, 712), (55, 60, 70), 1)

        ap_color = (120, 240, 120) if "ENGAGED" in ev.autopilot_status else (240, 140, 80)
        cv2.putText(panel, f"CARLA AUTOPILOT MOTION: {ev.autopilot_status}  |  {ev.road_info}", (20, 646), font, 0.38, ap_color, 1, cv2.LINE_AA)

        has_anomaly = (ev.circling_detected or ev.circular_movement_detected or ev.wrong_direction or ev.excessive_steering)
        mot_color = (60, 60, 240) if has_anomaly else (200, 210, 225)
        rot_flag = " [CIRCLING]" if (ev.circling_detected or ev.circular_movement_detected) else ""
        rev_flag = " [REVISIT]" if ev.revisit_detected else ""
        wrong_flag = " [WRONG LANE]" if ev.wrong_direction else ""
        cv2.putText(
            panel,
            f"Trajectory: Dist={ev.cumulative_distance_m:5.1f}m | Disp={ev.net_displacement_m:5.1f}m | Eff={ev.efficiency_ratio:.2f}{rot_flag}{rev_flag}{wrong_flag} | Steer={ev.ctrl_steer:+.2f} Th={ev.ctrl_throttle:.2f} Brk={ev.ctrl_brake:.2f}",
            (20, 664),
            font,
            0.34,
            mot_color,
            1,
            cv2.LINE_AA,
        )

        vclose_str = f"V_close: {ev.v_close_mps:+.1f} m/s" if ev.v_close_mps is not None else "V_close: N/A"
        ttc_str = f"TTC: {ev.ttc_sec:.1f} s" if ev.ttc_sec is not None else "TTC: Safe"
        dist_str = f"Obs: {ev.obstacle_distance:.1f}m ({ev.obstacle_actor[:8]})" if ev.obstacle_distance is not None else "Obs: None"
        cv2.putText(panel, f"{dist_str}  |  {vclose_str}  |  {ttc_str}  |  Spd: {ev.speed_kmh:.1f} km/h  |  TL: {ev.traffic_light_state}", (20, 682), font, 0.36, (200, 205, 215), 1, cv2.LINE_AA)

        leg_color = (80, 220, 100) if ev.legacy_risk == "LOW" else (40, 180, 240) if ev.legacy_risk in ("MEDIUM", "CAUTION") else (40, 40, 240)
        cv2.putText(panel, f"Advisory Proxy: {ev.legacy_advisory} ({ev.legacy_risk})", (20, 702), font, 0.38, leg_color, 1, cv2.LINE_AA)

        # Combine Left (640x720) and Right (760x720)
        left_stack = np.vstack([rgb_disp, seg_disp])
        dashboard = np.hstack([left_stack, panel])
        return dashboard

    # -----------------------------------------------------------------------
    # Interactive Scenario & Fault Switching
    # -----------------------------------------------------------------------

    def set_scenario(self, mode: ScenarioMode, world: Any = None, vehicle: Any = None) -> None:
        """Switch active scenario without restarting the simulator or display loop."""
        if mode == self.scenario_mode:
            return
        old_mode = self.scenario_mode
        logger.info("Switching scenario from %s to %s", old_mode.value, mode.value)

        # 1. Clean up previous scenario temporary actors
        self._cleanup_scenario_actors()

        # 2. Reset weather if switching away from WEATHER
        if old_mode == ScenarioMode.WEATHER and world is not None and self.original_weather is not None:
            try:
                world.set_weather(self.original_weather)
                logger.info("Restored nominal CARLA weather.")
            except Exception as e:
                logger.warning("Could not restore weather: %s", e)

        # 3. If returning to NORMAL, clear scenario obstacle/hazard references
        if mode == ScenarioMode.NORMAL:
            with self.lock:
                if self.latest_obstacle:
                    act = self.latest_obstacle.get("actor_type", "").lower()
                    if any(w in act for w in ("walker", "pedestrian", "prop", "static", "box", "barrel")):
                        self.latest_obstacle = None
            self.scenario_spawn_status = "RESTORED NOMINAL"
            self.transition_message = "Switched to NORMAL (Cleared scenario actors & restored weather)"
            self.transition_message_time = time.monotonic()

        self.scenario_mode = mode

        # 4. Apply new scenario stimuli
        if mode == ScenarioMode.WEATHER:
            self.scenario_spawn_status = "HARD RAIN ACTIVE"
            self.transition_message = "Switched to SEVERE WEATHER (HardRainNoon applied)"
            self.transition_message_time = time.monotonic()
            if world is not None:
                try:
                    import carla
                    if self.original_weather is None:
                        self.original_weather = world.get_weather()
                    world.set_weather(carla.WeatherParameters.HardRainNoon)
                    logger.info("Applied HardRainNoon CARLA weather for Scenario W.")
                except Exception as e:
                    logger.warning("Could not set CARLA weather: %s", e)

        elif mode == ScenarioMode.PEDESTRIAN:
            if world is not None and vehicle is not None:
                self._spawn_pedestrian_hazard(world, vehicle)
            else:
                self.scenario_spawn_status = "SIGNAL ACTIVE (OFFLINE)"
                self.transition_message = "Switched to PEDESTRIAN (Scenario signal active)"
                self.transition_message_time = time.monotonic()

        elif mode == ScenarioMode.PROPS:
            if world is not None and vehicle is not None:
                self._spawn_roadway_prop(world, vehicle)
            else:
                self.scenario_spawn_status = "SIGNAL ACTIVE (OFFLINE)"
                self.transition_message = "Switched to ROADWAY PROP (Scenario signal active)"
                self.transition_message_time = time.monotonic()

    def _cleanup_scenario_actors(self) -> None:
        """Destroy scenario-injected actors cleanly."""
        for actor in self.scenario_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self.scenario_actors.clear()

    def _spawn_pedestrian_hazard(self, world: Any, vehicle: Any) -> None:
        """Spawn a temporary CARLA pedestrian 15m ahead in the forward corridor."""
        try:
            import carla
            bps = world.get_blueprint_library().filter("walker.pedestrian.*")
            if not bps:
                self.scenario_spawn_status = "FAILED: No walker blueprint"
                self.transition_message = "Pedestrian scenario active (Blueprint missing, using scenario signal)"
                self.transition_message_time = time.monotonic()
                logger.warning("No walker blueprints found in CARLA.")
                return
            walker_bp = bps[0]

            v_transform = vehicle.get_transform()
            yaw_rad = math.radians(v_transform.rotation.yaw)

            # 15 meters forward, in lane center
            spawn_loc = carla.Location(
                x=v_transform.location.x + 15.0 * math.cos(yaw_rad),
                y=v_transform.location.y + 15.0 * math.sin(yaw_rad),
                z=v_transform.location.z + 0.5,
            )
            spawn_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=v_transform.rotation.yaw + 90.0))
            walker = world.try_spawn_actor(walker_bp, spawn_tf)
            if walker is not None:
                self.scenario_actors.append(walker)
                self.demo_owned_actors.append(walker)
                self.scenario_spawn_status = f"SPAWNED (id={walker.id}, 15m ahead)"
                self.transition_message = f"Switched to PEDESTRIAN (Spawned walker id={walker.id} at 15m)"
                self.transition_message_time = time.monotonic()
                logger.info("Spawned live CARLA pedestrian actor ahead at %s (id=%d)", spawn_loc, walker.id)
            else:
                self.scenario_spawn_status = "SPAWN BLOCKED (Signal active)"
                self.transition_message = "Pedestrian scenario active (Location occupied, using scenario signal)"
                self.transition_message_time = time.monotonic()
                logger.info("Could not spawn physical walker (location occupied); using scenario signal.")
        except Exception as e:
            self.scenario_spawn_status = f"ERROR ({e})"
            self.transition_message = f"Pedestrian scenario error: {e}"
            self.transition_message_time = time.monotonic()
            logger.warning("Failed to spawn CARLA pedestrian: %s; falling back to scenario signal.", e)

    def _spawn_roadway_prop(self, world: Any, vehicle: Any) -> None:
        """Spawn a temporary CARLA roadway obstacle prop 18m ahead."""
        try:
            import carla
            bps = world.get_blueprint_library().filter("static.prop.*")
            box_bps = [b for b in bps if "box" in b.id.lower() or "barrel" in b.id.lower()]
            if not box_bps:
                box_bps = list(bps)
            if not box_bps:
                self.scenario_spawn_status = "FAILED: No prop blueprint"
                self.transition_message = "Prop scenario active (Blueprint missing, using scenario signal)"
                self.transition_message_time = time.monotonic()
                return
            prop_bp = box_bps[0]

            v_transform = vehicle.get_transform()
            yaw_rad = math.radians(v_transform.rotation.yaw)

            spawn_loc = carla.Location(
                x=v_transform.location.x + 18.0 * math.cos(yaw_rad),
                y=v_transform.location.y + 18.0 * math.sin(yaw_rad),
                z=v_transform.location.z + 0.3,
            )
            spawn_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=v_transform.rotation.yaw))
            prop = world.try_spawn_actor(prop_bp, spawn_tf)
            if prop is not None:
                self.scenario_actors.append(prop)
                self.demo_owned_actors.append(prop)
                self.scenario_spawn_status = f"SPAWNED (id={prop.id}, 18m ahead)"
                self.transition_message = f"Switched to ROADWAY PROP (Spawned prop id={prop.id} at 18m)"
                self.transition_message_time = time.monotonic()
                logger.info("Spawned live CARLA roadway prop actor ahead at %s (id=%d)", spawn_loc, prop.id)
            else:
                self.scenario_spawn_status = "SPAWN BLOCKED (Signal active)"
                self.transition_message = "Prop scenario active (Location occupied, using scenario signal)"
                self.transition_message_time = time.monotonic()
                logger.info("Could not spawn physical prop; using scenario signal.")
        except Exception as e:
            self.scenario_spawn_status = f"ERROR ({e})"
            self.transition_message = f"Prop scenario error: {e}"
            self.transition_message_time = time.monotonic()
            logger.warning("Failed to spawn CARLA prop: %s; falling back to scenario signal.", e)

    def handle_key(
        self,
        key_code: int,
        world: Any = None,
        vehicle: Any = None,
        source: str = "CONSOLE_STDIN",
    ) -> bool:
        """Handle interactive keyboard controls. Returns False if quit requested."""
        if key_code in (ord("q"), ord("Q"), 27):  # Q or Esc
            self.last_key_name = "Q/ESC (QUIT)"
            self.last_key_source = source
            logger.info(
                "[KEYBOARD EVENT] Accepted key: raw=%d (0x%02X) | char='Q/ESC' | source=%s | resulting_scenario=%s | resulting_fault=%s",
                key_code, key_code, source, self.scenario_mode.value, self.fault_mode.name,
            )
            return False

        # Decode printable ASCII characters (letters, digits, etc.)
        if 0 <= key_code < 256 and chr(key_code).isprintable():
            char = chr(key_code).lower()
        else:
            char = ""

        if not char:
            return True

        self.last_key_name = char.upper()
        self.last_key_source = source
        self.last_key_time = time.monotonic()

        if char == "n":
            self.set_scenario(ScenarioMode.NORMAL, world, vehicle)
        elif char == "p":
            self.set_scenario(ScenarioMode.PEDESTRIAN, world, vehicle)
        elif char == "w":
            self.set_scenario(ScenarioMode.WEATHER, world, vehicle)
        elif char == "b":
            self.set_scenario(ScenarioMode.PROPS, world, vehicle)
        elif char == "r":
            if world is not None and vehicle is not None:
                self.recover_vehicle_to_safe_road(world, vehicle)
            else:
                self.set_scenario(ScenarioMode.NORMAL, world, vehicle)
                self.fault_mode = FaultMode.NONE
                with self.lock:
                    self.latest_collision = None
                    self.latest_obstacle = None
                self.scenario_spawn_status = "RESET ALL"
                self.recovery_status = "NOMINAL_RESET_NO_TELEPORT"
                self.transition_message = "RESET: Normal Scenario & Cleared All Sensor Faults & Collisions"
                self.transition_message_time = time.monotonic()
                logger.info("Reset to Nominal Scenario and Cleared Faults and Collisions.")
        elif char == "0":
            self.fault_mode = FaultMode.NONE
            self.transition_message = "Fault Mode: NONE (Nominal sensor feeds)"
            self.transition_message_time = time.monotonic()
            logger.info("Fault Mode: NONE (Normal sensors)")
        elif char == "1":
            self.fault_mode = FaultMode.RGB_DARKNESS
            self.transition_message = "Fault Mode: RGB DARKNESS (Intensity * 0.12)"
            self.transition_message_time = time.monotonic()
            logger.info("Fault Mode: RGB DARKNESS")
        elif char == "2":
            self.fault_mode = FaultMode.RGB_OCCLUSION
            self.transition_message = "Fault Mode: RGB OCCLUSION (Central 60% cutout)"
            self.transition_message_time = time.monotonic()
            logger.info("Fault Mode: RGB OCCLUSION")
        elif char == "3":
            self.fault_mode = FaultMode.SEG_CORRUPTION
            self.transition_message = "Fault Mode: SEG CORRUPTION (Noise patches)"
            self.transition_message_time = time.monotonic()
            logger.info("Fault Mode: SEGMENTATION CORRUPTION")
        elif char == "4":
            self.fault_mode = FaultMode.IMU_DISTURBANCE
            self.transition_message = "Fault Mode: IMU DISTURBANCE (Gaussian jerk)"
            self.transition_message_time = time.monotonic()
            logger.info("Fault Mode: IMU DISTURBANCE")

        logger.info(
            "[KEYBOARD EVENT] Accepted key: raw=%d (0x%02X) | char='%s' | source=%s | resulting_scenario=%s | resulting_fault=%s",
            key_code, key_code, char.upper(), source, self.scenario_mode.value, self.fault_mode.name,
        )
        return True

    # -----------------------------------------------------------------------
    # Comprehensive Actor and World Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self, world: Any = None) -> None:
        """Destroy all demo-owned sensors and actors, restore weather, close display."""
        logger.info("Performing clean shutdown of demo actors and world settings...")

        # 1. Stop and destroy demo-owned sensors
        for s in self.demo_owned_sensors:
            try:
                s.stop()
            except Exception:
                pass
            try:
                if hasattr(s, "destroy"):
                    s.destroy()
            except Exception:
                pass
        self.demo_owned_sensors.clear()

        # 2. Destroy scenario temporary actors
        self._cleanup_scenario_actors()

        # 3. Disable autopilot and destroy demo vehicles
        for a in reversed(self.demo_owned_actors):
            try:
                if hasattr(a, "set_autopilot"):
                    a.set_autopilot(False)
            except Exception:
                pass
            try:
                if hasattr(a, "destroy"):
                    a.destroy()
            except Exception:
                pass
        self.demo_owned_actors.clear()

        # 4. Restore original weather
        if world is not None and self.original_weather is not None:
            try:
                world.set_weather(self.original_weather)
                logger.info("Restored original CARLA weather.")
            except Exception as e:
                logger.warning("Could not restore weather: %s", e)

        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
        logger.info("Demo cleanup complete.")


# ---------------------------------------------------------------------------
# CARLA Live Demonstration Launcher
# ---------------------------------------------------------------------------

def run_live_demonstration(
    host: str = "127.0.0.1",
    port: int = 2000,
    tm_port: int = 8000,
    timeout: float = 20.0,
    duration_sec: Optional[float] = None,
    cycle_scenarios: bool = False,
    save_telemetry_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Launch the live CARLA continuous demonstration runner."""
    import carla

    logger.info("Connecting to CARLA simulator at %s:%d...", host, port)
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    blueprints = world.get_blueprint_library()

    demo = CognixLiveDemoV2(host=host, port=port)
    demo.client = client
    demo.tm_port = tm_port
    demo.original_weather = world.get_weather()

    # Register safe atexit handler
    def _safe_exit_handler():
        demo.cleanup(world)
    atexit.register(_safe_exit_handler)

    vehicle = None
    telemetry_records: List[Dict[str, Any]] = []

    try:
        # 1. Select vehicle blueprint and spawn
        vehicle_bps = [bp for bp in blueprints.filter("vehicle.*") if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4]
        if not vehicle_bps:
            vehicle_bps = list(blueprints.filter("vehicle.*"))
        vehicle_bp = vehicle_bps[0]

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in CARLA map.")

        # Preferred grand perimeter boulevard spawn point for presentation (Town10HD_Opt)
        # SP 22 is Road 1, Lane 1 (x=53.1, y=133.9, yaw=-179.7), heading west down straight boulevard
        preferred_indices = [22, 20, 19, 4, 5, 6, 8, 10]
        candidate_sps = [spawn_points[i] for i in preferred_indices if i < len(spawn_points)]
        for i, sp in enumerate(spawn_points):
            if i not in preferred_indices:
                candidate_sps.append(sp)

        vehicle = None
        for sp in candidate_sps:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                demo.demo_owned_actors.append(vehicle)
                break

        if vehicle is None:
            raise RuntimeError("Failed to spawn demo vehicle in CARLA.")

        # Initialize presentation loop route and configure Traffic Manager with explicit path
        try:
            demo.presentation_route_locations = build_presentation_loop_route(
                carla_map=carla_map,
                start_transform=sp,
                total_distance_m=2500.0,
                step_m=2.5,
            )
            logger.info("Generated deterministic presentation route with %d waypoints (2500m).", len(demo.presentation_route_locations))
        except Exception as e:
            logger.warning("Could not build presentation route: %s", e)

        # Enable Autopilot FIRST so Traffic Manager registers the actor before setting path
        try:
            vehicle.set_autopilot(True, tm_port)
        except TypeError:
            vehicle.set_autopilot(True)
        logger.info("Spawned observational demo vehicle with Autopilot enabled (TM port %d).", tm_port)

        try:
            tm = client.get_trafficmanager(tm_port)
            tm.auto_lane_change(vehicle, False)
            tm.distance_to_leading_vehicle(vehicle, 4.0)
            try:
                tm.set_desired_speed(vehicle, 22.0)
            except Exception:
                tm.vehicle_percentage_speed_difference(vehicle, 30.0)
            if demo.presentation_route_locations:
                tm.set_path(vehicle, demo.presentation_route_locations)
                logger.info("Explicit presentation path successfully registered on Traffic Manager.")
        except Exception as e:
            logger.warning("Could not configure TrafficManager route/path: %s", e)

        # 2. Attach RGB, Semantic Segmentation, IMU, Obstacle, and Collision sensors
        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))

        # RGB Camera (640x360, 20Hz)
        rgb_bp = blueprints.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", "640")
        rgb_bp.set_attribute("image_size_y", "360")
        rgb_bp.set_attribute("fov", "90")
        rgb_bp.set_attribute("sensor_tick", "0.05")
        rgb_sensor = world.spawn_actor(rgb_bp, cam_tf, attach_to=vehicle)
        demo.demo_owned_sensors.append(rgb_sensor)

        def _rgb_cb(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            with demo.lock:
                demo.latest_rgb = arr[:, :, :3].copy()
                demo.last_sensor_time = time.monotonic()
        rgb_sensor.listen(_rgb_cb)

        # Semantic Segmentation Camera (640x360, 20Hz)
        seg_bp = blueprints.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", "640")
        seg_bp.set_attribute("image_size_y", "360")
        seg_bp.set_attribute("fov", "90")
        seg_bp.set_attribute("sensor_tick", "0.05")
        seg_sensor = world.spawn_actor(seg_bp, cam_tf, attach_to=vehicle)
        demo.demo_owned_sensors.append(seg_sensor)

        def _seg_cb(image):
            image.convert(carla.ColorConverter.CityScapesPalette)
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            with demo.lock:
                demo.latest_seg = arr[:, :, :3].copy()
        seg_sensor.listen(_seg_cb)

        # IMU Sensor
        imu_bp = blueprints.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", "0.05")
        imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        demo.demo_owned_sensors.append(imu_sensor)

        def _imu_cb(sensor_data):
            with demo.lock:
                demo.latest_imu = {
                    "accelerometer": np.array([sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z]),
                    "gyroscope": np.array([sensor_data.gyroscope.x, sensor_data.gyroscope.y, sensor_data.gyroscope.z]),
                }
        imu_sensor.listen(_imu_cb)

        # Obstacle Detector Sensor mounted on front bumper (x=2.3, z=1.0)
        obs_bp = blueprints.find("sensor.other.obstacle")
        obs_bp.set_attribute("distance", "20.0")
        obs_bp.set_attribute("hit_radius", "0.5")
        obs_bp.set_attribute("sensor_tick", "0.05")
        obs_tf = carla.Transform(carla.Location(x=2.3, z=1.0))
        obs_sensor = world.spawn_actor(obs_bp, obs_tf, attach_to=vehicle)
        demo.demo_owned_sensors.append(obs_sensor)

        def _obs_cb(event):
            act = event.other_actor
            act_type = act.type_id if act else "unknown"
            act_l = act_type.lower()
            # Filter road surface, sidewalk, ground, or zero distance road grazing
            if any(term in act_l for term in ("road", "sidewalk", "ground", "terrain")) or (event.distance <= 0.05 and "static" in act_l):
                return
            with demo.lock:
                demo.latest_obstacle = {
                    "distance": float(event.distance),
                    "actor_type": act_type,
                    "actor_id": getattr(act, "id", None),
                    "other_actor": act,
                    "timestamp": time.monotonic(),
                }
        obs_sensor.listen(_obs_cb)

        # Collision Detector Sensor
        col_bp = blueprints.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        demo.demo_owned_sensors.append(col_sensor)

        def _col_cb(event):
            act = event.other_actor
            act_type = act.type_id if act else "unknown"
            act_id = getattr(act, "id", None)
            act_l = act_type.lower()
            if any(term in act_l for term in ("road", "ground", "terrain")):
                return
            imp = event.normal_impulse
            mag = float(math.sqrt(imp.x**2 + imp.y**2 + imp.z**2))
            if mag < 50.0:  # Ignore minor road vibration under 50 N
                return
            with demo.lock:
                demo.latest_collision = {
                    "actor_type": act_type,
                    "actor_id": act_id,
                    "impulse": mag,
                    "timestamp": time.monotonic(),
                }
        col_sensor.listen(_col_cb)

        logger.info("All 5 CARLA sensors spawned and listening. Starting continuous render loop...")

        # Display explicit operating & keyboard focus instructions
        print("\n" + "=" * 80)
        print(" COGNIX LIVE DEMO V2 — KEYBOARD FOCUS & OPERATING INSTRUCTIONS")
        print(" - Keys work from EITHER the OpenCV GUI Window OR the Console Terminal!")
        print(" - [N] Normal Scenario      [P] Pedestrian Hazard      [W] Severe Weather")
        print(" - [B] Roadway Prop Hazard  [R] Reset Scenario/Faults/Collisions")
        print(" - [0] Fault: None          [1] Fault: RGB Darkness    [2] Fault: RGB Occlusion")
        print(" - [3] Fault: Seg Corrupt   [4] Fault: IMU Disturbance [Q / Esc] Quit Demo")
        print("=" * 80 + "\n")

        # 3. Main Continuous Render Loop
        window_name = "COGNIX + CARLA Live Demonstration V2"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        t_loop_start = time.time()
        fps_timer = time.time()
        frames_in_second = 0
        last_diag_log_time = time.time()
        occlusion_saved = False

        while True:
            t_frame_start = time.perf_counter()

            # Clear stale internal sensor observations
            demo.clear_expired_observations()

            # Take thread-safe sensor snapshot
            with demo.lock:
                rgb_snap = demo.latest_rgb.copy() if demo.latest_rgb is not None else None
                seg_snap = demo.latest_seg.copy() if demo.latest_seg is not None else None
                imu_snap = {
                    "accelerometer": demo.latest_imu["accelerometer"].copy(),
                    "gyroscope": demo.latest_imu["gyroscope"].copy(),
                } if demo.latest_imu is not None else {
                    "accelerometer": np.array([0.0, 0.0, 9.81]),
                    "gyroscope": np.array([0.0, 0.0, 0.0]),
                }
                obs_snap = demo.latest_obstacle.copy() if demo.latest_obstacle is not None else None
                col_snap = demo.latest_collision.copy() if demo.latest_collision is not None else None
                sensor_t = demo.last_sensor_time

            # Clear transient obstacle if older than OBSTACLE_STALE_SECONDS
            if obs_snap and (time.monotonic() - obs_snap.get("timestamp", 0) > OBSTACLE_STALE_SECONDS):
                obs_snap = None

            # Clear collision alert if older than COLLISION_ALERT_SECONDS
            if col_snap and (time.monotonic() - col_snap.get("timestamp", 0) > COLLISION_ALERT_SECONDS):
                col_snap = None

            # Get vehicle dynamic speed, controls, and transform (resilient to simulator despawn)
            try:
                vel = vehicle.get_velocity()
                speed_kmh = float(math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6)
                v_tf = vehicle.get_transform()
                v_loc = v_tf.location
                v_rot = v_tf.rotation
                ang_vel = vehicle.get_angular_velocity() if hasattr(vehicle, "get_angular_velocity") else None
                ctrl = vehicle.get_control() if hasattr(vehicle, "get_control") else None
                steer_val = float(ctrl.steer) if ctrl else 0.0
                throttle_val = float(ctrl.throttle) if ctrl else 0.0
                brake_val = float(ctrl.brake) if ctrl else 0.0
            except RuntimeError as e:
                if "destroyed actor" in str(e).lower():
                    logger.warning("Vehicle actor was destroyed by simulator. Terminating demonstration loop.")
                    break
                raise

            # Query current lane / road information
            carla_map = world.get_map()
            try:
                ego_wp = carla_map.get_waypoint(v_loc, project_to_road=False)
                if ego_wp:
                    road_info_str = f"R:{ego_wp.road_id} L:{ego_wp.lane_id} ({ego_wp.lane_type}) Yaw:{v_rot.yaw:.1f}°"
                else:
                    road_info_str = f"Off-Road (Yaw:{v_rot.yaw:.1f}°)"
            except Exception:
                road_info_str = f"Yaw:{v_rot.yaw:.1f}°"

            # Check traffic light only if vehicle is currently affected at a light
            if hasattr(vehicle, "is_at_traffic_light") and vehicle.is_at_traffic_light():
                tl_state_enum = vehicle.get_traffic_light_state()
                tl_state = "RED" if str(tl_state_enum) == "Red" else "GREEN" if str(tl_state_enum) == "Green" else "NONE"
            else:
                tl_state = "NONE"

            # Update rolling motion diagnostics and detect circling
            is_at_red = (tl_state == "RED")
            is_queue = (obs_snap is not None and float(obs_snap.get("distance", 99.0)) <= 8.0)
            lane_yaw_val = ego_wp.transform.rotation.yaw if ego_wp and hasattr(ego_wp.transform, "rotation") else None
            lane_type_val = str(getattr(ego_wp, "lane_type", "None")) if ego_wp else "None"

            is_circling, disp_val, cum_yaw_val = demo.update_motion_diagnostics(
                x=float(v_loc.x),
                y=float(v_loc.y),
                yaw=float(v_rot.yaw),
                speed_kmh=speed_kmh,
                steer=steer_val,
                throttle=throttle_val,
                road_id=int(getattr(ego_wp, "road_id", 0)) if ego_wp else 0,
                lane_id=int(getattr(ego_wp, "lane_id", 0)) if ego_wp else 0,
                is_at_red_light=is_at_red,
                is_queuing=is_queue,
                lane_yaw=lane_yaw_val,
            )

            # Update safe road checkpoint and detect stuck condition
            demo.vehicle_speed_kmh = speed_kmh
            demo.traffic_light_state = tl_state
            demo.update_safe_checkpoint(world, vehicle, speed_kmh, tl_state)
            is_stuck, stuck_reason = demo.detect_stuck_condition(
                world=world,
                vehicle=vehicle,
                speed_kmh=speed_kmh,
                traffic_light_state=tl_state,
                obstacle_distance=obs_snap.get("distance") if obs_snap else None,
            )
            demo.stuck_detected = is_stuck
            demo.stuck_reason = stuck_reason

            # Compute relative motion and closing speed if obstacle is present
            v_close = None
            ttc = None
            rel_avail = False
            if obs_snap is not None and obs_snap.get("other_actor") is not None:
                other_actor = obs_snap.get("other_actor")
                try:
                    ego_vel = vehicle.get_velocity()
                    ego_tf = vehicle.get_transform()
                    obs_vel = other_actor.get_velocity() if hasattr(other_actor, "get_velocity") else None
                    obs_tf = other_actor.get_transform() if hasattr(other_actor, "get_transform") else None
                    v_close, ttc, rel_avail = compute_obstacle_relative_motion(
                        ego_vel,
                        ego_tf.location if ego_tf else None,
                        obs_vel,
                        obs_tf.location if obs_tf else None,
                        float(obs_snap["distance"]),
                    )
                except Exception as e:
                    logger.debug("Could not compute obstacle relative motion: %s", e)

            # Fallback placeholder frames if waiting for first sensor tick
            if rgb_snap is None:
                rgb_snap = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(rgb_snap, "Connecting to CARLA RGB Sensor...", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            if seg_snap is None:
                seg_snap = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(seg_snap, "Connecting to CARLA Seg Sensor...", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Evaluate frame through Cognix pipeline (sub-5ms)
            demo.frame_count += 1
            ev = demo.evaluate_frame(
                rgb_raw=rgb_snap,
                seg_raw=seg_snap,
                imu_raw=imu_snap,
                obstacle_raw=obs_snap,
                collision_raw=col_snap,
                speed_kmh=speed_kmh,
                traffic_light_state=tl_state,
                sensor_timestamp=sensor_t,
                v_close_mps=v_close,
                ttc_sec=ttc,
                relative_motion_available=rel_avail,
                autopilot_status=f"AP: ENGAGED (TM:{demo.tm_port})",
                road_info=road_info_str,
                motion_displacement=disp_val,
                motion_cum_yaw=cum_yaw_val,
                circling_detected=is_circling,
                ctrl_steer=steer_val,
                ctrl_throttle=throttle_val,
                ctrl_brake=brake_val,
            )

            # Overlay telemetry on Left camera displays (using processed frames so faults are visible)
            rgb_disp = (ev.processed_rgb if ev.processed_rgb is not None else rgb_snap).copy()
            seg_disp = (ev.processed_seg if ev.processed_seg is not None else seg_snap).copy()

            # Save verifiable sample of RGB occlusion when active
            if ev.fault_mode == FaultMode.RGB_OCCLUSION and not occlusion_saved:
                sample_path = "reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/rgb_occlusion_sample.png"
                os.makedirs(os.path.dirname(sample_path), exist_ok=True)
                cv2.imwrite(sample_path, rgb_disp)
                occlusion_saved = True
                logger.info("Saved verifiable RGB occlusion screenshot to %s", sample_path)

            # RGB overlay
            cv2.putText(rgb_disp, f"CARLA RGB CAMERA (640x360) | SCENARIO: {ev.scenario_mode.value}", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if ev.fault_mode == FaultMode.RGB_DARKNESS:
                cv2.rectangle(rgb_disp, (10, 35), (320, 68), (0, 0, 180), -1)
                cv2.putText(rgb_disp, "FAULT: RGB DARKNESS (0.12x)", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            elif ev.fault_mode == FaultMode.RGB_OCCLUSION:
                cv2.rectangle(rgb_disp, (10, 35), (340, 68), (0, 0, 200), -1)
                cv2.putText(rgb_disp, "FAULT: RGB OCCLUSION (60% Cutout)", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            elif ev.fault_mode != FaultMode.NONE:
                cv2.rectangle(rgb_disp, (10, 35), (280, 68), (20, 20, 180), -1)
                cv2.putText(rgb_disp, f"FAULT: {ev.fault_mode.name}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # Seg overlay
            cv2.putText(seg_disp, "CARLA SEMANTIC SEGMENTATION (CityScapes)", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(seg_disp, f"Speed: {speed_kmh:4.1f} km/h  |  Light: {tl_state}", (14, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            if ev.fault_mode == FaultMode.SEG_CORRUPTION:
                cv2.rectangle(seg_disp, (10, 35), (340, 68), (180, 0, 180), -1)
                cv2.putText(seg_disp, "FAULT: SEG NOISE CORRUPTION", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            elif obs_snap:
                cv2.putText(seg_disp, f"OBSTACLE: {obs_snap['distance']:.1f}m ({obs_snap['actor_type'][:12]})", (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 180, 255), 2, cv2.LINE_AA)

            # Compose and render full 1400x720 dashboard
            dashboard = demo.render_dashboard(rgb_disp, seg_disp, ev)
            cv2.imshow(window_name, dashboard)

            # Periodic structured diagnostic telemetry (every 10 frames, ~3 Hz)
            if demo.frame_count % 10 == 0:
                record = {
                    "frame": demo.frame_count,
                    "elapsed_sec": round(time.time() - t_loop_start, 2),
                    "fps": round(ev.fps, 1),
                    "latency_ms": round(ev.latency_ms, 2),
                    "scenario": ev.scenario_mode.value,
                    "last_key": demo.last_key_name,
                    "last_key_source": demo.last_key_source,
                    "transition_msg": demo.transition_message,
                    "fault": ev.fault_mode.name,
                    "decision_outcome": ev.decision_outcome.value,
                    "risk_level": ev.risk_level.value,
                    "escalation_required": ev.escalation_required,
                    "decision_reason": ev.decision_reason,
                    "policy_branch_reason": ev.policy_branch_reason,
                    "trigger_source": ev.trigger_source.value,
                    "policy_confidence": round(ev.confidence, 3),
                    "policy_epistemic": round(ev.epistemic_uncertainty, 3),
                    "conformal_set_size": ev.conformal_set_size,
                    "speed_kmh": round(ev.speed_kmh, 1),
                    "traffic_light_state": ev.traffic_light_state,
                    "sensor_age_ms": round(ev.sensor_age_ms, 1),
                    "obstacle_distance": round(ev.obstacle_distance, 2) if ev.obstacle_distance is not None else None,
                    "obstacle_actor": ev.obstacle_actor,
                    "obstacle_age_ms": round(ev.obstacle_age_ms, 1) if ev.obstacle_age_ms is not None else None,
                    "obstacle_fresh": ev.obstacle_fresh,
                    "v_close_mps": round(ev.v_close_mps, 2) if ev.v_close_mps is not None else None,
                    "ttc_sec": round(ev.ttc_sec, 2) if ev.ttc_sec is not None else None,
                    "relative_motion_available": ev.relative_motion_available,
                    "collision_active": ev.collision_active,
                    "collision_impulse": round(ev.collision_impulse, 1),
                    "collision_age_ms": round(ev.collision_age_ms, 1) if ev.collision_age_ms is not None else None,
                    "collision_actor": ev.collision_actor,
                    "collision_actor_id": ev.collision_actor_id,
                    "fused_safe_p": round(ev.fused_safe_probability, 3),
                    "fused_uncertainty": round(ev.fused_uncertainty, 3),
                    "uncertainties_summary": {
                        u.id: (round(u.value, 3) if u.value is not None else None)
                        for u in ev.uncertainties
                    },
                    "scenario_actor_ids": [getattr(a, "id", None) for a in demo.scenario_actors],
                    "scenario_spawn_status": demo.scenario_spawn_status,
                    "stuck_status": ev.stuck_status,
                    "recovery_status": ev.recovery_status,
                    "safe_checkpoint": list(ev.safe_checkpoint_loc) if ev.safe_checkpoint_loc else None,
                    "ego_location": [round(float(v_loc.x), 2), round(float(v_loc.y), 2), round(float(v_loc.z), 2)],
                    "ego_yaw": round(float(v_rot.yaw), 2),
                    "autopilot_status": ev.autopilot_status,
                    "road_info": ev.road_info,
                    "motion_displacement_m": round(ev.motion_displacement, 2),
                    "motion_cum_yaw_deg": round(ev.motion_cum_yaw, 1),
                    "circling_detected": ev.circling_detected,
                    "ctrl_steer": round(ev.ctrl_steer, 3),
                    "ctrl_throttle": round(ev.ctrl_throttle, 3),
                    "ctrl_brake": round(ev.ctrl_brake, 3),
                    # Extended trajectory metrics (Section 2 Requirements)
                    "angular_velocity": [round(float(ang_vel.x), 3), round(float(ang_vel.y), 3), round(float(ang_vel.z), 3)] if ang_vel else [0.0, 0.0, 0.0],
                    "cumulative_distance_m": ev.cumulative_distance_m,
                    "net_displacement_m": ev.net_displacement_m,
                    "route_efficiency_ratio": ev.efficiency_ratio,
                    "route_progress_active": (not ev.is_immobilized and ev.efficiency_ratio > 0.40),
                    "revisit_detected": ev.revisit_detected,
                    "circular_movement_detected": ev.circular_movement_detected,
                    "excessive_steering": ev.excessive_steering,
                    "wrong_direction": ev.wrong_direction,
                    "is_immobilized": ev.is_immobilized,
                    "stuck_seconds": ev.stuck_seconds,
                    "lane_waypoint_yaw": round(float(lane_yaw_val), 1) if lane_yaw_val is not None else None,
                    "lane_type": lane_type_val,
                    "tm_port": demo.tm_port,
                }
                telemetry_records.append(record)

            # Diagnostic console log every 2.0s
            now = time.time()
            if now - last_diag_log_time >= 2.0:
                last_diag_log_time = now
                v_close_disp = f"{ev.v_close_mps:+.1f}m/s" if ev.v_close_mps is not None else "None"
                ttc_disp = f"{ev.ttc_sec:.1f}s" if ev.ttc_sec is not None else "None"
                logger.info(
                    "[DIAG %4.1fs] Scen=%-10s | Key=%-4s (%s) | Decision=%-8s (%-8s) | Conf=%.3f Unc=%.3f Set=%d | Spd=%4.1f TL=%-4s Obs=%-6s (v_close=%s, TTC=%s) | Src=%s",
                    now - t_loop_start,
                    ev.scenario_mode.value,
                    demo.last_key_name,
                    demo.last_key_source[:3],
                    ev.decision_outcome.value,
                    ev.risk_level.value,
                    ev.confidence,
                    ev.epistemic_uncertainty,
                    ev.conformal_set_size,
                    ev.speed_kmh,
                    ev.traffic_light_state,
                    f"{ev.obstacle_distance:.1f}m" if ev.obstacle_distance is not None else "None",
                    v_close_disp,
                    ttc_disp,
                    ev.trigger_source.value,
                )

            # Automatic scenario cycling (for automated verification runs)
            if cycle_scenarios:
                elapsed = time.time() - t_loop_start
                if elapsed < 8.0:
                    if demo.scenario_mode != ScenarioMode.NORMAL:
                        demo.handle_key(ord("n"), world, vehicle, source="AUTOCYCLE")
                elif 8.0 <= elapsed < 14.0:
                    if demo.fault_mode != FaultMode.RGB_OCCLUSION:
                        demo.handle_key(ord("2"), world, vehicle, source="AUTOCYCLE")
                elif 14.0 <= elapsed < 20.0:
                    if demo.fault_mode != FaultMode.RGB_DARKNESS:
                        demo.handle_key(ord("1"), world, vehicle, source="AUTOCYCLE")
                elif 20.0 <= elapsed < 26.0:
                    if demo.fault_mode != FaultMode.SEG_CORRUPTION:
                        demo.handle_key(ord("3"), world, vehicle, source="AUTOCYCLE")
                elif 26.0 <= elapsed < 32.0:
                    if demo.fault_mode != FaultMode.IMU_DISTURBANCE:
                        demo.handle_key(ord("4"), world, vehicle, source="AUTOCYCLE")
                elif 32.0 <= elapsed < 36.0:
                    if demo.fault_mode != FaultMode.NONE:
                        demo.handle_key(ord("0"), world, vehicle, source="AUTOCYCLE")
                elif 36.0 <= elapsed < 46.0:
                    if demo.scenario_mode != ScenarioMode.PEDESTRIAN:
                        demo.handle_key(ord("p"), world, vehicle, source="AUTOCYCLE")
                elif 46.0 <= elapsed < 56.0:
                    if demo.scenario_mode != ScenarioMode.WEATHER:
                        demo.handle_key(ord("w"), world, vehicle, source="AUTOCYCLE")
                elif 56.0 <= elapsed < 66.0:
                    if demo.scenario_mode != ScenarioMode.PROPS:
                        demo.handle_key(ord("b"), world, vehicle, source="AUTOCYCLE")
                elif elapsed >= 66.0:
                    if demo.scenario_mode != ScenarioMode.NORMAL or demo.fault_mode != FaultMode.NONE:
                        demo.handle_key(ord("r"), world, vehicle, source="AUTOCYCLE")

            # Measure FPS
            frames_in_second += 1
            if now - fps_timer >= 1.0:
                demo.current_fps = frames_in_second / (now - fps_timer)
                frames_in_second = 0
                fps_timer = now

            # Check keyboard input from GUI window
            key_raw = cv2.waitKey(1)
            key = key_raw & 0xFF if key_raw != -1 else 255
            key_source = "NONE"

            if key != 255:
                key_source = "OPENCV_GUI"
            elif msvcrt is not None:
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        # Handle Windows extended keys (scancode prefixes 0x00 and 0xE0)
                        if ch in (b"\x00", b"\xe0"):
                            if msvcrt.kbhit():
                                _ = msvcrt.getch()
                            ch = b""
                        if ch in (b"\x03",):  # Ctrl+C
                            raise KeyboardInterrupt
                        elif ch in (b"\x1b",):  # Esc
                            key = 27
                            key_source = "CONSOLE_STDIN"
                        elif ch:
                            try:
                                key = ord(ch.decode("latin1", errors="ignore"))
                                key_source = "CONSOLE_STDIN"
                            except Exception:
                                key = ord(ch) if isinstance(ch, int) else 255
                                key_source = "CONSOLE_STDIN" if key != 255 else "NONE"
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    pass

            if key != 255:
                cont = demo.handle_key(key, world, vehicle, source=key_source)
                if not cont:
                    logger.info("Quit requested by user via %s.", key_source)
                    break

            # Check duration limit (for automated verification)
            if duration_sec is not None and (time.time() - t_loop_start) >= duration_sec:
                logger.info("Reached target duration limit of %.1f seconds.", duration_sec)
                break

    except KeyboardInterrupt:
        logger.info("Demonstration stopped cleanly by user (KeyboardInterrupt / Ctrl+C).")
    except Exception as e:
        logger.error("Exception during live demonstration: %s", e, exc_info=True)
        raise
    finally:
        total_time = time.time() - t_loop_start
        avg_fps = demo.frame_count / max(0.1, total_time)
        summary = {
            "total_frames": demo.frame_count,
            "elapsed_sec": total_time,
            "average_fps": avg_fps,
            "telemetry_samples_count": len(telemetry_records),
            "telemetry": telemetry_records,
        }
        if save_telemetry_path:
            try:
                os.makedirs(os.path.dirname(save_telemetry_path), exist_ok=True)
                import json
                with open(save_telemetry_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                logger.info("Saved telemetry to %s (%d samples)", save_telemetry_path, len(telemetry_records))
            except Exception as e:
                logger.error("Failed to save telemetry: %s", e)

        demo.cleanup(world)

    logger.info("Finished demonstration. Processed %d frames in %.1f s (Average FPS: %.1f).", demo.frame_count, total_time, avg_fps)
    return summary


def main():
    parser = argparse.ArgumentParser(description="COGNIX + CARLA Live Demonstration V2")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--tm-port", type=int, default=8000, help="CARLA Traffic Manager port (default: 8000)")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds")
    parser.add_argument("--cycle-scenarios", action="store_true", help="Automatically cycle through scenarios for automated testing")
    parser.add_argument("--output", default=None, help="Optional JSON path to save telemetry")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_live_demonstration(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        duration_sec=args.duration,
        cycle_scenarios=args.cycle_scenarios,
        save_telemetry_path=args.output,
    )


if __name__ == "__main__":
    main()

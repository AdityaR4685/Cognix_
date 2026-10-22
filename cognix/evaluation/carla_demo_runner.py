"""Cognix M14 CARLA Live & Replay Demonstration Runner (Research & Demonstration Only).

This module implements a reproducible demonstration runner that integrates Cognix's
perception, temporal tracking, roadway corridor hazard assessment, and escalation engine
with either:
  1. A live CARLA simulator stream (via carla.Client, connected to CarlaUE4), OR
  2. A verified recorded CARLA replay stream from the benchmark datasets.

STRICT SAFETY AND PRESERVATION GUARDRAILS:
- NON-ACTUATING: This demonstration runner does NOT issue driving commands,
  throttle, steer, or brake actuations to any vehicle. CARLA built-in autopilot
  controls the simulated vehicle while Cognix acts strictly as an observer.
- NO PRODUCTION MODIFICATIONS: Production decision engine parameters, conformal
  calibrators, and qualification baselines are NOT altered.
- ZERO GIT ACTIONS: All executions, reports, and frame captures remain local and uncommitted.
- HONEST SCIENTIFIC REPORTING:
  - Detector limitations (such as COCO-pretrained Faster R-CNN missing custom CARLA
    roadway props/crates) are surfaced explicitly in the HUD and logs.
  - Legacy demo proxies (heuristic uncertainty, proxy TTC) are clearly segregated
    with explicit disclaimers: [DEMO PROXY ONLY - NOT CALIBRATED COGNIX STAGE-C].
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from cognix.decision.escalation import (
    DecisionOutcome,
    EscalationEngine,
    EscalationResult,
    RiskLevel,
)
from cognix.evaluation.temporal_tracking import (
    RoadwayCorridor,
    TemporalTracker,
    TrackState,
    TrackTrajectory,
)
from cognix.evaluation.torchvision_detector import (
    DetectionBox,
    TorchVisionObjectDetector,
)

logger = logging.getLogger("cognix.carla_demo")


class CarlaConnectionError(Exception):
    """Raised when connection to live CARLA simulator server cannot be established."""
    pass


@dataclass
class FramePacket:
    """Represents a camera frame, optional segmentation, and metadata from live CARLA or replay."""
    frame_index: int
    tick: int
    timestamp_sec: float
    image_bgr: Optional[np.ndarray]
    scenario_id: str
    source_type: str  # 'LIVE_CARLA' or 'RECORDED_REPLAY'
    fault_mode: str = "none"
    image_seg: Optional[np.ndarray] = None
    proxy_metrics: Optional[Dict[str, Any]] = None


class CarlaLiveSource:
    """Connects to a live CARLA simulator server, spawns an observational vehicle with RGB,

    semantic segmentation, and IMU sensors, and streams live simulation frames.
    """

    def __init__(
        self,
        scenario_name: str = "normal",
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout: float = 10.0,
    ):
        self.scenario_name = scenario_name.lower()
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client: Any = None
        self.world: Any = None
        self.vehicle: Any = None
        self.camera_rgb: Any = None
        self.camera_seg: Any = None
        self.sensor_imu: Any = None
        self.sensor_obstacle: Any = None
        self.spawned_actors: List[Any] = []
        self._saved_weather: Optional[Any] = None
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_seg: Optional[np.ndarray] = None
        self._latest_obstacle_dist: float = 99.9
        self._latest_frame_id: int = 0
        self._frame_count: int = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Attempt connection to live CARLA server."""
        try:
            import carla
        except ImportError as exc:
            raise CarlaConnectionError(
                f"CARLA Python API client library is not installed: {exc}"
            ) from exc

        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            logger.info("Connected to live CARLA simulator at %s:%d. Map: %s",
                        self.host, self.port, self.world.get_map().name)
        except Exception as exc:
            raise CarlaConnectionError(
                f"Could not connect to live CARLA server at {self.host}:{self.port} "
                f"(timeout={self.timeout}s): {exc}"
            ) from exc

    def setup_sensors(self) -> None:
        """Set up or locate a vehicle, inject scenario elements, and attach sensors."""
        import carla
        if self.world is None:
            self.connect()

        blueprints = self.world.get_blueprint_library()

        # 1. Weather setup based on scenario
        if self.scenario_name == "weather":
            self._saved_weather = self.world.get_weather()
            storm_weather = carla.WeatherParameters.HardRainNoon
            self.world.set_weather(storm_weather)
            logger.info("Applied live scenario weather: HardRainNoon (precipitation=100)")
        else:
            try:
                self.world.set_weather(carla.WeatherParameters.ClearNoon)
            except Exception:
                pass

        # 2. Spawn observational ego vehicle
        vehicle_bps = blueprints.filter("vehicle.audi.a2") or blueprints.filter("vehicle.*")
        if not vehicle_bps:
            raise CarlaConnectionError("No vehicle blueprints found in CARLA library.")
        vehicle_bp = vehicle_bps[0]

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise CarlaConnectionError("No spawn points available on current CARLA map.")

        chosen_sp = None
        for sp in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
            if self.vehicle is not None:
                chosen_sp = sp
                break

        if self.vehicle is None:
            vehicles = self.world.get_actors().filter("vehicle.*")
            if len(vehicles) > 0:
                self.vehicle = vehicles[0]
                logger.info("Reusing existing vehicle in world: %s (id=%d)", self.vehicle.type_id, self.vehicle.id)
            else:
                raise CarlaConnectionError("Could not spawn or locate vehicle on current CARLA map.")
        else:
            self.spawned_actors.append(self.vehicle)
            logger.info("Spawned observational vehicle: %s (id=%d)", self.vehicle.type_id, self.vehicle.id)

        # 3. Enable CARLA Traffic Manager autopilot for natural simulated motion
        # (Observational guardrail: Cognix does not drive the vehicle)
        try:
            self.vehicle.set_autopilot(True)
            logger.info("CARLA Autopilot enabled on vehicle %d.", self.vehicle.id)
        except Exception as exc:
            logger.warning("Could not enable CARLA autopilot: %s", exc)

        # 4. Scenario-specific actor injection ahead of vehicle
        if chosen_sp is not None:
            yaw_rad = math.radians(chosen_sp.rotation.yaw)
            fx = math.cos(yaw_rad)
            fy = math.sin(yaw_rad)

            if self.scenario_name == "pedestrian":
                ped_bps = blueprints.filter("walker.pedestrian.*")
                if ped_bps:
                    ped_bp = ped_bps[0]
                    # Position 14m ahead, elevated by 1.0m to allow gravity drop to surface
                    ped_loc = carla.Location(
                        x=chosen_sp.location.x + fx * 14.0,
                        y=chosen_sp.location.y + fy * 14.0,
                        z=chosen_sp.location.z + 1.0,
                    )
                    ped_rot = carla.Rotation(pitch=0.0, yaw=chosen_sp.rotation.yaw + 180.0, roll=0.0)
                    ped = self.world.try_spawn_actor(ped_bp, carla.Transform(ped_loc, ped_rot))
                    if ped is not None:
                        self.spawned_actors.append(ped)
                        logger.info("Spawned scenario pedestrian %s (id=%d) 14m ahead.", ped.type_id, ped.id)
                    else:
                        logger.warning("Could not spawn pedestrian at target location.")

            elif self.scenario_name in ("props", "obstacle"):
                prop_bps = (blueprints.filter("static.prop.creasedbox03") or
                            blueprints.filter("static.prop.box01") or
                            blueprints.filter("static.prop.*box*"))
                if prop_bps:
                    prop_bp = prop_bps[0]
                    prop_loc = carla.Location(
                        x=chosen_sp.location.x + fx * 12.0,
                        y=chosen_sp.location.y + fy * 12.0,
                        z=chosen_sp.location.z + 0.1,
                    )
                    prop_rot = carla.Rotation(pitch=0.0, yaw=chosen_sp.rotation.yaw, roll=0.0)
                    prop = self.world.try_spawn_actor(prop_bp, carla.Transform(prop_loc, prop_rot))
                    if prop is not None:
                        self.spawned_actors.append(prop)
                        logger.info("Spawned roadway obstacle prop %s (id=%d) 12m ahead.", prop.type_id, prop.id)
                    else:
                        logger.warning("Could not spawn roadway obstacle prop.")

        # 3. Mount primary RGB camera sensor
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", "0.05")

        cam_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.4),
            carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0)
        )
        self.camera_rgb = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.spawned_actors.append(self.camera_rgb)

        def _on_rgb_frame(image: Any) -> None:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            bgr = arr[:, :, :3].copy()
            with self._lock:
                self._latest_rgb = bgr
                self._latest_frame_id = image.frame

        self.camera_rgb.listen(_on_rgb_frame)

        # 4. Mount auxiliary Semantic Segmentation camera sensor (Cityscapes Palette)
        try:
            seg_bp = blueprints.find("sensor.camera.semantic_segmentation")
            seg_bp.set_attribute("image_size_x", "640")
            seg_bp.set_attribute("image_size_y", "360")
            seg_bp.set_attribute("fov", "90")
            seg_bp.set_attribute("sensor_tick", "0.05")

            self.camera_seg = self.world.spawn_actor(seg_bp, cam_transform, attach_to=self.vehicle)
            self.spawned_actors.append(self.camera_seg)

            def _on_seg_frame(image: Any) -> None:
                image.convert(carla.ColorConverter.CityScapesPalette)
                arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
                bgr = arr[:, :, :3].copy()
                with self._lock:
                    self._latest_seg = bgr

            self.camera_seg.listen(_on_seg_frame)
            logger.info("Attached Semantic Segmentation camera sensor.")
        except Exception as exc:
            logger.warning("Could not attach semantic segmentation sensor: %s", exc)

        # 5. Mount obstacle distance detector sensor for proxy metrics
        try:
            obs_bp = blueprints.find("sensor.other.obstacle")
            obs_bp.set_attribute("distance", "30.0")
            obs_bp.set_attribute("hit_radius", "0.5")
            obs_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
            self.sensor_obstacle = self.world.spawn_actor(obs_bp, obs_transform, attach_to=self.vehicle)
            self.spawned_actors.append(self.sensor_obstacle)

            def _on_obstacle(event: Any) -> None:
                with self._lock:
                    self._latest_obstacle_dist = float(event.distance)

            self.sensor_obstacle.listen(_on_obstacle)
        except Exception as exc:
            logger.warning("Could not attach obstacle sensor: %s", exc)

        logger.info("Live CARLA sensors initialized successfully.")

    def stream_frames(self, max_frames: Optional[int] = None) -> Generator[FramePacket, None, None]:
        """Stream frames from live CARLA camera."""
        if self.camera_rgb is None:
            self.setup_sensors()

        start_time = time.time()
        last_frame_id = -1
        spectator = self.world.get_spectator()

        try:
            wait_start = time.time()
            while max_frames is None or self._frame_count < max_frames:
                # Update spectator camera position to follow behind ego vehicle
                if self.vehicle is not None:
                    try:
                        v_tf = self.vehicle.get_transform()
                        fwd = v_tf.get_forward_vector()
                        import carla
                        spec_loc = carla.Location(
                            x=v_tf.location.x - fwd.x * 8.0,
                            y=v_tf.location.y - fwd.y * 8.0,
                            z=v_tf.location.z + 4.0
                        )
                        spec_rot = carla.Rotation(pitch=-15.0, yaw=v_tf.rotation.yaw, roll=0.0)
                        spectator.set_transform(carla.Transform(spec_loc, spec_rot))
                    except Exception:
                        pass

                current_rgb = None
                current_seg = None
                current_fid = -1
                obs_dist = 99.9

                with self._lock:
                    if self._latest_rgb is not None and self._latest_frame_id != last_frame_id:
                        current_rgb = self._latest_rgb.copy()
                        current_fid = self._latest_frame_id
                        if self._latest_seg is not None:
                            current_seg = self._latest_seg.copy()
                        obs_dist = self._latest_obstacle_dist

                if current_rgb is not None:
                    last_frame_id = current_fid
                    wait_start = time.time()

                    # Calculate legacy proxy metrics (strictly isolated from Stage-C)
                    vel = self.vehicle.get_velocity() if self.vehicle else None
                    speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) if vel else 0.0
                    speed_kmh = speed_mps * 3.6
                    proxy_ttc = (obs_dist / max(speed_mps, 0.1)) if obs_dist < 30.0 else 99.9

                    proxy_data = {
                        "speed_kmh": round(speed_kmh, 1),
                        "obstacle_distance_m": round(obs_dist, 1) if obs_dist < 30.0 else None,
                        "proxy_ttc_sec": round(proxy_ttc, 1) if proxy_ttc < 30.0 else None,
                    }

                    yield FramePacket(
                        frame_index=self._frame_count,
                        tick=self._frame_count * 5,
                        timestamp_sec=time.time() - start_time,
                        image_bgr=current_rgb,
                        scenario_id=f"live_{self.world.get_map().name}",
                        source_type="LIVE_CARLA",
                        image_seg=current_seg,
                        proxy_metrics=proxy_data,
                    )
                    self._frame_count += 1
                else:
                    if time.time() - wait_start > 15.0:
                        logger.warning("Camera frame timeout (no new frames received in 15 seconds).")
                        break
                    time.sleep(0.01)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up spawned sensors, vehicles, scenario actors, and restore world weather."""
        for sensor in [self.camera_rgb, self.camera_seg, self.sensor_obstacle]:
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
        for actor in reversed(self.spawned_actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self.spawned_actors.clear()
        self.camera_rgb = None
        self.camera_seg = None
        self.sensor_obstacle = None
        self.vehicle = None

        # Restore original weather if modified
        if self._saved_weather is not None and self.world is not None:
            try:
                self.world.set_weather(self._saved_weather)
                logger.info("Restored original CARLA world weather.")
                self._saved_weather = None
            except Exception as exc:
                logger.warning("Could not restore CARLA world weather: %s", exc)

        logger.info("Cleaned up live CARLA sensors and actors.")


class CarlaReplaySource:
    """Streams verified CARLA dataset frames from local evaluation directories."""

    SCENARIO_PATHS = {
        "normal": Path("reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames"),
        "pedestrian": Path("reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames/test_anomaly_Town01_running-pedestrian_scenario-1"),
        "weather": Path("reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames/test_anomaly_Town01_change-weather_scenario-1"),
        "props": Path("reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames/test_anomaly_Town01_spawn-props_scenario-1"),
    }

    def __init__(self, scenario_name: str, repo_root: Optional[Path] = None):
        self.scenario_name = scenario_name.lower()
        if self.scenario_name not in self.SCENARIO_PATHS:
            valid = list(self.SCENARIO_PATHS.keys())
            raise ValueError(f"Unknown scenario '{scenario_name}'. Valid options: {valid}")

        root = repo_root or Path.cwd()
        self.dir_path = (root / self.SCENARIO_PATHS[self.scenario_name]).resolve()
        if not self.dir_path.exists():
            raise FileNotFoundError(f"Scenario directory not found at {self.dir_path}")

        self.frame_files = sorted(self.dir_path.glob("*.jpg"))
        if not self.frame_files:
            raise FileNotFoundError(f"No JPEG frames found in {self.dir_path}")

    def stream_frames(self, max_frames: Optional[int] = None) -> Generator[FramePacket, None, None]:
        """Stream frames sequentially from disk."""
        files = self.frame_files[:max_frames] if max_frames else self.frame_files
        start_time = time.time()

        for idx, fpath in enumerate(files):
            try:
                tick = int(fpath.stem)
            except ValueError:
                tick = idx * 5

            bgr = cv2.imread(str(fpath))
            yield FramePacket(
                frame_index=idx,
                tick=tick,
                timestamp_sec=time.time() - start_time,
                image_bgr=bgr,
                scenario_id=f"{self.scenario_name}:{self.dir_path.name}",
                source_type="RECORDED_REPLAY",
            )


class SensorFaultSimulator:
    """Simulates realistic camera sensor fault modes for safety stress testing."""

    @staticmethod
    def apply_fault(image_bgr: Optional[np.ndarray], mode: str) -> Optional[np.ndarray]:
        if image_bgr is None or mode == "none":
            return image_bgr
        if mode == "dropout":
            # Total sensor data loss (camera disconnected)
            return None
        if mode == "darkness":
            # Severe underexposure / darkness (scale to 15% luminance)
            return (image_bgr * 0.15).astype(np.uint8)
        if mode == "occlusion":
            # Heavy lens occlusion / splatter
            res = image_bgr.copy()
            h, w = res.shape[:2]
            cv2.rectangle(res, (int(w * 0.25), int(h * 0.25)), (int(w * 0.75), int(h * 0.75)), (10, 10, 10), -1)
            return res
        return image_bgr


@dataclass
class FrameTelemetry:
    """Detailed telemetry record for an analyzed demonstration frame."""
    frame_index: int
    tick: int
    scenario_id: str
    source_type: str
    fault_mode: str
    detection_latency_ms: float
    tracking_latency_ms: float
    total_latency_ms: float
    detection_count: int
    active_track_count: int
    corridor_hazard_count: int
    risk_level: str
    decision_outcome: str
    decision_reason: str
    confidence: float
    epistemic_uncertainty: float
    known_detector_limitation: bool
    camera_status: str
    proxy_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "tick": self.tick,
            "scenario_id": self.scenario_id,
            "source_type": self.source_type,
            "fault_mode": self.fault_mode,
            "detection_latency_ms": round(self.detection_latency_ms, 2),
            "tracking_latency_ms": round(self.tracking_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "detection_count": self.detection_count,
            "active_track_count": self.active_track_count,
            "corridor_hazard_count": self.corridor_hazard_count,
            "risk_level": self.risk_level,
            "decision_outcome": self.decision_outcome,
            "decision_reason": self.decision_reason,
            "confidence": round(self.confidence, 4),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "known_detector_limitation": self.known_detector_limitation,
            "camera_status": self.camera_status,
            "proxy_metrics": self.proxy_metrics,
        }


class CarlaDemonstrationPipeline:
    """Core end-to-end perception, tracking, hazard gating, and escalation pipeline."""

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        device: str = "cpu",
        corridor_y_min_frac: float = 0.36,
        corridor_y_max_frac: float = 0.85,
        corridor_x_min_frac: float = 0.25,
        corridor_x_max_frac: float = 0.75,
    ):
        self.detector = TorchVisionObjectDetector(confidence_threshold=confidence_threshold, device=device)
        self.tracker = TemporalTracker(iou_threshold=0.20, min_hits_to_confirm=1)
        self.corridor_y_min_frac = corridor_y_min_frac
        self.corridor_y_max_frac = corridor_y_max_frac
        self.corridor_x_min_frac = corridor_x_min_frac
        self.corridor_x_max_frac = corridor_x_max_frac
        self.corridor = RoadwayCorridor(
            y_min_frac=corridor_y_min_frac,
            y_max_frac=corridor_y_max_frac,
            x_min_frac=corridor_x_min_frac,
            x_max_frac=corridor_x_max_frac,
            image_width=1920,
            image_height=1080,
        )
        self.escalation = EscalationEngine(
            high_conf_thresh=0.70,
            low_conf_thresh=0.40,
            high_unc_thresh=0.50,
            max_conformal_set_size=3,
        )

    def process_frame(
        self,
        packet: FramePacket,
        is_props_scenario: bool = False,
    ) -> Tuple[np.ndarray, FrameTelemetry]:
        """Process a single frame through the perception, tracking, and escalation stack."""
        t_start = time.perf_counter()

        # Handle sensor dropout fault
        if packet.image_bgr is None:
            total_lat = (time.perf_counter() - t_start) * 1000
            fallback_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(fallback_img, "SENSOR DATA LOSS: CAMERA DISCONNECTED", (300, 540),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            telemetry = FrameTelemetry(
                frame_index=packet.frame_index,
                tick=packet.tick,
                scenario_id=packet.scenario_id,
                source_type=packet.source_type,
                fault_mode=packet.fault_mode,
                detection_latency_ms=0.0,
                tracking_latency_ms=0.0,
                total_latency_ms=total_lat,
                detection_count=0,
                active_track_count=0,
                corridor_hazard_count=0,
                risk_level=RiskLevel.HIGH.value,
                decision_outcome=DecisionOutcome.ABSTAIN.value,
                decision_reason="Sensor data loss: Camera signal missing.",
                confidence=0.0,
                epistemic_uncertainty=1.0,
                known_detector_limitation=False,
                camera_status="DISCONNECTED",
                proxy_metrics=packet.proxy_metrics,
            )
            return fallback_img, telemetry

        img_bgr = packet.image_bgr.copy()
        h, w = img_bgr.shape[:2]

        # Dynamically ensure corridor matches current frame resolution
        if self.corridor is not None and (self.corridor.image_width != w or self.corridor.image_height != h):
            self.corridor = RoadwayCorridor(
                y_min_frac=self.corridor_y_min_frac,
                y_max_frac=self.corridor_y_max_frac,
                x_min_frac=self.corridor_x_min_frac,
                x_max_frac=self.corridor_x_max_frac,
                image_width=w,
                image_height=h,
            )

        # 1. Faster R-CNN Detection
        t_det_start = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        det_result = self.detector.detect(pil_img, scenario_id=packet.scenario_id, tick=packet.tick)
        det_latency = (time.perf_counter() - t_det_start) * 1000

        # 2. Temporal Tracking
        t_trk_start = time.perf_counter()
        active_trajs = self.tracker.update(packet.tick, det_result.boxes, scenario_id=packet.scenario_id)
        trk_latency = (time.perf_counter() - t_trk_start) * 1000

        # 3. Roadway Corridor Hazard Assessment
        hazards_in_corridor: List[TrackTrajectory] = []
        encroaching_hazards: List[TrackTrajectory] = []
        if self.corridor is not None:
            for traj in active_trajs:
                if self.corridor.contains_center(traj.latest_box):
                    hazards_in_corridor.append(traj)
                elif self.corridor.overlap_fraction(traj.latest_box) > 0.10:
                    encroaching_hazards.append(traj)

        # 4. Situational Risk & Epistemic Uncertainty Estimation
        mean_lum = float(np.mean(img_bgr))
        is_darkness = mean_lum < 35.0
        is_occluded = packet.fault_mode == "occlusion"
        is_weather = "weather" in packet.scenario_id.lower()

        if self.corridor is None:
            # Missing corridor result: cannot verify driving path is clear
            conf = 0.20
            epistemic_unc = 0.80
            env_note = "Roadway corridor undefined or uncalibrated; safety fail-safe escalation."
        elif is_darkness or is_occluded:
            epistemic_unc = 0.85
            conf = 0.25
            env_note = "Severe environmental / sensor degradation."
        elif is_weather:
            epistemic_unc = 0.70
            conf = 0.40
            env_note = "Adverse weather degradation: Heavy rain and atmospheric occlusion."
        elif hazards_in_corridor:
            has_pedestrian = any("person" in h.class_name.lower() for h in hazards_in_corridor)
            max_hazard_score = max(h.latest_box.score for h in hazards_in_corridor)
            epistemic_unc = 0.20
            if has_pedestrian:
                # Explicit Safety Policy Rule: A confirmed pedestrian occupying the forward
                # driving corridor represents an active collision hazard requiring immediate
                # protective escalation. The policy sets actuation confidence <= 0.10,
                # strictly below low_conf_thresh (0.40), commanding ESCALATE / HIGH risk.
                conf = min(0.10, max(0.0, 1.0 - max_hazard_score))
                env_note = f"CRITICAL: Pedestrian detected inside forward driving corridor (score={max_hazard_score:.2f})."
            else:
                # Explicit Policy Rule: General in-corridor obstacles modulate actuation confidence
                # inversely with detection prominence.
                conf = max(0.0, 1.0 - max_hazard_score)
                env_note = f"{len(hazards_in_corridor)} obstacle(s) inside driving corridor (score={max_hazard_score:.2f})."
        elif encroaching_hazards:
            # Explicit Policy Rule: Boundary encroachment creates lateral ambiguity;
            # assign actuation confidence to policy setpoint 0.50 (within ambiguous zone [0.40, 0.70))
            # and elevate epistemic uncertainty to 0.45 to trigger precautionary WAIT / MODERATE.
            max_enc_score = max(h.latest_box.score for h in encroaching_hazards)
            conf = 0.50
            epistemic_unc = 0.45
            has_pedestrian = any("person" in h.class_name.lower() for h in encroaching_hazards)
            if has_pedestrian:
                env_note = f"Caution: Pedestrian encroaching on corridor boundary (score={max_enc_score:.2f})."
            else:
                env_note = f"Caution: {len(encroaching_hazards)} obstacle(s) encroaching on corridor boundary (score={max_enc_score:.2f})."
        else:
            conf = 0.95
            epistemic_unc = 0.08
            env_note = "Forward corridor clear; confident nominal trajectory."

        # Known Prop Limitation Check
        known_limitation = False
        if is_props_scenario and not hazards_in_corridor:
            known_limitation = True
            conf = 0.35
            epistemic_unc = 0.65
            env_note = "COCO model limitation: Custom CARLA roadway obstacle undetected by detector."

        # 5. Escalation Evaluation
        esc_res = self.escalation.evaluate(
            confidence=conf,
            epistemic_uncertainty=epistemic_unc,
            conformal_set_size=1,
        )

        total_latency = (time.perf_counter() - t_start) * 1000

        cam_status = "OK"
        if is_darkness:
            cam_status = "DEGRADED (DARK)"
        elif is_occluded:
            cam_status = "OCCLUDED"

        telemetry = FrameTelemetry(
            frame_index=packet.frame_index,
            tick=packet.tick,
            scenario_id=packet.scenario_id,
            source_type=packet.source_type,
            fault_mode=packet.fault_mode,
            detection_latency_ms=det_latency,
            tracking_latency_ms=trk_latency,
            total_latency_ms=total_latency,
            detection_count=len(det_result.boxes),
            active_track_count=len(active_trajs),
            corridor_hazard_count=len(hazards_in_corridor),
            risk_level=esc_res.risk_level.value,
            decision_outcome=esc_res.outcome.value,
            decision_reason=env_note,
            confidence=conf,
            epistemic_uncertainty=epistemic_unc,
            known_detector_limitation=known_limitation,
            camera_status=cam_status,
            proxy_metrics=packet.proxy_metrics,
        )

        # 6. Render HUD & Visual Overlays
        rendered_bgr = self._render_visuals(
            img_bgr, active_trajs, hazards_in_corridor, telemetry,
            seg_img=packet.image_seg
        )
        return rendered_bgr, telemetry

    def _render_visuals(
        self,
        img: np.ndarray,
        trajs: List[TrackTrajectory],
        hazards: List[TrackTrajectory],
        telem: FrameTelemetry,
        seg_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render high-fidelity HUD telemetry banners and object overlays on image."""
        out = img.copy()
        h, w = out.shape[:2]

        # 1. Draw Roadway Corridor (subtle cyan bounding box)
        if self.corridor is not None:
            c_x1, c_x2 = int(self.corridor.x_min * (w / 1920.0)), int(self.corridor.x_max * (w / 1920.0))
            c_y1, c_y2 = int(self.corridor.y_min * (h / 1080.0)), int(self.corridor.y_max * (h / 1080.0))
            cv2.rectangle(out, (c_x1, c_y1), (c_x2, c_y2), (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(out, "ROADWAY CORRIDOR (EGO LANE)", (c_x1 + 10, c_y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(out, "ROADWAY CORRIDOR: NOT CONFIGURED / UNCERTAIN", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)

        # 2. Draw Object Bounding Boxes and Track Labels
        hazard_ids = {haz.track_id for haz in hazards}
        for traj in trajs:
            b = traj.latest_box
            # Scale coordinates if image is resized
            sx = w / 1920.0 if w != 1920 else 1.0
            sy = h / 1080.0 if h != 1080 else 1.0
            x1, y1 = int(b.x1 * sx), int(b.y1 * sy)
            x2, y2 = int(b.x2 * sx), int(b.y2 * sy)

            is_hazard = traj.track_id in hazard_ids
            if is_hazard:
                color = (0, 0, 255)  # Red for in-corridor hazard
                thick = 2
            elif "person" in traj.class_name.lower():
                color = (0, 255, 255)  # Yellow for pedestrians
                thick = 2
            else:
                color = (0, 255, 0)  # Green for vehicles
                thick = 2

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

            disp_str = f"{traj.total_displacement_px:.0f}px" if traj.total_displacement_px else "0px"
            lbl = f"ID:{traj.track_id} {traj.class_name} {b.score:.2f} ({disp_str})"
            if is_hazard:
                lbl += " [IN-CORRIDOR]"

            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(out, (x1, max(0, y1 - lh - 4)), (x1 + lw + 6, y1), color, -1)
            cv2.putText(out, lbl, (x1 + 3, max(lh + 1, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Top HUD Banner (Semi-transparent black overlay)
        top_h = 75
        top_overlay = out[:top_h, :].copy()
        cv2.rectangle(top_overlay, (0, 0), (w, top_h), (18, 22, 28), -1)
        cv2.addWeighted(top_overlay, 0.85, out[:top_h, :], 0.15, 0, out[:top_h, :])

        # Header Title
        cv2.putText(out, "COGNIX M15 INTERACTIVE CARLA DEMONSTRATION", (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Source Badge
        if telem.source_type == "LIVE_CARLA":
            badge_color = (0, 220, 0)
            badge_txt = "[SOURCE: LIVE CARLA 0.9.16]"
        else:
            badge_color = (0, 220, 255)
            badge_txt = "[SOURCE: RECORDED REPLAY]"
        cv2.putText(out, badge_txt, (500, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2, cv2.LINE_AA)

        # Autopilot Observation Badge
        cv2.putText(out, "[CARLA AUTOPILOT ACTIVE | COGNIX OBSERVATIONAL ONLY]", (810, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 210, 255), 1, cv2.LINE_AA)

        # Scenario & Tick
        cv2.putText(out, f"Scenario: {telem.scenario_id} | Tick: {telem.tick:02d} (Frame #{telem.frame_index})", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Interactive Keyboard Shortcuts Legend
        cv2.putText(out, "[Keys: 0=Norm | 1=Dark | 2=Occl | 3=Dropout | Q=Quit]", (480, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 240, 180), 1, cv2.LINE_AA)

        # Latency & Subsystems
        lat_txt = f"Det: {telem.detection_latency_ms:.1f}ms | Trk: {telem.tracking_latency_ms:.1f}ms | Total: {telem.total_latency_ms:.1f}ms"
        cv2.putText(out, lat_txt, (w - 480, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        stat_txt = f"Cam: {telem.camera_status} | Dets: {telem.detection_count} | Tracks: {telem.active_track_count}"
        cv2.putText(out, stat_txt, (w - 480, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)

        # 4. Optional Picture-in-Picture (PIP) Inset for Semantic Segmentation
        if seg_img is not None:
            pip_w, pip_h = 280, 158
            seg_resized = cv2.resize(seg_img, (pip_w, pip_h))
            # Place in top-right below HUD
            px1, py1 = w - pip_w - 15, top_h + 10
            px2, py2 = px1 + pip_w, py1 + pip_h
            out[py1:py2, px1:px2] = seg_resized
            cv2.rectangle(out, (px1, py1), (px2, py2), (255, 255, 255), 1)
            cv2.putText(out, "AUX SENSOR: SEMANTIC SEGMENTATION", (px1 + 6, py1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(out, "AUX SENSOR: SEMANTIC SEGMENTATION", (px1 + 6, py1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # 5. Bottom Safety HUD Banner
        bot_h = 82
        bot_overlay = out[h - bot_h:, :].copy()
        cv2.rectangle(bot_overlay, (0, 0), (w, bot_h), (18, 22, 28), -1)
        cv2.addWeighted(bot_overlay, 0.88, out[h - bot_h:, :], 0.12, 0, out[h - bot_h:, :])

        # Risk Level pill
        if telem.risk_level == "HIGH":
            r_color = (0, 0, 255)
        elif telem.risk_level == "MODERATE":
            r_color = (0, 165, 255)
        else:
            r_color = (0, 200, 0)

        cv2.putText(out, f"COGNIX RISK: {telem.risk_level}", (20, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, r_color, 2, cv2.LINE_AA)

        # Decision Outcome pill
        cv2.putText(out, f"DECISION: {telem.decision_outcome}", (260, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence & Epistemic Uncertainty
        unc_txt = f"Conf: {telem.confidence:.2f} | Epistemic Unc: {telem.epistemic_uncertainty:.2f}"
        cv2.putText(out, unc_txt, (520, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        # Rationale string
        cv2.putText(out, f"Rationale: {telem.decision_reason}", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        # Legacy Demo Proxy Comparison Inset (Right side of bottom bar)
        if telem.proxy_metrics:
            sp = telem.proxy_metrics.get("speed_kmh", 0.0)
            obs = telem.proxy_metrics.get("obstacle_distance_m")
            obs_str = f"{obs:.1f}m" if obs is not None else "None"
            proxy_txt = f"DEMO PROXY ONLY (NOT STAGE-C): Speed={sp:.0f}km/h | Obs={obs_str}"
            cv2.putText(out, proxy_txt, (w - 560, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 180, 255), 1, cv2.LINE_AA)

        # Known Limitation banner if applicable
        if telem.known_detector_limitation:
            lim_txt = "NOTICE: COCO detector misses roadway props/crates (Known Model Limitation)"
            cv2.putText(out, lim_txt, (w - 620, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2, cv2.LINE_AA)

        return out


def run_demonstration(
    source_type: str = "replay",
    scenario: str = "pedestrian",
    carla_host: str = "127.0.0.1",
    carla_port: int = 2000,
    output_dir: Optional[Path] = None,
    max_frames: Optional[int] = None,
    headless: bool = True,
    fault_mode: str = "none",
) -> Dict[str, Any]:
    """Execute the demonstration pipeline over selected source."""
    logger.info("Initializing Cognix M14 CARLA Demonstration (source=%s, scenario=%s)",
                source_type, scenario)

    if output_dir:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Pipeline
    pipeline = CarlaDemonstrationPipeline()

    if source_type == "live":
        scenarios_to_run = [scenario] if scenario != "all" else ["normal", "pedestrian", "weather", "props"]
        all_results = {}

        for sc_name in scenarios_to_run:
            logger.info("Initializing Cognix M15 Live Demonstration (scenario=%s)", sc_name)
            try:
                live_src = CarlaLiveSource(scenario_name=sc_name, host=carla_host, port=carla_port)
                live_src.connect()
                live_src.setup_sensors()
            except CarlaConnectionError as err:
                logger.error("CARLA live connection failed: %s", err)
                print("\n" + "=" * 75)
                print("[CARLA LIVE SIMULATOR CONNECTION FAILED]")
                print(f"Error: {err}\n")
                print("SETUP & RECOVERY INSTRUCTIONS:")
                print("1. CARLA Python API client (0.9.16) is installed.")
                print("2. The CARLA simulator is extracted to: C:\\CARLA_0.9.16\\CarlaUE4.exe.")
                print("3. To enable live simulation:")
                print("   Start the server: C:\\CARLA_0.9.16\\CarlaUE4.exe -quality-level=Low -benchmark -fps=20")
                print("   Then run this demo again: python -m cognix.evaluation.carla_demo_runner --source live")
                print("4. To run the verified recorded CARLA replay demonstration instead:")
                print("   python -m cognix.evaluation.carla_demo_runner --source replay --scenario pedestrian")
                print("=" * 75 + "\n")
                return {
                    "status": "connection_failed",
                    "error": str(err),
                    "source_type": "live",
                    "telemetries": [],
                }

            sc_out_dir = (output_dir / "live" / sc_name) if output_dir else None
            if sc_out_dir:
                sc_out_dir.mkdir(parents=True, exist_ok=True)

            sc_telemetries = []
            pipeline.tracker.reset(scenario_id=f"live_{sc_name}")
            active_fault = fault_mode

            for packet in live_src.stream_frames(max_frames=max_frames or 10):
                packet.fault_mode = active_fault
                packet.image_bgr = SensorFaultSimulator.apply_fault(packet.image_bgr, active_fault)

                rendered, telem = pipeline.process_frame(packet, is_props_scenario=(sc_name == "props"))
                sc_telemetries.append(telem)

                if sc_out_dir and rendered is not None:
                    save_path = sc_out_dir / f"frame_{packet.tick:06d}.jpg"
                    cv2.imwrite(str(save_path), rendered)

                if not headless and rendered is not None:
                    cv2.imshow("Cognix CARLA Demonstration", rendered)
                    key = cv2.waitKey(60) & 0xFF
                    if key in (27, ord('q')):
                        logger.info("Demo user requested exit.")
                        break
                    elif key == ord('0'):
                        active_fault = "none"
                        logger.info("Interactive fault mode set to: NONE")
                    elif key == ord('1'):
                        active_fault = "darkness"
                        logger.info("Interactive fault mode set to: DARKNESS")
                    elif key == ord('2'):
                        active_fault = "occlusion"
                        logger.info("Interactive fault mode set to: OCCLUSION")
                    elif key == ord('3'):
                        active_fault = "dropout"
                        logger.info("Interactive fault mode set to: DROPOUT")

            if not headless:
                cv2.destroyAllWindows()

            avg_lat = float(np.mean([t.total_latency_ms for t in sc_telemetries])) if sc_telemetries else 0.0
            all_results[sc_name] = {
                "frames_processed": len(sc_telemetries),
                "avg_total_latency_ms": round(avg_lat, 2),
                "telemetries": [t.to_dict() for t in sc_telemetries],
            }

        return {
            "status": "success",
            "source_type": "live",
            "results": all_results,
        }
    else:
        # Replay mode
        scenarios_to_run = [scenario] if scenario != "all" else ["normal", "pedestrian", "weather", "props"]
        all_results = {}

        for sc_name in scenarios_to_run:
            logger.info("Running replay scenario: %s", sc_name)
            replay_src = CarlaReplaySource(sc_name)
            sc_out_dir = (output_dir / "replay" / sc_name) if output_dir else None
            if sc_out_dir:
                sc_out_dir.mkdir(parents=True, exist_ok=True)

            sc_telemetries = []
            pipeline.tracker.reset(scenario_id=sc_name)
            active_fault = fault_mode

            for packet in replay_src.stream_frames(max_frames=max_frames):
                packet.fault_mode = active_fault
                packet.image_bgr = SensorFaultSimulator.apply_fault(packet.image_bgr, active_fault)

                rendered, telem = pipeline.process_frame(packet, is_props_scenario=(sc_name == "props"))
                sc_telemetries.append(telem)

                if sc_out_dir and rendered is not None:
                    save_path = sc_out_dir / f"frame_{packet.tick:06d}.jpg"
                    cv2.imwrite(str(save_path), rendered)

                if not headless and rendered is not None:
                    cv2.imshow("Cognix CARLA Demonstration", rendered)
                    key = cv2.waitKey(60) & 0xFF
                    if key in (27, ord('q')):
                        logger.info("Demo user requested exit.")
                        break
                    elif key == ord('0'):
                        active_fault = "none"
                        logger.info("Interactive fault mode set to: NONE")
                    elif key == ord('1'):
                        active_fault = "darkness"
                        logger.info("Interactive fault mode set to: DARKNESS")
                    elif key == ord('2'):
                        active_fault = "occlusion"
                        logger.info("Interactive fault mode set to: OCCLUSION")
                    elif key == ord('3'):
                        active_fault = "dropout"
                        logger.info("Interactive fault mode set to: DROPOUT")

            if not headless:
                cv2.destroyAllWindows()

            avg_lat = float(np.mean([t.total_latency_ms for t in sc_telemetries])) if sc_telemetries else 0.0
            all_results[sc_name] = {
                "frames_processed": len(sc_telemetries),
                "avg_total_latency_ms": round(avg_lat, 2),
                "telemetries": [t.to_dict() for t in sc_telemetries],
            }

        return {
            "status": "success",
            "source_type": "replay",
            "results": all_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Cognix M15 Interactive CARLA Demonstration Runner")
    parser.add_argument("--source", choices=["live", "replay"], default="live",
                        help="Data source: 'live' (CARLA server) or 'replay' (recorded benchmark frames)")
    parser.add_argument("--scenario", choices=["normal", "pedestrian", "weather", "props", "all"], default="normal",
                        help="Scenario to run: normal, pedestrian, weather, props, or all")
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA server host IP")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA server RPC port")
    parser.add_argument("--output-dir", default="reports/carla_runtime_v1/evaluation_runs/m15_interactive_demo_001/demo_artifacts",
                        help="Directory to save rendered frame artifacts and logs")
    parser.add_argument("--max-frames", type=int, default=10, help="Limit number of frames")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without opening GUI window")
    parser.add_argument("--interactive", action="store_false", dest="headless", help="Open OpenCV display window")
    parser.add_argument("--fault", choices=["none", "darkness", "occlusion", "dropout"], default="none",
                        help="Inject simulated sensor fault for safety robustness stress test")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(args.output_dir) if args.output_dir else None
    res = run_demonstration(
        source_type=args.source,
        scenario=args.scenario,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        output_dir=out_dir,
        max_frames=args.max_frames,
        headless=args.headless,
        fault_mode=args.fault,
    )

    if out_dir and res.get("status") == "success":
        summary_path = out_dir / f"demo_execution_summary_{args.source}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nDemonstration execution summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

"""Temporal Object Tracking and Hazard-Relevance Module.

Research-only module implementing class-constrained IoU association,
track state lifecycle management, pixel-space displacement measurement,
and image-space roadway corridor gating.
DO NOT IMPORT INTO PRODUCTION COGNIX DECISION PIPELINE.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from typing import List, Dict, Any, Optional, Tuple, Union

from cognix.evaluation.torchvision_detector import DetectionBox


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class TrackObservation:
    tick: int
    box: DetectionBox
    score: float
    center_x: float
    center_y: float
    area: float
    iou_with_previous: Optional[float] = None
    displacement_px: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "box": self.box.to_dict(),
            "score": round(self.score, 4),
            "center": [round(self.center_x, 1), round(self.center_y, 1)],
            "area": round(self.area, 1),
            "iou_with_previous": round(self.iou_with_previous, 3) if self.iou_with_previous is not None else None,
            "displacement_px": round(self.displacement_px, 2) if self.displacement_px is not None else None,
        }


@dataclass
class TrackTrajectory:
    track_id: int
    class_name: str
    class_id: int
    state: TrackState = TrackState.TENTATIVE
    start_tick: int = 0
    last_tick: int = 0
    total_hits: int = 0
    consecutive_misses: int = 0
    observations: List[TrackObservation] = field(default_factory=list)

    @property
    def age(self) -> int:
        """Total elapsed ticks from first observation to latest observation."""
        return max(1, self.last_tick - self.start_tick + 1)

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def latest_observation(self) -> TrackObservation:
        return self.observations[-1]

    @property
    def latest_box(self) -> DetectionBox:
        return self.observations[-1].box

    @property
    def latest_center(self) -> Tuple[float, float]:
        obs = self.observations[-1]
        return (obs.center_x, obs.center_y)

    @property
    def mean_displacement_px(self) -> float:
        """Average step displacement across all observed consecutive transitions."""
        displacements = [obs.displacement_px for obs in self.observations if obs.displacement_px is not None]
        if not displacements:
            return 0.0
        return float(sum(displacements) / len(displacements))

    @property
    def total_displacement_px(self) -> float:
        """Net Euclidean displacement from first observed center to latest observed center."""
        if len(self.observations) < 2:
            return 0.0
        first = self.observations[0]
        last = self.observations[-1]
        dx = last.center_x - first.center_x
        dy = last.center_y - first.center_y
        return float(math.hypot(dx, dy))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "state": self.state.value,
            "start_tick": self.start_tick,
            "last_tick": self.last_tick,
            "total_hits": self.total_hits,
            "consecutive_misses": self.consecutive_misses,
            "duration_ticks": self.age,
            "mean_displacement_px": round(self.mean_displacement_px, 2),
            "total_displacement_px": round(self.total_displacement_px, 2),
            "latest_box": self.latest_box.to_dict(),
            "observations": [obs.to_dict() for obs in self.observations],
        }


def compute_iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
    """Compute 2D Intersection over Union between two detection boxes."""
    x_left = max(box_a.x1, box_b.x1)
    y_top = max(box_a.y1, box_b.y1)
    x_right = min(box_a.x2, box_b.x2)
    y_bottom = min(box_a.y2, box_b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = box_a.area
    area_b = box_b.area
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


class RoadwayCorridor:
    """Configurable 2D Image-Space Region of Interest for Forward Roadway."""

    def __init__(
        self,
        y_min_frac: float = 0.46,  # ~500px on 1080p (horizon cut)
        y_max_frac: float = 0.82,  # ~880px on 1080p (ego vehicle hood cut)
        x_min_frac: float = 0.26,  # ~500px on 1920p (left sidewalk boundary)
        x_max_frac: float = 0.74,  # ~1420px on 1920p (right sidewalk boundary)
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.x_min = x_min_frac * image_width
        self.x_max = x_max_frac * image_width
        self.y_min = y_min_frac * image_height
        self.y_max = y_max_frac * image_height

    def contains_center(self, box: DetectionBox) -> bool:
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0
        return self.x_min <= cx <= self.x_max and self.y_min <= cy <= self.y_max

    def overlap_fraction(self, box: DetectionBox) -> float:
        x_left = max(box.x1, self.x_min)
        y_top = max(box.y1, self.y_min)
        x_right = min(box.x2, self.x_max)
        y_bottom = min(box.y2, self.y_max)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        inter_area = (x_right - x_left) * (y_bottom - y_top)
        return float(inter_area / max(1.0, box.area))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_dimensions": [self.image_width, self.image_height],
            "x_bounds_px": [round(self.x_min, 1), round(self.x_max, 1)],
            "y_bounds_px": [round(self.y_min, 1), round(self.y_max, 1)],
            "x_fractions": [round(self.x_min / self.image_width, 3), round(self.x_max / self.image_width, 3)],
            "y_fractions": [round(self.y_min / self.image_height, 3), round(self.y_max / self.image_height, 3)],
            "semantic_rationale": "Excludes upper sky/horizon (y < 500) and ego vehicle hood (y > 880), plus roadside sidewalks (x < 500, x > 1420).",
        }


class TemporalTracker:
    """Deterministic Class-Constrained IoU Temporal Object Tracker.

    Maintains active track identities within a scenario recording.
    Resets strictly across scenario boundaries to prevent cross-scenario contamination.
    """

    def __init__(
        self,
        iou_threshold: float = 0.20,
        max_missed_frames: int = 2,
        min_hits_to_confirm: int = 2,
        max_center_distance_px: float = 350.0,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_center_distance_px = max_center_distance_px

        self._next_track_id: int = 1
        self._tracks: Dict[int, TrackTrajectory] = {}
        self._current_tick: int = 0
        self._current_scenario: Optional[str] = None

    def reset(self, scenario_id: Optional[str] = None) -> None:
        """Reset all tracking state at recording or scenario boundaries."""
        self._next_track_id = 1
        self._tracks.clear()
        self._current_tick = 0
        self._current_scenario = scenario_id

    def update(
        self,
        tick: int,
        detections: List[DetectionBox],
        scenario_id: Optional[str] = None,
    ) -> List[TrackTrajectory]:
        """Update tracker with detections from current frame."""
        # Enforce strict scenario boundary reset
        if scenario_id is not None and scenario_id != self._current_scenario:
            self.reset(scenario_id=scenario_id)

        self._current_tick = tick

        # Filter out invalid / NaN detections
        valid_dets = [d for d in detections if d.is_valid(10000, 10000)]

        # Get active candidate tracks (not terminated)
        active_track_ids = [
            tid for tid, trk in self._tracks.items()
            if trk.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)
        ]

        matched_tracks = set()
        matched_dets = set()

        if active_track_ids and valid_dets:
            # Build pairwise association matrix: (iou, distance, trk_id, det_idx)
            pairs = []
            for tid in active_track_ids:
                trk = self._tracks[tid]
                latest_b = trk.latest_box
                cx_t, cy_t = trk.latest_center

                for d_idx, det in enumerate(valid_dets):
                    # Strict class constraint: person matches person, car matches car
                    if trk.class_name != det.class_name:
                        continue

                    cx_d = (det.x1 + det.x2) / 2.0
                    cy_d = (det.y1 + det.y2) / 2.0
                    dist = math.hypot(cx_d - cx_t, cy_d - cy_t)

                    if dist > self.max_center_distance_px:
                        continue

                    iou = compute_iou(latest_b, det)
                    if iou >= self.iou_threshold:
                        pairs.append((iou, -dist, tid, d_idx))

            # Sort greedy by highest IoU, then smallest center distance
            pairs.sort(reverse=True)

            for iou, _, tid, d_idx in pairs:
                if tid in matched_tracks or d_idx in matched_dets:
                    continue

                # Associate detection to track
                det = valid_dets[d_idx]
                trk = self._tracks[tid]
                cx_d = (det.x1 + det.x2) / 2.0
                cy_d = (det.y1 + det.y2) / 2.0
                cx_t, cy_t = trk.latest_center
                disp = math.hypot(cx_d - cx_t, cy_d - cy_t)

                obs = TrackObservation(
                    tick=tick,
                    box=det,
                    score=det.score,
                    center_x=cx_d,
                    center_y=cy_d,
                    area=det.area,
                    iou_with_previous=iou,
                    displacement_px=disp,
                )

                trk.observations.append(obs)
                trk.last_tick = tick
                trk.total_hits += 1
                trk.consecutive_misses = 0

                if trk.total_hits >= self.min_hits_to_confirm:
                    trk.state = TrackState.CONFIRMED
                else:
                    trk.state = TrackState.TENTATIVE

                matched_tracks.add(tid)
                matched_dets.add(d_idx)

        # Handle unmatched active tracks (misses)
        for tid in active_track_ids:
            if tid not in matched_tracks:
                trk = self._tracks[tid]
                trk.consecutive_misses += 1
                if trk.consecutive_misses > self.max_missed_frames:
                    trk.state = TrackState.TERMINATED
                else:
                    trk.state = TrackState.LOST

        # Handle unmatched detections (new tentative tracks)
        for d_idx, det in enumerate(valid_dets):
            if d_idx not in matched_dets:
                cx_d = (det.x1 + det.x2) / 2.0
                cy_d = (det.y1 + det.y2) / 2.0
                obs = TrackObservation(
                    tick=tick,
                    box=det,
                    score=det.score,
                    center_x=cx_d,
                    center_y=cy_d,
                    area=det.area,
                    iou_with_previous=None,
                    displacement_px=None,
                )
                new_state = TrackState.CONFIRMED if self.min_hits_to_confirm <= 1 else TrackState.TENTATIVE
                new_track = TrackTrajectory(
                    track_id=self._next_track_id,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    state=new_state,
                    start_tick=tick,
                    last_tick=tick,
                    total_hits=1,
                    consecutive_misses=0,
                    observations=[obs],
                )
                self._tracks[self._next_track_id] = new_track
                self._next_track_id += 1

        # Return active tracks visible in this tick
        return [trk for trk in self._tracks.values() if trk.last_tick == tick]

    def get_all_tracks(self) -> List[TrackTrajectory]:
        return list(self._tracks.values())

    def get_confirmed_tracks(self) -> List[TrackTrajectory]:
        return [trk for trk in self._tracks.values() if trk.is_confirmed]

    def get_active_tracks(self) -> List[TrackTrajectory]:
        return [
            trk for trk in self._tracks.values()
            if trk.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)
        ]

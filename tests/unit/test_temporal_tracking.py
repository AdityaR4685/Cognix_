"""Unit tests for temporal object tracking and hazard-relevance module."""
import math
import pytest

from cognix.evaluation.torchvision_detector import DetectionBox
from cognix.evaluation.temporal_tracking import (
    TrackState,
    TrackObservation,
    TrackTrajectory,
    TemporalTracker,
    RoadwayCorridor,
    compute_iou,
)


class TestTemporalTracking:
    def test_compute_iou_identical_and_disjoint(self):
        b1 = DetectionBox(10.0, 10.0, 50.0, 50.0, 0.9, 1, "person")
        b2 = DetectionBox(10.0, 10.0, 50.0, 50.0, 0.8, 1, "person")
        assert math.isclose(compute_iou(b1, b2), 1.0, abs_tol=1e-4)

        b_disjoint = DetectionBox(100.0, 100.0, 150.0, 150.0, 0.9, 1, "person")
        assert compute_iou(b1, b_disjoint) == 0.0

    def test_deterministic_association_consecutive_frames(self):
        tracker = TemporalTracker(iou_threshold=0.2, min_hits_to_confirm=2)

        # Frame 1: Person at (100, 100, 150, 200)
        d1 = [DetectionBox(100.0, 100.0, 150.0, 200.0, 0.95, 1, "person")]
        t1 = tracker.update(tick=1, detections=d1, scenario_id="scenario_A")
        assert len(t1) == 1
        assert t1[0].track_id == 1
        assert t1[0].state == TrackState.TENTATIVE
        assert t1[0].total_hits == 1

        # Frame 2: Person moves slightly to (105, 102, 155, 202)
        d2 = [DetectionBox(105.0, 102.0, 155.0, 202.0, 0.94, 1, "person")]
        t2 = tracker.update(tick=2, detections=d2, scenario_id="scenario_A")
        assert len(t2) == 1
        assert t2[0].track_id == 1
        assert t2[0].state == TrackState.CONFIRMED
        assert t2[0].total_hits == 2
        # Displacement: sqrt(5^2 + 2^2) ≈ 5.385 px
        assert math.isclose(t2[0].latest_observation.displacement_px, 5.385, abs_tol=1e-2)
        assert math.isclose(t2[0].mean_displacement_px, 5.385, abs_tol=1e-2)

    def test_class_consistency_prevents_mismatched_association(self):
        tracker = TemporalTracker(iou_threshold=0.1)

        # Frame 1: Person at (200, 200, 300, 400)
        d1 = [DetectionBox(200.0, 200.0, 300.0, 400.0, 0.9, 1, "person")]
        tracker.update(tick=1, detections=d1, scenario_id="sc_class")

        # Frame 2: Car appears at exact same bounding box
        d2 = [DetectionBox(200.0, 200.0, 300.0, 400.0, 0.9, 3, "car")]
        t2 = tracker.update(tick=2, detections=d2, scenario_id="sc_class")

        # Must not associate car to person track
        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 2
        assert all_tracks[0].class_name == "person"
        assert all_tracks[1].class_name == "car"
        assert all_tracks[0].track_id != all_tracks[1].track_id

    def test_handling_empty_frames_and_track_termination(self):
        tracker = TemporalTracker(max_missed_frames=2, min_hits_to_confirm=1)

        # Frame 1: Car detected
        d1 = [DetectionBox(500.0, 500.0, 700.0, 700.0, 0.95, 3, "car")]
        tracker.update(tick=1, detections=d1, scenario_id="sc_miss")
        assert tracker._tracks[1].state == TrackState.CONFIRMED

        # Frame 2: Miss 1 (LOST)
        tracker.update(tick=2, detections=[], scenario_id="sc_miss")
        assert tracker._tracks[1].state == TrackState.LOST
        assert tracker._tracks[1].consecutive_misses == 1

        # Frame 3: Miss 2 (LOST)
        tracker.update(tick=3, detections=[], scenario_id="sc_miss")
        assert tracker._tracks[1].state == TrackState.LOST
        assert tracker._tracks[1].consecutive_misses == 2

        # Frame 4: Miss 3 -> TERMINATED (> max_missed_frames=2)
        tracker.update(tick=4, detections=[], scenario_id="sc_miss")
        assert tracker._tracks[1].state == TrackState.TERMINATED

    def test_temporary_occlusion_and_recovery(self):
        tracker = TemporalTracker(max_missed_frames=2, min_hits_to_confirm=1)

        # Frame 1: Person detected
        d1 = [DetectionBox(100.0, 100.0, 140.0, 180.0, 0.9, 1, "person")]
        tracker.update(tick=1, detections=d1, scenario_id="sc_occ")

        # Frame 2: Temporary occlusion (empty)
        tracker.update(tick=2, detections=[], scenario_id="sc_occ")
        assert tracker._tracks[1].state == TrackState.LOST

        # Frame 3: Person reappears near previous position
        d3 = [DetectionBox(105.0, 102.0, 145.0, 182.0, 0.9, 1, "person")]
        t3 = tracker.update(tick=3, detections=d3, scenario_id="sc_occ")
        assert len(t3) == 1
        assert t3[0].track_id == 1
        assert t3[0].state == TrackState.CONFIRMED
        assert t3[0].consecutive_misses == 0

    def test_strict_scenario_boundary_reset(self):
        tracker = TemporalTracker()

        # Scenario A
        d_a = [DetectionBox(100.0, 100.0, 150.0, 200.0, 0.9, 1, "person")]
        tracker.update(tick=1, detections=d_a, scenario_id="Scenario_Alpha")
        assert len(tracker.get_all_tracks()) == 1
        assert tracker.get_all_tracks()[0].track_id == 1

        # Switch to Scenario B with identical bounding box
        d_b = [DetectionBox(100.0, 100.0, 150.0, 200.0, 0.9, 1, "person")]
        tracker.update(tick=1, detections=d_b, scenario_id="Scenario_Beta")

        # Tracker must have reset: new track has ID 1 under Scenario_Beta
        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 1
        assert all_tracks[0].track_id == 1
        assert tracker._current_scenario == "Scenario_Beta"

    def test_roadway_corridor_filtering(self):
        corridor = RoadwayCorridor(
            y_min_frac=0.46, y_max_frac=0.82,
            x_min_frac=0.26, x_max_frac=0.74,
            image_width=1000, image_height=1000,
        )

        # Center in corridor: (500, 600)
        box_in = DetectionBox(450.0, 550.0, 550.0, 650.0, 0.9, 1, "person")
        assert corridor.contains_center(box_in) is True
        assert corridor.overlap_fraction(box_in) > 0.9

        # Sidewalk pedestrian: center at x = 100 (left sidewalk)
        box_sidewalk = DetectionBox(50.0, 550.0, 150.0, 650.0, 0.9, 1, "person")
        assert corridor.contains_center(box_sidewalk) is False
        assert corridor.overlap_fraction(box_sidewalk) == 0.0

        # Sky / horizon object: center at y = 200
        box_sky = DetectionBox(450.0, 150.0, 550.0, 250.0, 0.9, 1, "person")
        assert corridor.contains_center(box_sky) is False

        # Ego vehicle hood: center at y = 950
        box_hood = DetectionBox(450.0, 900.0, 550.0, 1000.0, 0.9, 3, "car")
        assert corridor.contains_center(box_hood) is False

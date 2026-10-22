"""Unit tests for Ground-Truth Annotation and Object Tracking Validation (Milestone 8).

Validates:
1. Schema constraints: xyxy format, bounds, non-degeneracy.
2. Frame constraints: duplicate object_id and duplicate track_id rejection.
3. Scenario isolation: track identity independence across scenarios.
4. Ignored region handling: ignored boxes do not penalize false alarms.
5. Detection metrics: true positives, false positives, false negatives, precision, recall.
6. Tracking metrics: identity switches, fragmentations, MT/ML categorization.
7. Separation of concerns: ground truth isolation from inference inputs.
8. Serialization and cryptographic hash reproducibility.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pytest

from cognix.evaluation.ground_truth_schema import (
    AnnotationDataset,
    FrameAnnotation,
    AnnotatedObject,
    BoundingBox,
    ObjectCategory,
    OcclusionStatus,
    VisibilityStatus,
    ConfirmationStatus,
    ValidationError,
)
from cognix.evaluation.tracking_evaluator import (
    TrackingEvaluator,
    ClassDetectionMetrics,
    TrackingEvaluationMetrics,
)


class TestGroundTruthSchema:
    """Tests for schema, bounding box geometry, and constraints."""

    def test_valid_bounding_box(self):
        box = BoundingBox(x1=100.0, y1=200.0, x2=300.0, y2=450.0)
        box.validate(image_width=1920, image_height=1080)
        assert box.width == 200.0
        assert box.height == 250.0
        assert box.area == 50000.0
        assert box.centroid == (200.0, 325.0)

    def test_out_of_bounds_box_raises_validation_error(self):
        # Exceeds width
        with pytest.raises(ValidationError, match="Invalid horizontal coordinates"):
            b1 = BoundingBox(x1=100.0, y1=200.0, x2=2000.0, y2=450.0)
            b1.validate(image_width=1920, image_height=1080)

        # Negative coordinate
        with pytest.raises(ValidationError, match="Invalid vertical coordinates"):
            b2 = BoundingBox(x1=100.0, y1=-10.0, x2=300.0, y2=450.0)
            b2.validate(image_width=1920, image_height=1080)

    def test_inverted_or_degenerate_box_raises_validation_error(self):
        # Inverted x
        with pytest.raises(ValidationError, match="Invalid horizontal coordinates"):
            b1 = BoundingBox(x1=400.0, y1=100.0, x2=200.0, y2=300.0)
            b1.validate(image_width=1920, image_height=1080)

        # Zero width
        with pytest.raises(ValidationError, match="Degenerate box area"):
            b2 = BoundingBox(x1=100.0, y1=100.0, x2=101.0, y2=300.0)
            b2.validate(image_width=1920, image_height=1080)

    def test_duplicate_track_id_in_same_frame_raises(self):
        obj1 = AnnotatedObject(
            object_id="obj_1",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(10, 10, 50, 100),
        )
        obj2 = AnnotatedObject(
            object_id="obj_2",
            track_id=1,  # Duplicate in same frame!
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(100, 10, 150, 100),
        )
        frame = FrameAnnotation(
            scenario_id="sc_test",
            tick=0,
            frame_hash_sha256="abc",
            objects=[obj1, obj2],
        )
        with pytest.raises(ValidationError, match="Duplicate track_id 1"):
            frame.validate()

    def test_duplicate_object_id_in_same_frame_raises(self):
        obj1 = AnnotatedObject(
            object_id="obj_dup",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(10, 10, 50, 100),
        )
        obj2 = AnnotatedObject(
            object_id="obj_dup",  # Duplicate object_id!
            track_id=2,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(100, 10, 150, 100),
        )
        frame = FrameAnnotation(
            scenario_id="sc_test",
            tick=0,
            frame_hash_sha256="abc",
            objects=[obj1, obj2],
        )
        with pytest.raises(ValidationError, match="Duplicate object_id 'obj_dup'"):
            frame.validate()

    def test_serialization_and_hash_reproducibility(self, tmp_path):
        obj = AnnotatedObject(
            object_id="obj_1",
            track_id=42,
            category=ObjectCategory.VEHICLE,
            bbox=BoundingBox(100.0, 200.0, 300.0, 400.0),
            occlusion=OcclusionStatus.PARTIAL,
            visibility=VisibilityStatus.CLEAR,
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        frame = FrameAnnotation(
            scenario_id="test/anomaly/Town01/running-pedestrian/scenario-1",
            tick=5,
            frame_hash_sha256="d04b1234567890abcdef",
            objects=[obj],
            reviewer_id="SafetyAuditor",
            confirmed_at_utc="2026-10-20T12:00:00Z",
        )
        dataset = AnnotationDataset(
            schema_version="1.0.0",
            created_at_utc="2026-10-20T12:00:00Z",
            frames=[frame],
        )

        out_path = tmp_path / "test_annotations.json"
        sha1 = dataset.save_json(out_path)

        loaded = AnnotationDataset.load_json(out_path)
        sha2 = loaded.compute_sha256()

        assert sha1 == sha2
        assert len(loaded.frames) == 1
        assert loaded.frames[0].objects[0].track_id == 42
        assert loaded.frames[0].objects[0].category == ObjectCategory.VEHICLE


class TestTrackingEvaluator:
    """Tests for detection matching, ignore regions, and tracking metrics."""

    def test_detection_matching_true_positives_and_false_alarms(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Ground truth: 1 person, 1 vehicle (human confirmed)
        gt_person = AnnotatedObject(
            object_id="gt_1",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(100.0, 100.0, 200.0, 300.0),
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        gt_car = AnnotatedObject(
            object_id="gt_2",
            track_id=2,
            category=ObjectCategory.VEHICLE,
            bbox=BoundingBox(500.0, 500.0, 800.0, 700.0),
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        frame = FrameAnnotation(
            scenario_id="sc_1",
            tick=0,
            frame_hash_sha256="hash0",
            objects=[gt_person, gt_car],
            reviewer_id="Rev1",
        )
        dataset = AnnotationDataset(frames=[frame])

        # Predictions:
        # 1. Matches gt_person closely (TP)
        # 2. Distant box with no GT match (FP)
        # GT car is missed (FN)
        preds = {
            ("sc_1", 0): [
                {"x1": 105.0, "y1": 105.0, "x2": 195.0, "y2": 295.0, "class_name": "person"},
                {"x1": 900.0, "y1": 200.0, "x2": 1000.0, "y2": 300.0, "class_name": "vehicle"},
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        assert overall["true_positives"] == 1
        assert overall["false_positives"] == 1
        assert overall["false_negatives"] == 1
        assert overall["precision"] == 0.5
        assert overall["recall"] == 0.5

    def test_ignored_region_does_not_penalize_false_positive(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Ground truth: 1 ignore region (e.g. distant glare or ambiguous clutter)
        gt_ignore = AnnotatedObject(
            object_id="ign_1",
            track_id=99,
            category=ObjectCategory.IGNORE_AMBIGUOUS,
            bbox=BoundingBox(400.0, 400.0, 600.0, 600.0),
            is_ambiguous=True,
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        frame = FrameAnnotation(
            scenario_id="sc_ign",
            tick=0,
            frame_hash_sha256="hash_ign",
            objects=[gt_ignore],
            reviewer_id="Rev1",
        )
        dataset = AnnotationDataset(frames=[frame])

        # Prediction overlapping the ignore region
        preds = {
            ("sc_ign", 0): [
                {"x1": 410.0, "y1": 410.0, "x2": 590.0, "y2": 590.0, "class_name": "vehicle"}
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        # Ignored match should NOT count as False Positive
        assert overall["true_positives"] == 0
        assert overall["false_positives"] == 0
        assert overall["ignored_matches"] == 1

    def test_identity_switch_and_fragmentation_detection(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # GT Track 1 over 4 frames (ticks 0, 5, 10, 15)
        gt_frames = []
        for tick in [0, 5, 10, 15]:
            gt_obj = AnnotatedObject(
                object_id=f"gt_{tick}",
                track_id=1,
                category=ObjectCategory.PERSON,
                bbox=BoundingBox(100.0 + tick * 2, 200.0, 150.0 + tick * 2, 300.0),
                confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
            )
            gt_frames.append(
                FrameAnnotation(
                    scenario_id="sc_switch",
                    tick=tick,
                    frame_hash_sha256=f"h_{tick}",
                    objects=[gt_obj],
                    reviewer_id="Rev",
                )
            )
        dataset = AnnotationDataset(frames=gt_frames)

        # Predictions:
        # In tick 0: Pred Track 10 matches GT Track 1
        # In tick 5: Pred Track 20 matches GT Track 1 (Identity Switch!)
        # In tick 10: Missed (gap)
        # In tick 15: Pred Track 20 re-matches GT Track 1 (Resumption -> CLEAR MOT Fragmentation!)
        trajectories = {
            "sc_switch": [
                {
                    "track_id": 10,
                    "observations": [{"tick": 0, "box": {"x1": 100.0, "y1": 200.0, "x2": 150.0, "y2": 300.0}}],
                },
                {
                    "track_id": 20,
                    "observations": [
                        {"tick": 5, "box": {"x1": 110.0, "y1": 200.0, "x2": 160.0, "y2": 300.0}},
                        {"tick": 15, "box": {"x1": 130.0, "y1": 200.0, "x2": 180.0, "y2": 300.0}},
                    ],
                },
            ]
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.num_gt_tracks == 1
        assert metrics.identity_switches == 1
        assert metrics.fragmentations == 1
        assert metrics.partially_tracked == 1  # 3/4 frames detected (75.0%)

    def test_scenario_boundary_isolation_disallows_cross_scenario_matching(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Two different scenarios with same track_id
        f1 = FrameAnnotation(
            scenario_id="scenario_A",
            tick=0,
            frame_hash_sha256="hA",
            objects=[
                AnnotatedObject(
                    "oA", 1, ObjectCategory.PERSON, BoundingBox(10, 10, 50, 50),
                    confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
                )
            ],
            reviewer_id="Rev",
        )
        f2 = FrameAnnotation(
            scenario_id="scenario_B",
            tick=0,
            frame_hash_sha256="hB",
            objects=[
                AnnotatedObject(
                    "oB", 1, ObjectCategory.PERSON, BoundingBox(10, 10, 50, 50),
                    confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
                )
            ],
            reviewer_id="Rev",
        )
        dataset = AnnotationDataset(frames=[f1, f2])

        # Prediction only in scenario_A
        trajectories = {
            "scenario_A": [
                {"track_id": 101, "observations": [{"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}]}
            ],
            "scenario_B": [],
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.num_gt_tracks == 2
        # scenario_A track should be mostly tracked, scenario_B track mostly lost
        assert metrics.mostly_tracked == 1
        assert metrics.mostly_lost == 1
        assert metrics.identity_switches == 0

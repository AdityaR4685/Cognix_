"""Unit tests for Pre-Registered Evaluation Protocol and Annotation Quality Safeguards (Milestone 10).

Validates:
1. Detector-to-Ground-Truth Class Mapping (COCO -> Cognix Evaluation Taxonomy).
2. Strict rejection of class mismatches on spatial overlap (IoU >= 0.50 with wrong class is FP + FN).
3. Ignore-region semantics (IoU >= 0.50 absorbs unmatched detections as ignored; background FP is penalized).
4. Machine-checkable annotation completeness states (UNREVIEWED, CONFIRMED_WITH_OBJECTS, CONFIRMED_EMPTY, REQUIRES_ADJUDICATION).
5. IncompleteAnnotationError gating on official runs vs exploratory pass-through.
6. Reviewed-empty frame evaluation (clean 0 FP on empty, penalized on false alarm).
7. Tracking evaluation class agreement enforcement.
"""
import pytest

from cognix.evaluation.ground_truth_schema import (
    AnnotationDataset,
    FrameAnnotation,
    AnnotatedObject,
    BoundingBox,
    ObjectCategory,
    ConfirmationStatus,
    FrameReviewState,
)
from cognix.evaluation.tracking_evaluator import (
    TrackingEvaluator,
    IncompleteAnnotationError,
    map_detector_class_to_taxonomy,
    COCO_TO_EVALUATION_TAXONOMY,
    UNSUPPORTED_CLASS_CATEGORY,
)


class TestClassMappingAndTaxonomy:
    """Tests for MS COCO 80 to evaluation taxonomy mapping."""

    def test_pedestrian_mapping(self):
        assert map_detector_class_to_taxonomy("person") == ObjectCategory.PERSON.value
        assert map_detector_class_to_taxonomy("Person") == ObjectCategory.PERSON.value

    def test_vehicle_mappings(self):
        assert map_detector_class_to_taxonomy("car") == ObjectCategory.VEHICLE.value
        assert map_detector_class_to_taxonomy("truck") == ObjectCategory.VEHICLE.value
        assert map_detector_class_to_taxonomy("bus") == ObjectCategory.VEHICLE.value

    def test_two_wheeler_mappings(self):
        assert map_detector_class_to_taxonomy("bicycle") == ObjectCategory.BICYCLE_OR_MOTORCYCLE.value
        assert map_detector_class_to_taxonomy("motorcycle") == ObjectCategory.BICYCLE_OR_MOTORCYCLE.value

    def test_road_debris_has_no_coco_counterpart(self):
        # MS COCO contains zero classes mapping to road_debris_or_prop
        assert "road_debris_or_prop" not in [v for k, v in COCO_TO_EVALUATION_TAXONOMY.items() if k not in ObjectCategory._value2member_map_]
        # Any arbitrary COCO object like traffic light or bench maps to unsupported
        assert map_detector_class_to_taxonomy("traffic light") == UNSUPPORTED_CLASS_CATEGORY
        assert map_detector_class_to_taxonomy("bench") == UNSUPPORTED_CLASS_CATEGORY
        assert map_detector_class_to_taxonomy("dog") == UNSUPPORTED_CLASS_CATEGORY


class TestStrictClassMismatchRejection:
    """Tests that spatial overlap with mismatched class is strictly rejected as a match."""

    def test_car_prediction_over_pedestrian_gt_is_rejected_as_match(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Ground truth: 1 confirmed pedestrian
        gt_person = AnnotatedObject(
            object_id="gt_ped_1",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(100.0, 100.0, 200.0, 300.0),
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        frame = FrameAnnotation(
            scenario_id="sc_mismatch",
            tick=0,
            frame_hash_sha256="hash_mismatch",
            objects=[gt_person],
            reviewer_id="LeadAuditor",
        )
        dataset = AnnotationDataset(frames=[frame])

        # Prediction: Detector predicts a 'car' exactly on top of the person (IoU ~ 0.95)
        preds = {
            ("sc_mismatch", 0): [
                {"x1": 102.0, "y1": 102.0, "x2": 198.0, "y2": 298.0, "class_name": "car"}
            ]
        }

        # Must reject match: 0 TP, 1 FP (vehicle), 1 FN (person)
        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        assert overall["true_positives"] == 0
        assert overall["false_positives"] == 1
        assert overall["false_negatives"] == 1
        assert overall["precision"] == 0.0
        assert overall["recall"] == 0.0

        # Per-class breakdown
        per_class = res["per_class"]
        assert per_class["person"]["false_negatives"] == 1
        assert per_class["person"]["true_positives"] == 0
        assert per_class["vehicle"]["false_positives"] == 1
        assert per_class["vehicle"]["true_positives"] == 0

    def test_truck_prediction_over_car_gt_is_accepted_as_vehicle(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Ground truth: 1 confirmed car (category: VEHICLE)
        gt_car = AnnotatedObject(
            object_id="gt_veh_1",
            track_id=1,
            category=ObjectCategory.VEHICLE,
            bbox=BoundingBox(300.0, 300.0, 600.0, 600.0),
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        frame = FrameAnnotation(
            scenario_id="sc_veh_compat",
            tick=0,
            frame_hash_sha256="hash_veh",
            objects=[gt_car],
            reviewer_id="LeadAuditor",
        )
        dataset = AnnotationDataset(frames=[frame])

        # Prediction: Detector predicts 'truck' (maps to VEHICLE taxonomy)
        preds = {
            ("sc_veh_compat", 0): [
                {"x1": 305.0, "y1": 305.0, "x2": 595.0, "y2": 595.0, "class_name": "truck"}
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        assert overall["true_positives"] == 1
        assert overall["false_positives"] == 0
        assert overall["false_negatives"] == 0
        assert overall["precision"] == 1.0
        assert overall["recall"] == 1.0


class TestIgnoreRegionSemantics:
    """Tests that ignore regions absorb ambiguous predictions without false-positive penalty."""

    def test_ignore_region_absorbs_unmatched_prediction_with_sufficient_iou(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # GT: 1 confirmed ignore region (e.g. distant glare / severe occlusion)
        gt_ignore = AnnotatedObject(
            object_id="ign_box_1",
            track_id=99,
            category=ObjectCategory.IGNORE_AMBIGUOUS,
            bbox=BoundingBox(500.0, 200.0, 700.0, 400.0),
            is_ambiguous=True,
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        frame = FrameAnnotation(
            scenario_id="sc_ign",
            tick=10,
            frame_hash_sha256="hash_ign",
            objects=[gt_ignore],
            reviewer_id="LeadAuditor",
        )
        dataset = AnnotationDataset(frames=[frame])

        # Prediction overlapping ignore box with IoU >= 0.50
        preds = {
            ("sc_ign", 10): [
                {"x1": 510.0, "y1": 210.0, "x2": 690.0, "y2": 390.0, "class_name": "car"}
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        assert overall["true_positives"] == 0
        assert overall["false_positives"] == 0
        assert overall["ignored_matches"] == 1

    def test_background_false_alarm_without_ignore_box_is_penalized(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # GT: confirmed empty frame
        frame = FrameAnnotation(
            scenario_id="sc_empty",
            tick=10,
            frame_hash_sha256="hash_empty",
            objects=[],
            reviewer_id="LeadAuditor",
        )
        dataset = AnnotationDataset(frames=[frame])

        # Prediction on clear pavement / background
        preds = {
            ("sc_empty", 10): [
                {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 300.0, "class_name": "person"}
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        assert overall["false_positives"] == 1
        assert overall["ignored_matches"] == 0
        assert overall["precision"] == 0.0


class TestAnnotationCompletenessSafeguards:
    """Tests machine-checkable completeness states and IncompleteAnnotationError gating."""

    def test_unreviewed_dataset_raises_incomplete_annotation_error(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Unreviewed frame (no reviewer_id, draft object)
        obj = AnnotatedObject(
            object_id="draft_1",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(10, 10, 50, 100),
            confirmation_status=ConfirmationStatus.PENDING_REVIEW,
        )
        frame = FrameAnnotation(
            scenario_id="sc_unreviewed",
            tick=0,
            frame_hash_sha256="h0",
            objects=[obj],
            reviewer_id=None,
        )
        dataset = AnnotationDataset(frames=[frame])

        assert frame.review_state == FrameReviewState.UNREVIEWED
        assert not dataset.is_fully_confirmed

        # Calling official evaluation without exploratory flag must raise IncompleteAnnotationError
        with pytest.raises(IncompleteAnnotationError, match="Official evaluation refused on unconfirmed"):
            evaluator.evaluate_detections(dataset, {})

        with pytest.raises(IncompleteAnnotationError, match="Official tracking evaluation refused"):
            evaluator.evaluate_tracking(dataset, {})

    def test_exploratory_mode_bypasses_gating_with_warning(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        frame = FrameAnnotation(
            scenario_id="sc_unreviewed",
            tick=0,
            frame_hash_sha256="h0",
            objects=[],
            reviewer_id=None,
        )
        dataset = AnnotationDataset(frames=[frame])

        # Pass exploratory=True
        res = evaluator.evaluate_detections(dataset, {}, exploratory=True)
        assert res["is_human_verified"] is False
        assert "warning" in res
        assert "provisional" in res["warning"]

    def test_adjudication_required_state_detection(self):
        # Object flagged with conflict in review notes
        obj = AnnotatedObject(
            object_id="adj_1",
            track_id=1,
            category=ObjectCategory.VEHICLE,
            bbox=BoundingBox(100, 100, 200, 200),
            confirmation_status=ConfirmationStatus.PENDING_REVIEW,
            review_notes="conflict between annotator A and B, requires adjudication",
        )
        frame = FrameAnnotation(
            scenario_id="sc_adj",
            tick=0,
            frame_hash_sha256="h_adj",
            objects=[obj],
            reviewer_id="AnnotatorA",
        )
        dataset = AnnotationDataset(frames=[frame])

        assert frame.review_state == FrameReviewState.REQUIRES_ADJUDICATION
        counts = dataset.review_state_counts
        assert counts[FrameReviewState.REQUIRES_ADJUDICATION.value] == 1

        with pytest.raises(IncompleteAnnotationError, match="1 frame\\(s\\) requiring adjudication"):
            evaluator = TrackingEvaluator()
            evaluator.evaluate_detections(dataset, {})

    def test_confirmed_empty_and_confirmed_with_objects(self):
        # Frame 1: empty and confirmed
        f1 = FrameAnnotation(
            scenario_id="sc_clean",
            tick=0,
            frame_hash_sha256="h0",
            objects=[],
            reviewer_id="Auditor",
            confirmed_at_utc="2026-10-20T12:00:00Z",
        )
        # Frame 2: with confirmed object
        obj = AnnotatedObject(
            object_id="conf_1",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(50, 50, 150, 150),
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        f2 = FrameAnnotation(
            scenario_id="sc_clean",
            tick=1,
            frame_hash_sha256="h1",
            objects=[obj],
            reviewer_id="Auditor",
            confirmed_at_utc="2026-10-20T12:01:00Z",
        )
        dataset = AnnotationDataset(frames=[f1, f2])

        assert f1.review_state == FrameReviewState.CONFIRMED_EMPTY
        assert f2.review_state == FrameReviewState.CONFIRMED_WITH_OBJECTS
        assert dataset.is_fully_confirmed
        counts = dataset.review_state_counts
        assert counts[FrameReviewState.CONFIRMED_EMPTY.value] == 1
        assert counts[FrameReviewState.CONFIRMED_WITH_OBJECTS.value] == 1


class TestTrackingClassAgreement:
    """Tests that tracking evaluation enforces class agreement across trajectory associations."""

    def test_tracking_rejects_association_when_prediction_class_mismatches_gt(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        gt_obj = AnnotatedObject(
            object_id="gt_ped",
            track_id=1,
            category=ObjectCategory.PERSON,
            bbox=BoundingBox(100, 100, 200, 300),
            confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED,
        )
        frame = FrameAnnotation(
            scenario_id="sc_track_mismatch",
            tick=0,
            frame_hash_sha256="h_trk",
            objects=[gt_obj],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[frame])

        # Predicted trajectory has class 'car'
        trajectories = {
            "sc_track_mismatch": [
                {
                    "track_id": 99,
                    "class_name": "car",  # Mismatch with PERSON
                    "observations": [
                        {"tick": 0, "box": {"x1": 100, "y1": 100, "x2": 200, "y2": 300, "class_name": "car"}}
                    ],
                }
            ]
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.num_gt_tracks == 1
        assert metrics.mostly_lost == 1
        assert metrics.mostly_tracked == 0
        assert metrics.total_fn_detections == 1

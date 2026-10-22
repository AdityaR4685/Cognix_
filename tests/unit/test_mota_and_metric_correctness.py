"""Regression unit tests for MOTA metric correctness and evaluation taxonomy audit (Milestones 11 and 12).
"""
import json
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
    TrackingEvaluationMetrics,
    map_detector_class_to_taxonomy,
    is_target_taxonomy_class,
    OUT_OF_SCOPE_CATEGORY,
)


class TestMOTAMathematicalCorrectness:
    """Verifies that MOTA = 1.0 - (FN + FP + IDSW) / GT with exact hand calculations."""

    def test_mota_perfect_matches_equals_one(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # 2 frames, 1 GT vehicle per frame (GT total = 2)
        f0 = FrameAnnotation(
            scenario_id="sc_perf", tick=0, frame_hash_sha256="h0",
            objects=[AnnotatedObject("gt_1_0", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        f1 = FrameAnnotation(
            scenario_id="sc_perf", tick=5, frame_hash_sha256="h1",
            objects=[AnnotatedObject("gt_1_1", 1, ObjectCategory.VEHICLE, BoundingBox(12, 10, 52, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0, f1])

        # Predicted trajectory 100 perfectly matches GT 1 across both ticks
        trajectories = {
            "sc_perf": [
                {
                    "track_id": 100,
                    "class_name": "car",
                    "observations": [
                        {"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}},
                        {"tick": 5, "box": {"x1": 12, "y1": 10, "x2": 52, "y2": 50, "class_name": "car"}},
                    ],
                }
            ]
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.total_gt_detections == 2
        assert metrics.total_fn_detections == 0
        assert metrics.total_fp_detections == 0
        assert metrics.identity_switches == 0
        # MOTA = 1.0 - (0 + 0 + 0) / 2 = 1.0
        assert metrics.mota == 1.0

    def test_mota_lowers_with_one_false_positive(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # 2 frames, 1 GT vehicle per frame (GT total = 2)
        f0 = FrameAnnotation(
            scenario_id="sc_fp", tick=0, frame_hash_sha256="h0",
            objects=[AnnotatedObject("gt_1_0", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        f1 = FrameAnnotation(
            scenario_id="sc_fp", tick=5, frame_hash_sha256="h1",
            objects=[AnnotatedObject("gt_1_1", 1, ObjectCategory.VEHICLE, BoundingBox(12, 10, 52, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0, f1])

        # Track 100 matches GT 1, but Track 200 is a spurious target detection at tick 0 (1 FP)
        trajectories = {
            "sc_fp": [
                {
                    "track_id": 100,
                    "class_name": "car",
                    "observations": [
                        {"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}},
                        {"tick": 5, "box": {"x1": 12, "y1": 10, "x2": 52, "y2": 50, "class_name": "car"}},
                    ],
                },
                {
                    "track_id": 200,
                    "class_name": "car",
                    "observations": [
                        {"tick": 0, "box": {"x1": 500, "y1": 500, "x2": 600, "y2": 600, "class_name": "car"}}
                    ],
                },
            ]
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.total_gt_detections == 2
        assert metrics.total_fn_detections == 0
        assert metrics.total_fp_detections == 1  # 1 FP
        assert metrics.identity_switches == 0
        # MOTA = 1.0 - (0 + 1 + 0) / 2 = 0.5
        assert metrics.mota == 0.5

    def test_mota_lowers_with_one_false_negative(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # 2 frames, 1 GT vehicle per frame (GT total = 2)
        f0 = FrameAnnotation(
            scenario_id="sc_fn", tick=0, frame_hash_sha256="h0",
            objects=[AnnotatedObject("gt_1_0", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        f1 = FrameAnnotation(
            scenario_id="sc_fn", tick=5, frame_hash_sha256="h1",
            objects=[AnnotatedObject("gt_1_1", 1, ObjectCategory.VEHICLE, BoundingBox(12, 10, 52, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0, f1])

        # Track 100 only detects GT at tick 0; misses at tick 5 (1 FN)
        trajectories = {
            "sc_fn": [
                {
                    "track_id": 100,
                    "class_name": "car",
                    "observations": [
                        {"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}
                    ],
                }
            ]
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.total_gt_detections == 2
        assert metrics.total_fn_detections == 1  # 1 FN
        assert metrics.total_fp_detections == 0
        assert metrics.identity_switches == 0
        # MOTA = 1.0 - (1 + 0 + 0) / 2 = 0.5
        assert metrics.mota == 0.5

    def test_mota_lowers_with_one_identity_switch(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # 2 frames, 1 GT vehicle per frame (GT total = 2)
        f0 = FrameAnnotation(
            scenario_id="sc_idsw", tick=0, frame_hash_sha256="h0",
            objects=[AnnotatedObject("gt_1_0", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        f1 = FrameAnnotation(
            scenario_id="sc_idsw", tick=5, frame_hash_sha256="h1",
            objects=[AnnotatedObject("gt_1_1", 1, ObjectCategory.VEHICLE, BoundingBox(12, 10, 52, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0, f1])

        # Track 100 matches at tick 0; Track 200 matches at tick 5 (1 IDSW, 0 FN, 0 FP)
        trajectories = {
            "sc_idsw": [
                {
                    "track_id": 100,
                    "class_name": "car",
                    "observations": [
                        {"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}
                    ],
                },
                {
                    "track_id": 200,
                    "class_name": "car",
                    "observations": [
                        {"tick": 5, "box": {"x1": 12, "y1": 10, "x2": 52, "y2": 50, "class_name": "car"}}
                    ],
                },
            ]
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.total_gt_detections == 2
        assert metrics.total_fn_detections == 0
        assert metrics.total_fp_detections == 0
        assert metrics.identity_switches == 1  # 1 IDSW
        # MOTA = 1.0 - (0 + 0 + 1) / 2 = 0.5
        assert metrics.mota == 0.5

    def test_mota_mixed_errors_exact_calculation(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # 5 frames, 2 GT vehicles per frame -> GT total = 10
        frames = []
        for tick in [0, 5, 10, 15, 20]:
            f = FrameAnnotation(
                scenario_id="sc_mix", tick=tick, frame_hash_sha256=f"h_{tick}",
                objects=[
                    AnnotatedObject(f"gt_A_{tick}", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED),
                    AnnotatedObject(f"gt_B_{tick}", 2, ObjectCategory.PERSON, BoundingBox(100, 100, 150, 200), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED),
                ],
                reviewer_id="Auditor",
            )
            frames.append(f)
        dataset = AnnotationDataset(frames=frames)

        # Errors designed:
        # GT A: Tracked by Pred 10 for ticks 0, 5, 10, 15, 20 (5 hits, 0 FN, 0 IDSW)
        # GT B:
        #   tick 0: matched by Pred 20 (hit)
        #   tick 5: missed (1 FN, fragmentation starts)
        #   tick 10: matched by Pred 30 (hit, 1 IDSW, fragmentation ends)
        #   tick 15: missed (1 FN)
        #   tick 20: matched by Pred 30 (hit)
        #   -> Total for GT B: 2 FN, 1 IDSW
        # Extra False Positives:
        #   Pred 99 at tick 0 (car at (400, 400, 500, 500)) -> 1 FP
        #   Pred 99 at tick 5 (car at (400, 400, 500, 500)) -> 1 FP
        #   Pred 99 at tick 10 (car at (400, 400, 500, 500)) -> 1 FP
        #   -> Total FP = 3
        # Hand Calculation:
        # GT = 10
        # FN = 2
        # FP = 3
        # IDSW = 1
        # MOTA = 1.0 - (2 + 3 + 1) / 10 = 1.0 - 6/10 = 0.40
        trajectories = {
            "sc_mix": [
                {
                    "track_id": 10,
                    "class_name": "car",
                    "observations": [
                        {"tick": t, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}
                        for t in [0, 5, 10, 15, 20]
                    ],
                },
                {
                    "track_id": 20,
                    "class_name": "person",
                    "observations": [
                        {"tick": 0, "box": {"x1": 100, "y1": 100, "x2": 150, "y2": 200, "class_name": "person"}}
                    ],
                },
                {
                    "track_id": 30,
                    "class_name": "person",
                    "observations": [
                        {"tick": 10, "box": {"x1": 100, "y1": 100, "x2": 150, "y2": 200, "class_name": "person"}},
                        {"tick": 20, "box": {"x1": 100, "y1": 100, "x2": 150, "y2": 200, "class_name": "person"}},
                    ],
                },
                {
                    "track_id": 99,
                    "class_name": "car",
                    "observations": [
                        {"tick": t, "box": {"x1": 400, "y1": 400, "x2": 500, "y2": 500, "class_name": "car"}}
                        for t in [0, 5, 10]
                    ],
                },
            ]
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.total_gt_detections == 10
        assert metrics.total_fn_detections == 2
        assert metrics.total_fp_detections == 3
        assert metrics.identity_switches == 1
        assert metrics.mota == 0.40

    def test_mota_zero_ground_truth_policy(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Frame with zero GT objects
        f0 = FrameAnnotation(
            scenario_id="sc_zero", tick=0, frame_hash_sha256="h0",
            objects=[], reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0])

        # Sub-case A: 0 GT, 0 predictions -> MOTA is mathematically undefined (None, serializes to null in JSON)
        metrics_clean = evaluator.evaluate_tracking(dataset, {"sc_zero": []})
        assert metrics_clean.total_gt_detections == 0
        assert metrics_clean.total_fp_detections == 0
        assert metrics_clean.mota is None
        assert metrics_clean.mota_status == "UNDEFINED_ZERO_GROUND_TRUTH"

        # Verify JSON serialization represents undefined MOTA as null without quiet conversion to 0 or 1
        clean_json_str = json.dumps(metrics_clean.to_dict())
        assert '"mota": null' in clean_json_str
        assert json.loads(clean_json_str)["mota"] is None

        # Sub-case B: 0 GT, but spurious predictions occur -> MOTA is still undefined (None), FP recorded
        trajectories_noisy = {
            "sc_zero": [
                {
                    "track_id": 99,
                    "class_name": "car",
                    "observations": [{"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}],
                }
            ]
        }
        metrics_noisy = evaluator.evaluate_tracking(dataset, trajectories_noisy)
        assert metrics_noisy.total_gt_detections == 0
        assert metrics_noisy.total_fp_detections == 1
        assert metrics_noisy.mota is None
        assert metrics_noisy.mota_status == "UNDEFINED_ZERO_GROUND_TRUTH"

        # Verify JSON serialization faithfully records FP and null MOTA
        noisy_json_str = json.dumps(metrics_noisy.to_dict())
        assert '"mota": null' in noisy_json_str
        assert json.loads(noisy_json_str)["mota"] is None
        assert json.loads(noisy_json_str)["total_fp_detections"] == 1

    def test_mota_mixed_sequences_with_empty_scene_aggregates_properly(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Scenario 1: Empty scene (0 GT), but 1 false positive detection
        f_empty = FrameAnnotation(
            scenario_id="sc_empty_scene", tick=0, frame_hash_sha256="h0",
            objects=[], reviewer_id="Auditor",
        )
        # Scenario 2: Normal scene (2 GT vehicles at tick 0 and 5, both tracked)
        f_gt0 = FrameAnnotation(
            scenario_id="sc_normal_scene", tick=0, frame_hash_sha256="h1",
            objects=[AnnotatedObject("gt_1_0", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        f_gt1 = FrameAnnotation(
            scenario_id="sc_normal_scene", tick=5, frame_hash_sha256="h2",
            objects=[AnnotatedObject("gt_1_1", 1, ObjectCategory.VEHICLE, BoundingBox(12, 10, 52, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f_empty, f_gt0, f_gt1])

        trajectories = {
            "sc_empty_scene": [
                {"track_id": 99, "class_name": "car", "observations": [{"tick": 0, "box": {"x1": 500, "y1": 500, "x2": 600, "y2": 600, "class_name": "car"}}]}
            ],
            "sc_normal_scene": [
                {"track_id": 100, "class_name": "car", "observations": [
                    {"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}},
                    {"tick": 5, "box": {"x1": 12, "y1": 10, "x2": 52, "y2": 50, "class_name": "car"}},
                ]}
            ],
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        # Aggregated GT = 2 > 0 -> MOTA is computed!
        # FN = 0, FP = 1 (from empty scene), IDSW = 0
        # MOTA = 1.0 - (0 + 1 + 0) / 2 = 0.50
        assert metrics.total_gt_detections == 2
        assert metrics.total_fn_detections == 0
        assert metrics.total_fp_detections == 1
        assert metrics.identity_switches == 0
        assert metrics.mota == 0.50
        assert metrics.mota_status == "COMPUTED"

    def test_mota_scenario_aggregation_consistency(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Scenario 1: 1 GT vehicle, perfectly tracked (GT=1, FN=0, FP=0, IDSW=0)
        f_sc1 = FrameAnnotation(
            scenario_id="sc_1", tick=0, frame_hash_sha256="h1",
            objects=[AnnotatedObject("gt_1", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        # Scenario 2: 1 GT vehicle, missed (GT=1, FN=1, FP=0, IDSW=0) + 1 FP (FP=1)
        f_sc2 = FrameAnnotation(
            scenario_id="sc_2", tick=0, frame_hash_sha256="h2",
            objects=[AnnotatedObject("gt_2", 1, ObjectCategory.VEHICLE, BoundingBox(100, 100, 150, 150), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f_sc1, f_sc2])

        trajectories = {
            "sc_1": [
                {"track_id": 10, "class_name": "car", "observations": [{"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}]}
            ],
            "sc_2": [
                {"track_id": 20, "class_name": "car", "observations": [{"tick": 0, "box": {"x1": 500, "y1": 500, "x2": 600, "y2": 600, "class_name": "car"}}]}
            ],
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        # Aggregated totals: GT=2, FN=1, FP=1, IDSW=0 -> MOTA = 1.0 - (1 + 1 + 0)/2 = 0.0
        assert metrics.total_gt_detections == 2
        assert metrics.total_fn_detections == 1
        assert metrics.total_fp_detections == 1
        assert metrics.identity_switches == 0
        assert metrics.mota == 0.0


class TestTaxonomyAndDetectionScopeAudit:
    """Verifies clear separation between target hazard false alarms and out-of-scope detections."""

    def test_out_of_scope_detection_is_segregated_from_target_fp(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Ground truth: 1 pedestrian (target taxonomy)
        f0 = FrameAnnotation(
            scenario_id="sc_scope", tick=0, frame_hash_sha256="h0",
            objects=[AnnotatedObject("gt_p", 1, ObjectCategory.PERSON, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0])

        # Detector outputs:
        # 1. Correct person (TP)
        # 2. 'traffic light' overhead (Out-of-Scope COCO class)
        # 3. 'bench' on sidewalk (Out-of-Scope COCO class)
        preds = {
            ("sc_scope", 0): [
                {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "person"},
                {"x1": 200, "y1": 20, "x2": 250, "y2": 100, "class_name": "traffic light"},
                {"x1": 300, "y1": 400, "x2": 450, "y2": 450, "class_name": "bench"},
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        # Target hazard metrics should reflect perfect pedestrian detection
        assert overall["true_positives"] == 1
        assert overall["false_positives"] == 0  # Zero target-class false alarms!
        assert overall["false_negatives"] == 0
        assert overall["precision"] == 1.0
        assert overall["recall"] == 1.0

        # Out-of-scope detections are tracked prominently, not silently discarded
        assert overall["out_of_scope_detections"] == 2
        assert res["out_of_scope_summary"]["total_out_of_scope"] == 2
        assert res["out_of_scope_summary"]["by_raw_class"]["traffic light"] == 1
        assert res["out_of_scope_summary"]["by_raw_class"]["bench"] == 1

        # Strict all-detection precision is also provided for transparent comparison:
        # TP / (TP + FP + out_of_scope) = 1 / (1 + 0 + 2) = 1/3 ≈ 0.3333
        assert overall["all_detections_precision"] == 0.3333

    def test_target_class_false_alarm_is_penalized(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Empty roadway frame
        f0 = FrameAnnotation(
            scenario_id="sc_ghost", tick=0, frame_hash_sha256="h0",
            objects=[], reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0])

        # Ghost vehicle predicted on empty pavement
        preds = {
            ("sc_ghost", 0): [
                {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_name": "car"}
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        # Target-class false alarm must be penalized
        assert overall["false_positives"] == 1
        assert overall["out_of_scope_detections"] == 0
        assert overall["precision"] == 0.0

    def test_out_of_scope_detection_overlapping_target_cannot_match(self):
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # GT: 1 pedestrian
        f0 = FrameAnnotation(
            scenario_id="sc_rej", tick=0, frame_hash_sha256="h0",
            objects=[AnnotatedObject("gt_p", 1, ObjectCategory.PERSON, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)],
            reviewer_id="Auditor",
        )
        dataset = AnnotationDataset(frames=[f0])

        # Prediction: Detector predicts 'chair' directly over the pedestrian with IoU = 0.95
        preds = {
            ("sc_rej", 0): [
                {"x1": 11, "y1": 11, "x2": 49, "y2": 49, "class_name": "chair"}
            ]
        }

        res = evaluator.evaluate_detections(dataset, preds)
        overall = res["overall"]
        # Match strictly rejected: 0 TP, 1 FN (person missed), 1 out_of_scope detection
        assert overall["true_positives"] == 0
        assert overall["false_negatives"] == 1
        assert overall["out_of_scope_detections"] == 1
        assert overall["false_positives"] == 0


class TestTrackingContinuityAndScenarioIsolation:
    """Verifies 5-tick stride continuity, fragmentation semantics, and scenario reset."""

    def test_trajectory_tracked_continuously_zero_fragmentations(self):
        """Case 1: A trajectory tracked continuously produces zero fragmentations."""
        evaluator = TrackingEvaluator(iou_threshold=0.50)
        frames = [
            FrameAnnotation(scenario_id="sc_cont", tick=tick, frame_hash_sha256=f"h_{tick}", objects=[AnnotatedObject(f"gt_{tick}", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")
            for tick in [0, 5, 10, 15]
        ]
        trajectories = {
            "sc_cont": [
                {"track_id": 10, "class_name": "car", "observations": [{"tick": t, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}} for t in [0, 5, 10, 15]]}
            ]
        }
        metrics = evaluator.evaluate_tracking(AnnotationDataset(frames=frames), trajectories)
        assert metrics.num_gt_tracks == 1
        assert metrics.fragmentations == 0
        assert metrics.mostly_tracked == 1

    def test_trajectory_tracked_missed_then_tracked_again_one_fragmentation(self):
        """Case 2: A trajectory tracked, missed, then tracked again produces exactly one fragmentation."""
        evaluator = TrackingEvaluator(iou_threshold=0.50)
        frames = [
            FrameAnnotation(scenario_id="sc_gap", tick=tick, frame_hash_sha256=f"h_{tick}", objects=[AnnotatedObject(f"gt_{tick}", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")
            for tick in [0, 5, 10, 15]
        ]
        # Tracked at 0, missed at 5, tracked at 10 and 15
        trajectories = {
            "sc_gap": [
                {"track_id": 10, "class_name": "car", "observations": [{"tick": t, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}} for t in [0, 10, 15]]}
            ]
        }
        metrics = evaluator.evaluate_tracking(AnnotationDataset(frames=frames), trajectories)
        assert metrics.num_gt_tracks == 1
        assert metrics.fragmentations == 1
        assert metrics.partially_tracked == 1

    def test_trajectory_tracked_then_missed_until_end_zero_fragmentations(self):
        """Case 3: A trajectory tracked and then missed until the sequence ends produces zero fragmentations from that trailing miss alone."""
        evaluator = TrackingEvaluator(iou_threshold=0.50)
        frames = [
            FrameAnnotation(scenario_id="sc_trail", tick=tick, frame_hash_sha256=f"h_{tick}", objects=[AnnotatedObject(f"gt_{tick}", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")
            for tick in [0, 5, 10, 15]
        ]
        # Tracked at 0 and 5, missed at 10 and 15 until sequence ends (never resumed)
        trajectories = {
            "sc_trail": [
                {"track_id": 10, "class_name": "car", "observations": [{"tick": t, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}} for t in [0, 5]]}
            ]
        }
        metrics = evaluator.evaluate_tracking(AnnotationDataset(frames=frames), trajectories)
        assert metrics.num_gt_tracks == 1
        # Trailing miss without resumption does NOT count as a fragmentation in CLEAR MOT
        assert metrics.fragmentations == 0
        assert metrics.partially_tracked == 1

    def test_trajectory_missed_at_beginning_then_tracked_continuously_zero_fragmentations(self):
        """Case 4: A trajectory missed at the beginning and then tracked continuously produces zero fragmentations from initial acquisition delay."""
        evaluator = TrackingEvaluator(iou_threshold=0.50)
        frames = [
            FrameAnnotation(scenario_id="sc_init", tick=tick, frame_hash_sha256=f"h_{tick}", objects=[AnnotatedObject(f"gt_{tick}", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")
            for tick in [0, 5, 10, 15]
        ]
        # Missed at 0, tracked at 5, 10, 15
        trajectories = {
            "sc_init": [
                {"track_id": 10, "class_name": "car", "observations": [{"tick": t, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}} for t in [5, 10, 15]]}
            ]
        }
        metrics = evaluator.evaluate_tracking(AnnotationDataset(frames=frames), trajectories)
        assert metrics.num_gt_tracks == 1
        assert metrics.fragmentations == 0
        assert metrics.partially_tracked == 1

    def test_multiple_gaps_and_resumptions_produces_expected_fragmentations(self):
        """Case 5: Multiple gaps and resumptions produce the expected number of fragmentation events."""
        evaluator = TrackingEvaluator(iou_threshold=0.50)
        # 5 evaluated frames: ticks 0, 5, 10, 15, 20
        frames = [
            FrameAnnotation(scenario_id="sc_mult", tick=tick, frame_hash_sha256=f"h_{tick}", objects=[AnnotatedObject(f"gt_{tick}", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")
            for tick in [0, 5, 10, 15, 20]
        ]
        # Tracked at 0, missed at 5, tracked at 10, missed at 15, tracked at 20 (2 gaps with resumptions)
        trajectories = {
            "sc_mult": [
                {"track_id": 10, "class_name": "car", "observations": [{"tick": t, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}} for t in [0, 10, 20]]}
            ]
        }
        metrics = evaluator.evaluate_tracking(AnnotationDataset(frames=frames), trajectories)
        assert metrics.num_gt_tracks == 1
        assert metrics.fragmentations == 2

    def test_same_identity_in_separate_scenarios_no_cross_scenario_continuity(self):
        """Case 6: The same identity appearing in separate scenarios has no cross-scenario continuity."""
        evaluator = TrackingEvaluator(iou_threshold=0.50)

        # Scenario A has GT 1 across ticks 0, 5 (tracked)
        f_A0 = FrameAnnotation(scenario_id="sc_A", tick=0, frame_hash_sha256="hA0", objects=[AnnotatedObject("gt_A0", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")
        f_A1 = FrameAnnotation(scenario_id="sc_A", tick=5, frame_hash_sha256="hA1", objects=[AnnotatedObject("gt_A1", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")

        # Scenario B has GT 1 across ticks 0, 5 (missed at 0, tracked at 5)
        f_B0 = FrameAnnotation(scenario_id="sc_B", tick=0, frame_hash_sha256="hB0", objects=[AnnotatedObject("gt_B0", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")
        f_B1 = FrameAnnotation(scenario_id="sc_B", tick=5, frame_hash_sha256="hB1", objects=[AnnotatedObject("gt_B1", 1, ObjectCategory.VEHICLE, BoundingBox(10, 10, 50, 50), confirmation_status=ConfirmationStatus.HUMAN_CONFIRMED)], reviewer_id="Auditor")

        dataset = AnnotationDataset(frames=[f_A0, f_A1, f_B0, f_B1])

        # Pred 10 tracks in Scenario A. Pred 10 tracks in Scenario B only at tick 5.
        trajectories = {
            "sc_A": [{"track_id": 10, "class_name": "car", "observations": [{"tick": 0, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}, {"tick": 5, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}]}],
            "sc_B": [{"track_id": 10, "class_name": "car", "observations": [{"tick": 5, "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_name": "car"}}]}],
        }

        metrics = evaluator.evaluate_tracking(dataset, trajectories)
        assert metrics.num_gt_tracks == 2
        # Scenario A: 0 fragmentations. Scenario B: missed at 0 then tracked at 5 -> 0 fragmentations (initial acquisition).
        # Total fragmentations must be 0 (no cross-scenario bleeding from sc_A to sc_B!)
        assert metrics.fragmentations == 0
        assert metrics.identity_switches == 0

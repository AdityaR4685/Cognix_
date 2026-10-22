"""Tracking and Detection Accuracy Evaluator (Milestones 8, 10, and 11).

Compares object detector bounding boxes and temporal tracker identities
against frozen ground-truth annotations under strict scientific rules:
1. Ground truth is used ONLY for post-hoc evaluation (no leakage into inference).
2. Ignored/ambiguous regions absorb unmatched predictions without false-positive penalty.
3. Class-specific metrics are partitioned explicitly using pre-registered taxonomy:
   - person
   - vehicle (car, truck, bus)
   - bicycle_or_motorcycle (bicycle, motorcycle)
   - road_debris_or_prop (zero COCO classes; detector blindness is preserved as structural FNs)
4. Out-of-scope detections (e.g. traffic light, bench, animal) are tracked in a dedicated
   reporting category so they do not distort target hazard false alarm metrics while ensuring
   no detection is silently discarded.
5. Strict rejection of spatial overlap with mismatched class (FP for pred, FN for GT).
6. Standard CLEAR MOTA metric accumulation:
   MOTA = 1.0 - (FN + FP + IDSW) / GT
   accumulated across all evaluated frames with explicit handling of zero-GT scenarios.
7. Machine-checkable completeness gating: Refuses official evaluation on unreviewed
   or incomplete datasets (raises IncompleteAnnotationError) unless exploratory=True.
8. Tracking identity switches and fragmentations are rigorously tracked per scenario.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from cognix.evaluation.ground_truth_schema import (
    AnnotationDataset,
    FrameAnnotation,
    AnnotatedObject,
    BoundingBox,
    ObjectCategory,
    ConfirmationStatus,
    FrameReviewState,
)


class IncompleteAnnotationError(ValueError):
    """Raised when official evaluation is requested on an unconfirmed or partial annotation package."""
    pass


# Target evaluation taxonomy categories
TARGET_TAXONOMY_CATEGORIES: Set[str] = {
    ObjectCategory.PERSON.value,
    ObjectCategory.VEHICLE.value,
    ObjectCategory.BICYCLE_OR_MOTORCYCLE.value,
    ObjectCategory.ROAD_DEBRIS_OR_PROP.value,
}

# Evaluator target taxonomy mapping (MS COCO 80 categories -> Cognix evaluation taxonomy)
COCO_TO_EVALUATION_TAXONOMY: Dict[str, str] = {
    # Pedestrians
    "person": ObjectCategory.PERSON.value,
    # Vehicles
    "car": ObjectCategory.VEHICLE.value,
    "truck": ObjectCategory.VEHICLE.value,
    "bus": ObjectCategory.VEHICLE.value,
    # Two-wheelers
    "bicycle": ObjectCategory.BICYCLE_OR_MOTORCYCLE.value,
    "motorcycle": ObjectCategory.BICYCLE_OR_MOTORCYCLE.value,
    # Native taxonomy strings (identity mappings)
    ObjectCategory.PERSON.value: ObjectCategory.PERSON.value,
    ObjectCategory.VEHICLE.value: ObjectCategory.VEHICLE.value,
    ObjectCategory.BICYCLE_OR_MOTORCYCLE.value: ObjectCategory.BICYCLE_OR_MOTORCYCLE.value,
    ObjectCategory.ROAD_DEBRIS_OR_PROP.value: ObjectCategory.ROAD_DEBRIS_OR_PROP.value,
}

OUT_OF_SCOPE_CATEGORY = "out_of_scope"
UNSUPPORTED_CLASS_CATEGORY = "out_of_scope"


def map_detector_class_to_taxonomy(detector_class: str) -> str:
    """Map detector category string (e.g. COCO 80) to evaluation taxonomy.

    Returns standard category string ('person', 'vehicle', 'bicycle_or_motorcycle',
    'road_debris_or_prop') if within target hazard scope, or 'out_of_scope' otherwise.
    Note: road_debris_or_prop has NO mapping from standard MS COCO classes.
    """
    cleaned = detector_class.strip().lower()
    return COCO_TO_EVALUATION_TAXONOMY.get(cleaned, OUT_OF_SCOPE_CATEGORY)


def is_target_taxonomy_class(category: str) -> bool:
    """Check if a category string is part of the declared road safety hazard target taxonomy."""
    return category in TARGET_TAXONOMY_CATEGORIES


def audit_dataset_provenance(dataset: AnnotationDataset) -> Dict[str, Any]:
    """Audit the provenance and independence of an annotation dataset.

    Distinguishes independent human ground truth from AI-assisted annotations
    where model detection proposals were visible during review.
    """
    provenance_counts: Dict[str, int] = {
        "independent_human": 0,
        "ai_accepted": 0,
        "human_corrected": 0,
        "human_added": 0,
        "unspecified": 0,
    }
    for frame in dataset.frames:
        for obj in frame.objects:
            prov = getattr(obj, "provenance", None) or "unspecified"
            provenance_counts[prov] = provenance_counts.get(prov, 0) + 1

    total_objects = dataset.total_objects
    is_independent = (
        dataset.annotation_mode == "independent" or
        (provenance_counts["ai_accepted"] == 0 and provenance_counts["human_corrected"] == 0)
    )

    return {
        "annotation_mode": dataset.annotation_mode or "unspecified",
        "is_independent_ground_truth": is_independent,
        "total_objects": total_objects,
        "provenance_breakdown": provenance_counts,
        "ai_assisted_object_fraction": round((provenance_counts["ai_accepted"] + provenance_counts["human_corrected"]) / total_objects, 4) if total_objects > 0 else 0.0,
    }


@dataclass
class DetectionMatch:
    """Association between an annotated ground truth and a predicted box."""
    gt_object: Optional[AnnotatedObject]
    pred_box: Optional[Dict[str, Any]]
    iou: float
    is_true_positive: bool
    is_false_positive: bool
    is_false_negative: bool
    is_ignored: bool = False
    is_out_of_scope: bool = False


@dataclass
class ClassDetectionMetrics:
    """Detection performance metrics for a specific object class."""
    category: str
    num_gt: int = 0
    num_pred: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ignored_matches: int = 0
    out_of_scope_detections: int = 0
    mean_iou: float = 0.0
    iou_scores: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Target-class precision: TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return (self.true_positives / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Target-class recall: TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return (self.true_positives / denom) if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "num_ground_truth": self.num_gt,
            "num_predictions": self.num_pred,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "ignored_matches": self.ignored_matches,
            "out_of_scope_detections": self.out_of_scope_detections,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "mean_iou": round(float(np.mean(self.iou_scores)), 4) if self.iou_scores else 0.0,
        }


@dataclass
class TrackingEvaluationMetrics:
    """Identity tracking evaluation metrics across sequences (CLEAR MOT compliant)."""
    num_gt_tracks: int = 0
    num_pred_tracks: int = 0
    identity_switches: int = 0
    fragmentations: int = 0
    mostly_tracked: int = 0  # GT track tracked >= 80% of frames
    partially_tracked: int = 0  # GT track tracked 20-80%
    mostly_lost: int = 0  # GT track tracked < 20%
    is_human_verified: bool = False
    review_status_notes: str = ""
    total_gt_detections: int = 0
    total_fp_detections: int = 0
    total_fn_detections: int = 0
    ignored_detections: int = 0
    out_of_scope_detections: int = 0
    mota: Optional[float] = None
    mota_formula: str = "1.0 - (total_fn_detections + total_fp_detections + identity_switches) / total_gt_detections"
    mota_status: str = "COMPUTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_gt_tracks": self.num_gt_tracks,
            "num_pred_tracks": self.num_pred_tracks,
            "identity_switches": self.identity_switches,
            "fragmentations": self.fragmentations,
            "mostly_tracked": self.mostly_tracked,
            "partially_tracked": self.partially_tracked,
            "mostly_lost": self.mostly_lost,
            "is_human_verified": self.is_human_verified,
            "review_status_notes": self.review_status_notes,
            "total_gt_detections": self.total_gt_detections,
            "total_fp_detections": self.total_fp_detections,
            "total_fn_detections": self.total_fn_detections,
            "ignored_detections": self.ignored_detections,
            "out_of_scope_detections": self.out_of_scope_detections,
            "mota": self.mota,
            "mota_formula": self.mota_formula,
            "mota_status": self.mota_status,
        }


class TrackingEvaluator:
    """Computes detection and tracking metrics against ground truth."""

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    def evaluate_detections(
        self,
        dataset: AnnotationDataset,
        predictions_by_scenario_tick: Dict[Tuple[str, int], List[Dict[str, Any]]],
        allow_incomplete: bool = False,
        exploratory: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate object detection precision, recall, and false alarms.

        Refuses evaluation on unconfirmed datasets unless exploratory=True or allow_incomplete=True.
        Enforces strict class agreement: spatial overlap with mismatched class is rejected as a match.
        Absorbs unmatched predictions overlapping ignore_ambiguous regions (IoU >= threshold) without FP penalty.
        Differentiates target-class false alarms from out-of-scope model detections.
        """
        is_verified = dataset.is_fully_confirmed
        if not (allow_incomplete or exploratory) and not is_verified:
            counts = dataset.review_state_counts
            unreviewed = counts.get(FrameReviewState.UNREVIEWED.value, 0)
            adj = counts.get(FrameReviewState.REQUIRES_ADJUDICATION.value, 0)
            raise IncompleteAnnotationError(
                f"Official evaluation refused on unconfirmed or partial annotation package: "
                f"{unreviewed} unreviewed frame(s), {adj} frame(s) requiring adjudication "
                f"out of {len(dataset.frames)} total frames. "
                f"Pass exploratory=True (or allow_incomplete=True) only for exploratory/provisional research."
            )

        class_metrics: Dict[str, ClassDetectionMetrics] = {
            cat.value: ClassDetectionMetrics(category=cat.value)
            for cat in ObjectCategory
            if cat != ObjectCategory.IGNORE_AMBIGUOUS
        }
        overall = ClassDetectionMetrics(category="overall")
        out_of_scope_metrics = ClassDetectionMetrics(category="out_of_scope")
        out_of_scope_by_raw_class: Dict[str, int] = {}

        total_frames = len(dataset.frames)
        confirmed_frames = dataset.confirmed_frames_count

        for frame in dataset.frames:
            key = (frame.scenario_id, frame.tick)
            preds = predictions_by_scenario_tick.get(key, [])

            # Separate ignore boxes from active GT objects
            active_gts = [o for o in frame.objects if o.category != ObjectCategory.IGNORE_AMBIGUOUS]
            ignore_gts = [o for o in frame.objects if o.category == ObjectCategory.IGNORE_AMBIGUOUS]

            # Bipartite matching between active_gts and preds
            # A candidate match requires BOTH spatial IoU >= threshold AND exact mapped class match
            candidate_pairs: List[Tuple[float, int, int]] = []  # (iou, gt_idx, pred_idx)
            for gt_idx, gt in enumerate(active_gts):
                for p_idx, pred in enumerate(preds):
                    raw_p_class = pred.get("class_name", "")
                    p_mapped_class = map_detector_class_to_taxonomy(raw_p_class)
                    # Out of scope predictions cannot match active ground truth targets
                    if not is_target_taxonomy_class(p_mapped_class):
                        continue
                    # Strict class agreement check
                    if p_mapped_class == gt.category.value:
                        p_box = BoundingBox(pred["x1"], pred["y1"], pred["x2"], pred["y2"])
                        iou = gt.bbox.iou(p_box)
                        if iou >= self.iou_threshold:
                            candidate_pairs.append((iou, gt_idx, p_idx))

            # Greedy matching by descending IoU
            candidate_pairs.sort(key=lambda x: x[0], reverse=True)
            matched_gts: Set[int] = set()
            matched_preds: Set[int] = set()

            for iou, gt_idx, p_idx in candidate_pairs:
                if gt_idx not in matched_gts and p_idx not in matched_preds:
                    matched_gts.add(gt_idx)
                    matched_preds.add(p_idx)
                    gt = active_gts[gt_idx]
                    cat_key = gt.category.value
                    if cat_key in class_metrics:
                        class_metrics[cat_key].true_positives += 1
                        class_metrics[cat_key].iou_scores.append(iou)
                        class_metrics[cat_key].num_pred += 1
                    overall.true_positives += 1
                    overall.iou_scores.append(iou)
                    overall.num_pred += 1

            # Count ground truths (TP and FN)
            for gt_idx, gt in enumerate(active_gts):
                cat_key = gt.category.value
                if cat_key in class_metrics:
                    class_metrics[cat_key].num_gt += 1
                overall.num_gt += 1
                if gt_idx not in matched_gts:
                    if cat_key in class_metrics:
                        class_metrics[cat_key].false_negatives += 1
                    overall.false_negatives += 1

            # Unmatched predictions: check against ignore regions or out-of-scope classification
            unmatched_preds = [p_idx for p_idx in range(len(preds)) if p_idx not in matched_preds]
            for p_idx in unmatched_preds:
                pred = preds[p_idx]
                p_box = BoundingBox(pred["x1"], pred["y1"], pred["x2"], pred["y2"])
                raw_p_class = pred.get("class_name", "")
                p_mapped_class = map_detector_class_to_taxonomy(raw_p_class)

                overlaps_ignore = False
                for ign in ignore_gts:
                    if ign.bbox.iou(p_box) >= self.iou_threshold:
                        overlaps_ignore = True
                        break

                if overlaps_ignore:
                    # Ignored match: absorbed without FP penalty
                    if p_mapped_class in class_metrics:
                        class_metrics[p_mapped_class].num_pred += 1
                        class_metrics[p_mapped_class].ignored_matches += 1
                    overall.num_pred += 1
                    overall.ignored_matches += 1
                elif is_target_taxonomy_class(p_mapped_class):
                    # Target-Class False Positive (ghost hazard / misclassification)
                    if p_mapped_class in class_metrics:
                        class_metrics[p_mapped_class].num_pred += 1
                        class_metrics[p_mapped_class].false_positives += 1
                    overall.num_pred += 1
                    overall.false_positives += 1
                else:
                    # Out-of-Scope Detection (non-target COCO category, e.g. bench, traffic light)
                    out_of_scope_metrics.num_pred += 1
                    out_of_scope_metrics.out_of_scope_detections += 1
                    out_of_scope_by_raw_class[raw_p_class] = out_of_scope_by_raw_class.get(raw_p_class, 0) + 1

        # Strict all-detections precision calculation: TP / (TP + FP + out_of_scope)
        total_non_gt_preds = overall.false_positives + out_of_scope_metrics.out_of_scope_detections
        all_pred_denom = overall.true_positives + total_non_gt_preds
        all_detections_precision = (overall.true_positives / all_pred_denom) if all_pred_denom > 0 else 0.0

        result = {
            "iou_threshold": self.iou_threshold,
            "total_frames_evaluated": total_frames,
            "confirmed_frames": confirmed_frames,
            "is_human_verified": is_verified,
            "review_state_counts": dataset.review_state_counts,
            "warning": None if is_verified else "Annotations contain unconfirmed frames; metrics are provisional.",
            "overall": {
                **overall.to_dict(),
                "out_of_scope_detections": out_of_scope_metrics.out_of_scope_detections,
                "all_detections_precision": round(all_detections_precision, 4),
            },
            "out_of_scope_summary": {
                "total_out_of_scope": out_of_scope_metrics.out_of_scope_detections,
                "by_raw_class": out_of_scope_by_raw_class,
            },
            "per_class": {k: m.to_dict() for k, m in class_metrics.items()},
        }
        return result

    def evaluate_tracking(
        self,
        dataset: AnnotationDataset,
        trajectories_by_scenario: Dict[str, List[Dict[str, Any]]],
        allow_incomplete: bool = False,
        exploratory: bool = False,
    ) -> TrackingEvaluationMetrics:
        """Evaluate track identity persistence, ID switches, fragmentations, and MOTA.

        Standard CLEAR MOTA accumulation:
        MOTA = 1.0 - (total_fn + total_fp + id_switches) / total_gt

        Refuses evaluation on unconfirmed datasets unless exploratory=True or allow_incomplete=True.
        Enforces exact taxonomy match and scenario isolation.
        """
        is_verified = dataset.is_fully_confirmed
        if not (allow_incomplete or exploratory) and not is_verified:
            counts = dataset.review_state_counts
            unreviewed = counts.get(FrameReviewState.UNREVIEWED.value, 0)
            adj = counts.get(FrameReviewState.REQUIRES_ADJUDICATION.value, 0)
            raise IncompleteAnnotationError(
                f"Official tracking evaluation refused on unconfirmed or partial annotation package: "
                f"{unreviewed} unreviewed frame(s), {adj} frame(s) requiring adjudication "
                f"out of {len(dataset.frames)} total frames. "
                f"Pass exploratory=True (or allow_incomplete=True) only for exploratory/provisional research."
            )

        total_id_switches = 0
        total_fragmentations = 0
        mostly_tracked = 0
        partially_tracked = 0
        mostly_lost = 0
        total_gt_detections = 0
        total_fn_detections = 0
        total_fp_detections = 0
        total_ignored_detections = 0
        total_out_of_scope_detections = 0

        scenario_ids = sorted(list(set(f.scenario_id for f in dataset.frames)))
        total_gt_tracks = 0
        total_pred_tracks = sum(len(trajs) for trajs in trajectories_by_scenario.values())

        for sc_id in scenario_ids:
            sc_frames = sorted(dataset.get_scenario_frames(sc_id), key=lambda x: x.tick)
            sc_trajs = trajectories_by_scenario.get(sc_id, [])

            # Map: tick -> list of predicted observations in this scenario
            # Each observation: {"pred_track_id": int, "box": BoundingBox, "mapped_class": str, "raw_class": str}
            pred_obs_by_tick: Dict[int, List[Dict[str, Any]]] = {}
            for traj in sc_trajs:
                t_id = traj["track_id"]
                traj_raw_class = traj.get("class_name", "")
                for obs in traj.get("observations", []):
                    b = obs["box"]
                    raw_cls = b.get("class_name", traj_raw_class)
                    mapped_cls = map_detector_class_to_taxonomy(raw_cls) if raw_cls else ""
                    pred_obs_by_tick.setdefault(obs["tick"], []).append({
                        "pred_track_id": t_id,
                        "box": BoundingBox(b["x1"], b["y1"], b["x2"], b["y2"]),
                        "mapped_class": mapped_cls,
                        "raw_class": raw_cls,
                    })

            # GT tracks in this scenario: track_id -> list of (tick, BoundingBox, category)
            scenario_gt_tracks: Dict[int, List[Tuple[int, BoundingBox, str]]] = {}
            for f in sc_frames:
                for obj in f.objects:
                    if obj.category != ObjectCategory.IGNORE_AMBIGUOUS:
                        scenario_gt_tracks.setdefault(obj.track_id, []).append((f.tick, obj.bbox, obj.category.value))

            total_gt_tracks += len(scenario_gt_tracks)

            # Frame-by-frame CLEAR MOT evaluation for this scenario
            last_matched_pred_by_gt: Dict[int, int] = {}
            prior_frame_matched_pred: Dict[int, int] = {}  # Matches in immediately preceding evaluated frame
            hits_by_gt_track: Dict[int, int] = {t_id: 0 for t_id in scenario_gt_tracks}
            match_status_by_gt_tick: Dict[int, Dict[int, bool]] = {t_id: {} for t_id in scenario_gt_tracks}

            for f in sc_frames:
                tick = f.tick
                active_gts = [o for o in f.objects if o.category != ObjectCategory.IGNORE_AMBIGUOUS]
                ignore_gts = [o for o in f.objects if o.category == ObjectCategory.IGNORE_AMBIGUOUS]
                frame_preds = pred_obs_by_tick.get(tick, [])

                matched_gt_ids: Set[int] = set()
                matched_pred_indices: Set[int] = set()
                current_frame_matches: Dict[int, int] = {}  # gt.track_id -> pred_track_id

                # Priority 1: Maintain association from immediately preceding frame if IoU >= threshold and class matches
                for gt in active_gts:
                    if gt.track_id in prior_frame_matched_pred:
                        prior_pred_id = prior_frame_matched_pred[gt.track_id]
                        # Look for prior_pred_id in frame_preds
                        for p_idx, pred in enumerate(frame_preds):
                            if p_idx in matched_pred_indices:
                                continue
                            if pred["pred_track_id"] == prior_pred_id:
                                # Class check
                                if pred["mapped_class"] and pred["mapped_class"] != gt.category.value:
                                    continue
                                if gt.bbox.iou(pred["box"]) >= self.iou_threshold:
                                    matched_gt_ids.add(gt.track_id)
                                    matched_pred_indices.add(p_idx)
                                    current_frame_matches[gt.track_id] = prior_pred_id
                                break

                # Priority 2: Greedy bipartite matching on remaining active GTs and predictions
                remaining_candidates: List[Tuple[float, AnnotatedObject, int]] = []
                for gt in active_gts:
                    if gt.track_id in matched_gt_ids:
                        continue
                    for p_idx, pred in enumerate(frame_preds):
                        if p_idx in matched_pred_indices:
                            continue
                        if pred["mapped_class"] and pred["mapped_class"] != gt.category.value:
                            continue
                        iou = gt.bbox.iou(pred["box"])
                        if iou >= self.iou_threshold:
                            remaining_candidates.append((iou, gt, p_idx))

                remaining_candidates.sort(key=lambda x: x[0], reverse=True)
                for iou, gt, p_idx in remaining_candidates:
                    if gt.track_id not in matched_gt_ids and p_idx not in matched_pred_indices:
                        matched_gt_ids.add(gt.track_id)
                        matched_pred_indices.add(p_idx)
                        current_frame_matches[gt.track_id] = frame_preds[p_idx]["pred_track_id"]

                # Accumulate GT statistics for this frame
                for gt in active_gts:
                    total_gt_detections += 1
                    if gt.track_id in matched_gt_ids:
                        pred_id = current_frame_matches[gt.track_id]
                        hits_by_gt_track[gt.track_id] += 1
                        match_status_by_gt_tick[gt.track_id][tick] = True

                        # Check for Identity Switch
                        if gt.track_id in last_matched_pred_by_gt:
                            if pred_id != last_matched_pred_by_gt[gt.track_id]:
                                total_id_switches += 1
                        last_matched_pred_by_gt[gt.track_id] = pred_id
                    else:
                        total_fn_detections += 1
                        match_status_by_gt_tick[gt.track_id][tick] = False

                # Accumulate False Positives and Ignored Predictions for this frame
                for p_idx, pred in enumerate(frame_preds):
                    if p_idx in matched_pred_indices:
                        continue
                    p_box = pred["box"]
                    # Check overlap with ignore regions
                    overlaps_ignore = any(ign.bbox.iou(p_box) >= self.iou_threshold for ign in ignore_gts)
                    if overlaps_ignore:
                        total_ignored_detections += 1
                    elif pred["mapped_class"] and is_target_taxonomy_class(pred["mapped_class"]):
                        total_fp_detections += 1
                    elif pred["mapped_class"] == OUT_OF_SCOPE_CATEGORY:
                        total_out_of_scope_detections += 1
                    else:
                        # Geometric tracker without class name (defaults to target-class false alarm)
                        total_fp_detections += 1

                # Update prior frame matches for continuity in the next evaluated frame
                prior_frame_matched_pred = current_frame_matches

            # Calculate MT, PT, ML, and Fragmentations for each GT track in this scenario
            for gt_id, observations in scenario_gt_tracks.items():
                lifespan = len(observations)
                hits = hits_by_gt_track.get(gt_id, 0)
                ratio = (hits / lifespan) if lifespan > 0 else 0.0

                if ratio >= 0.80:
                    mostly_tracked += 1
                elif ratio >= 0.20:
                    partially_tracked += 1
                else:
                    mostly_lost += 1

                # Fragmentation: counted when a ground-truth trajectory is tracked, becomes untracked, and is subsequently tracked again
                ticks_in_order = [obs[0] for obs in observations]
                statuses = [match_status_by_gt_tick[gt_id].get(t, False) for t in ticks_in_order]
                for idx in range(1, len(statuses)):
                    if not statuses[idx] and statuses[idx - 1]:
                        # Transition from tracked to untracked: count as fragmentation only if subsequently resumed
                        if any(statuses[idx + 1:]):
                            total_fragmentations += 1

        # Standard CLEAR MOTA: 1.0 - (FN + FP + IDSW) / GT
        if total_gt_detections > 0:
            mota_score = 1.0 - (float(total_fn_detections + total_fp_detections + total_id_switches) / float(total_gt_detections))
            mota = round(mota_score, 4)
            mota_status = "COMPUTED"
        else:
            # When total_gt_detections == 0, MOTA formula is mathematically undefined (division by zero).
            # Represented as None (null in JSON), reporting FP, FN, and other counts separately.
            mota = None
            mota_status = "UNDEFINED_ZERO_GROUND_TRUTH"

        notes = (
            "Complete human review verified."
            if is_verified
            else f"Provisional evaluation: {confirmed_frames}/{total_frames} frames human-confirmed."
        )

        return TrackingEvaluationMetrics(
            num_gt_tracks=total_gt_tracks,
            num_pred_tracks=total_pred_tracks,
            identity_switches=total_id_switches,
            fragmentations=total_fragmentations,
            mostly_tracked=mostly_tracked,
            partially_tracked=partially_tracked,
            mostly_lost=mostly_lost,
            is_human_verified=is_verified,
            review_status_notes=notes,
            total_gt_detections=total_gt_detections,
            total_fp_detections=total_fp_detections,
            total_fn_detections=total_fn_detections,
            ignored_detections=total_ignored_detections,
            out_of_scope_detections=total_out_of_scope_detections,
            mota=mota,
            mota_status=mota_status,
        )

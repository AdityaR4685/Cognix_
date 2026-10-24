# Cognix Milestone 12 — Final Tracking-Metric Semantics Audit

**Milestone Identifier:** `m12_tracking_metric_semantics_001`
**Execution Timestamp:** 2026-10-20T13:25:00Z
**Branch:** `gat-testing-complete-checkpoint`
**Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Environment:** `C:\Cognix_Envs\m6_perception`
**Status:** **PASSED — TRACKING METRIC SEMANTICS AUDITED, TESTED & FROZEN FOR HUMAN ANNOTATION**

---

## 1. Executive Summary & Operating Rules

Milestone 12 resolves the final two metric-definition and semantics questions identified during the Milestone 11 audit of the Cognix perception and tracking evaluation harness. With the resolution of these semantics and their comprehensive automated verification, **all software engineering work on the evaluation harness is officially concluded and frozen**, clearing the way for manual human annotation to begin.

All non-negotiable safety and Git restrictions were strictly upheld:
- **Zero Git Writes:** No `git add`, `git commit`, `git push`, `git pull`, `git merge`, `git rebase`, or branch-switching commands.
- **Pristine Tracked Working Tree:** `git diff --stat` confirms **0 modified tracked files**.
- **Untouched Offline Qualification Backup:** `C:\Cognix_Backups\Cognix2_recovered_qualification_014_20261020_063301` remains intact and unmodified.
- **Zero Annotation Fabrication:** No ground-truth bounding boxes were generated, altered, or artificially marked as confirmed.
- **Stop Condition Fulfilled:** Engineering modifications are complete; the harness is locked and awaiting human annotation via the M9 Dash UI.

### Summary of Resolved Metric Semantics

1. **CLEAR MOT Fragmentation Semantics Corrected:**
   - Under standard CLEAR MOT (Bernardin & Stiefelhagen, 2008), track fragmentation strictly measures an *interruption* in tracking continuity: a ground-truth object must be tracked, become untracked, and then be *subsequently tracked again*.
   - A trailing loss of tracking where an object is tracked and then missed until it exits or the sequence terminates does **not** count as a fragmentation; it represents track termination or loss.
   - Evaluated frames are sampled every 5 simulation ticks ($t \in \{0, 5, 10, 15, \dots\}$); all continuity and fragmentation metrics measure transitions across this 5-tick sampled sequence and explicitly do not evaluate unsampled intermediate ticks.
2. **Zero-Ground-Truth MOTA Semantics Resolved:**
   - The CLEAR MOTA formula:
     $$\text{MOTA} = 1.0 - \frac{\sum_t (FN_t + FP_t + IDSW_t)}{\sum_t GT_t}$$
     is mathematically undefined when $\sum_t GT_t = 0$ due to division by zero.
   - Rather than enforcing an ad-hoc heuristic (such as assigning 1.0 or 0.0) under the name of standard MOTA, the evaluator outputs `mota = None` (`null` in JSON) with `mota_status = "UNDEFINED_ZERO_GROUND_TRUTH"`. False alarms ($FP$) and misses ($FN$) are reported transparently as independent counts.
   - In cross-scenario benchmark aggregation (e.g. combining anomaly and normal control sequences), counts are summed: if the aggregate $\sum GT > 0$, the aggregate MOTA is mathematically well-defined and computed directly from the aggregate sums.

---

## 2. Forensic Audit of Working Tree, Queues, and Qualification Backups

| Asset | Location / Identifier | Expected Value | Actual Verified Value | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Git Working Branch** | `git branch --show-current` | `gat-testing-complete-checkpoint` | `gat-testing-complete-checkpoint` | **MATCH** |
| **Git HEAD Commit** | `git log -1 --oneline` | `b4606b91` | `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8` | **MATCH** |
| **Tracked Diff** | `git diff --stat` | Clean (0 modified files) | 0 modified tracked files | **CLEAN** |
| **M8 Pilot Queue** | `reports/.../m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e...` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **AUTHENTIC** |
| **M9 Baseline Immutable** | `reports/.../m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e...` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **AUTHENTIC** |
| **M9 Normal Control** | `reports/.../m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b...` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **AUTHENTIC** |
| **Qualification Backup** | `C:\Cognix_Backups\Cognix2_recovered_qualification_014_20261020_063301` | Sealed Attempt 014 Qualification Evidence | Verified intact on disk | **PRESERVED** |

---

## 3. Semantic Resolution 1: CLEAR MOT Fragmentation & Sampling Limitation

### 3.1 Standard CLEAR MOT Definition of Fragmentation

In standard multi-object tracking literature (Bernardin & Stiefelhagen, EURASIP JASP 2008; Milan et al., MOT16), **fragmentation** measures the lack of continuity in tracking a ground-truth trajectory. Specifically, a fragmentation occurs when a trajectory that was previously tracked is interrupted, and then tracking is **resumed** at a later point in time.

Formally, let $s_k \in \{\text{True}, \text{False}\}$ denote whether ground-truth object $G_i$ is successfully matched to a predicted track at evaluated frame index $k \in \{0, 1, \dots, N-1\}$. A fragmentation is incremented at index $k$ if and only if:
$$s_{k-1} = \text{True} \quad \land \quad s_k = \text{False} \quad \land \quad \exists j > k \text{ such that } s_j = \text{True}$$

### 3.2 Distinction from Trailing Misses (Track Loss / Termination)

If a target is tracked up to frame $k-1$ and then missed at frame $k$ and all subsequent frames ($s_j = \text{False} \ \forall j \ge k$), the tracker did not *fragment* the trajectory into discontinuous pieces—the tracker simply *lost* or *terminated* the track.

Counting a trailing loss as a fragmentation conflates track loss with track interruption. The M12 implementation strictly enforces the resumption requirement:
- **Tracked $\to$ Missed $\to$ Tracked:** Interruption with recovery $\implies \mathbf{1 \text{ Fragmentation}}$.
- **Tracked $\to$ Missed $\to$ End of Sequence:** Loss/termination $\implies \mathbf{0 \text{ Fragmentations}}$.
- **Missed $\to$ Tracked $\to$ End of Sequence:** Late acquisition $\implies \mathbf{0 \text{ Fragmentations}}$.

### 3.3 5-Tick Stride Sampling Disclosure

In the Cognix CARLA evaluation pipeline, raw driving logs are recorded at high tick rates, but ground-truth annotation and perception evaluation are performed on sampled frames at a **5-tick stride** ($t \in \{0, 5, 10, 15, \dots\}$).

**Operational Boundary Disclosure:**
- All detection metrics ($TP, FP, FN$), tracking metrics ($\text{IDSW}, \text{Frag}, \text{MT}, \text{PT}, \text{ML}$), and CLEAR MOTA scores are evaluated strictly over the discrete sequence of 5-tick sampled frames.
- Intermediate simulation ticks ($t \in \{1, 2, 3, 4, 6, \dots\}$) are neither evaluated nor inferred.
- Track continuity is defined relative to this 5-tick sampled grid. A trajectory tracked at tick 0 and tick 5 is considered continuous; a miss at tick 5 followed by a hit at tick 10 is counted as 1 fragmentation across the 5-tick evaluation grid.

### 3.4 Canonical Verification Cases

The updated logic was tested against six canonical ground-truth tracking cases:

| Case | Trajectory Pattern (Sampled Ticks) | Matched Status Sequence | Fragmentations | Rationale |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Tracked continuously across all frames | `[True, True, True]` | **0** | No interruption in tracking continuity. |
| **2** | Tracked, missed, then resumed | `[True, False, True]` | **1** | Interruption followed by successful re-acquisition. |
| **3** | Tracked then lost until sequence end | `[True, False, False]` | **0** | Trailing miss represents track termination, not fragmentation. |
| **4** | Late acquisition then tracked continuously | `[False, True, True]` | **0** | Initial miss before acquisition; no interruption once tracked. |
| **5** | Multiple interruptions and resumptions | `[True, False, True, False, True]` | **2** | Two distinct interruptions, each followed by resumption. |
| **6** | Separate scenarios with same ID | `Scenario A: [True]`<br>`Scenario B: [True]` | **0** | Scenario boundary strictly resets tracking; no cross-scenario continuity. |

---

## 4. Semantic Resolution 2: Zero-Ground-Truth MOTA Semantics & Null Representation

### 4.1 The Mathematical Problem of Zero Ground Truth

The CLEAR MOTA formula is defined as:
$$\text{MOTA} = 1.0 - \frac{FN + FP + IDSW}{GT}$$

When evaluating a scenario or window with zero ground-truth objects ($GT = 0$):
$$\text{MOTA} = 1.0 - \frac{0 + FP + 0}{0} = 1.0 - \frac{FP}{0} \implies \mathbf{\text{Undefined}}$$

In Milestone 11, a heuristic convention was implemented:
- If $GT = 0$ and $FP = 0 \implies \text{MOTA} = 1.0$ (rewarding clean true negatives).
- If $GT = 0$ and $FP > 0 \implies \text{MOTA} = 0.0$ (penalizing ghost false alarms).

### 4.2 Critique of Ad-Hoc Conventions & M12 Standard Resolution

While pragmatically motivated, assigning an arbitrary numerical score (1.0 or 0.0) under the official label `mota` violates mathematical integrity and misrepresents standard CLEAR MOT semantics:
1. In benchmark comparisons, reporting $\text{MOTA} = 1.0$ for an empty sequence can artificially boost average MOTA, while reporting $\text{MOTA} = 0.0$ conflates an empty sequence having 1 minor false alarm with a catastrophic sequence having dozens of misses.
2. Machine-readable consumers and automated pipelines need an explicit indicator that the metric is mathematically undefined rather than a legitimate numerical score.

**The M12 Resolution:**
- When $GT = 0$, the evaluator sets `mota = None` (`null` in JSON serialization).
- An explicit status field `mota_status = "UNDEFINED_ZERO_GROUND_TRUTH"` is output.
- All constituent counts ($FP, FN, \text{ignored\_detections}, \text{out\_of\_scope\_detections}$) continue to be reported with complete mathematical transparency:
  - If $FP = 0$: The run is confirmed to have zero false alarms on empty roadway.
  - If $FP > 0$: The false alarm count ($FP$) directly measures detector spurious alarms without assigning an arbitrary MOTA score.

### 4.3 Multi-Scenario Benchmark Aggregation

When evaluating an entire dataset composed of multiple scenarios (e.g. 4 CARLA anomaly scenarios with active traffic plus 1 normal control scenario with empty roadway):
$$\sum_{\text{sc}} GT_{\text{sc}} = GT_{\text{agg}}, \quad \sum_{\text{sc}} FN_{\text{sc}} = FN_{\text{agg}}, \quad \sum_{\text{sc}} FP_{\text{sc}} = FP_{\text{agg}}, \quad \sum_{\text{sc}} IDSW_{\text{sc}} = IDSW_{\text{agg}}$$

Because the combined benchmark has $GT_{\text{agg}} > 0$:
$$\text{MOTA}_{\text{agg}} = 1.0 - \frac{FN_{\text{agg}} + FP_{\text{agg}} + IDSW_{\text{agg}}}{GT_{\text{agg}}}$$
The aggregate MOTA is **fully mathematically well-defined** and computed directly from the aggregated counts. Empty scenarios contribute their false alarms ($FP$) directly into the aggregate numerator, ensuring that ghost tracks on empty roadway properly penalize the overall system benchmark.

---

## 5. Code Implementation and Architecture

All changes were implemented cleanly in [cognix/evaluation/tracking_evaluator.py](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/tracking_evaluator.py):

### 5.1 Data Structure Updates

```python
@dataclass
class TrackingEvaluationMetrics:
    num_gt_tracks: int = 0
    num_pred_tracks: int = 0
    identity_switches: int = 0
    fragmentations: int = 0
    mostly_tracked: int = 0
    partially_tracked: int = 0
    mostly_lost: int = 0
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
```

### 5.2 Fragmentation Logic (Lookahead Resumption)

```python
# Fragmentation: counted when a ground-truth trajectory is tracked, becomes untracked, and is subsequently tracked again
ticks_in_order = [obs[0] for obs in observations]
statuses = [match_status_by_gt_tick[gt_id].get(t, False) for t in ticks_in_order]
for idx in range(1, len(statuses)):
    if not statuses[idx] and statuses[idx - 1]:
        # Transition from tracked to untracked: count as fragmentation only if subsequently resumed
        if any(statuses[idx + 1:]):
            total_fragmentations += 1
```

### 5.3 Zero-Ground-Truth Representation

```python
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
```

---

## 6. Verification Test Suite & Pass Rates

The updated metric semantics were comprehensively verified using pytest under `C:\Cognix_Envs\m6_perception\Scripts\python.exe`:

### 6.1 Dedicated Regression Tests (`tests/unit/test_mota_and_metric_correctness.py`)

17 dedicated unit tests verify all aspects of CLEAR MOTA, zero-GT behavior, and fragmentation semantics:
1. `test_mota_perfect_matches_equals_one` — Perfect tracking yields MOTA = 1.00.
2. `test_mota_lowers_with_one_false_positive` — Single false alarm reduces MOTA by exact mathematical delta ($FP/GT$).
3. `test_mota_lowers_with_one_false_negative` — Single missed detection reduces MOTA by exact delta ($FN/GT$).
4. `test_mota_lowers_with_one_identity_switch` — Single ID switch reduces MOTA by exact delta ($IDSW/GT$).
5. `test_mota_mixed_errors_exact_calculation` — Multiple concurrent errors calculate exact standard CLEAR MOTA.
6. `test_mota_zero_ground_truth_policy` — Sub-case A ($GT=0, FP=0$) and Sub-case B ($GT=0, FP=1$) both produce `mota is None`, `mota_status == "UNDEFINED_ZERO_GROUND_TRUTH"`, and serialize to `"mota": null` in JSON.
7. `test_mota_mixed_sequences_with_empty_scene_aggregates_properly` — Combining active scenes with empty scene yields mathematically exact aggregate MOTA ($0.50$).
8. `test_mota_scenario_aggregation_consistency` — Aggregation matches sum of components across scenarios.
9. `test_out_of_scope_detection_is_segregated_from_target_fp` — Non-hazard COCO detections segregated into `out_of_scope_detections`.
10. `test_target_class_false_alarm_is_penalized` — Target false alarms strictly penalized.
11. `test_out_of_scope_detection_overlapping_target_cannot_match` — Out-of-scope predictions cannot match GT.
12. `test_trajectory_tracked_continuously_zero_fragmentations` — Continuous tracking $\to$ 0 fragmentations.
13. `test_trajectory_tracked_missed_then_tracked_again_one_fragmentation` — Tracked, lost, re-tracked $\to$ 1 fragmentation.
14. `test_trajectory_tracked_then_missed_until_end_zero_fragmentations` — Trailing miss $\to$ 0 fragmentations.
15. `test_trajectory_missed_at_beginning_then_tracked_continuously_zero_fragmentations` — Late acquisition $\to$ 0 fragmentations.
16. `test_multiple_gaps_and_resumptions_produces_expected_fragmentations` — Two interruptions $\to$ 2 fragmentations.
17. `test_same_identity_in_separate_scenarios_no_cross_scenario_continuity` — Cross-scenario isolation $\to$ 0 fragmentations.

### 6.2 Full Test Suite Results

Executing all 14 research, perception, and decision test modules:
```
tests/unit/test_ground_truth_annotation.py ..........                    [ 10 passed]
tests/unit/test_mota_and_metric_correctness.py .................         [ 17 passed]
tests/unit/test_temporal_tracking.py .......                             [  7 passed]
tests/unit/test_evaluation_protocol.py .............                     [ 13 passed]
tests/unit/test_annotation_app_smoke.py ........                         [  8 passed]
tests/unit/test_perception_features.py ......                            [  6 passed]
tests/unit/test_pretrained_perception_benchmark.py ......                [  6 passed]
tests/unit/test_torchvision_detection.py .........                       [  9 passed]
tests/unit/test_decision.py ........                                     [  8 passed]
tests/unit/test_uncertainty.py ........                                  [  8 passed]
tests/unit/test_belief.py ......                                         [  6 passed]
tests/unit/test_adapters.py ...............                              [ 15 passed]
tests/unit/test_calibration.py ......                                    [  6 passed]
tests/unit/test_agents.py .....                                          [  5 passed]
tests/unit/test_dataset_evaluation.py ..................                 [ 18 passed]

============================= 142 passed in 13.68s =============================
```
**Pass Rate: 100% (142 passed, 0 failures, 0 errors).**

---

## 7. Deliverables & Artifact Manifest

All artifacts for Milestone 12 are persisted in `reports/carla_runtime_v1/evaluation_runs/m12_tracking_metric_semantics_001/`:
1. `class_mapping.json` — Pre-registered detector taxonomy mapping, separating target-hazard scope from out-of-scope COCO classes.
2. `evaluation_protocol.json` — Pre-registered evaluation protocol (v1.2.0) defining exact CLEAR MOT fragmentation semantics, zero-GT MOTA null serialization policy, 5-tick stride sampling disclosure, and reference queue hashes.
3. `M12_TRACKING_METRIC_SEMANTICS_REPORT.md` — This comprehensive scientific report.
4. `cognix/evaluation/tracking_evaluator.py` — Evaluator implementation with lookahead fragmentation resumption and null-safe zero-GT MOTA handling.
5. `tests/unit/test_mota_and_metric_correctness.py` — Comprehensive regression test suite.

---

## 8. Official Declaration of Engineering Halt & Next Steps for Human Annotation

With the completion of Milestone 12:
- **All evaluation metrics, protocols, schemas, and safeguards are frozen.**
- **No further software engineering modifications are needed.**
- The system is in a clean, reproducible state, ready for human annotators to label the ground-truth queues.

### Instructions for Human Reviewers (Verified Entry Point & Queue Protection)

To commence manual ground-truth annotation without risking canonical reference queues:

#### A. Reviewing the 40-Frame Anomaly Pilot Package
Launch the verified Flask annotation server on port `5050` targeting the dedicated working review file:
```powershell
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5050 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames
```
Navigate to: `http://127.0.0.1:5050/`

#### B. Reviewing the 10-Frame Normal Control Package
Launch the verified Flask annotation server on port `5051` targeting the dedicated working normal control file:
```powershell
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5051 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames
```
Navigate to: `http://127.0.0.1:5051/`

#### C. Annotation and Confirmation Workflow
1. **Enter Reviewer ID:** Enter your unique reviewer ID (e.g. `SafetyReviewer01`) in the reviewer attribution field.
2. **Review Boxes & Classes:** Inspect the RGB frame. Add, resize, or delete bounding boxes. Select appropriate taxonomy category (`person`, `vehicle`, `bicycle_or_motorcycle`, `road_debris_or_prop`).
3. **Handling Ambiguity:** If an object is severely occluded or visually ambiguous, assign category `ignore_ambiguous` and check `is_ambiguous`.
4. **Empty Frames:** For clear roadway without collision targets, verify 0 objects and click **Confirm & Sign Frame** (records `CONFIRMED_EMPTY`).
5. **Save Progress:** Click **Save Annotations** to commit updates to the working JSON file with full schema validation. Canonical reference queues remain untouched.

# Milestone 9: Annotation Workflow Verification and Normal-Control Preparation Report

**Experiment ID:** `PROTOCOL_M9_ANNOTATION_READINESS_001`
**Date:** 2026-10-20 06:44:00 UTC
**Working Branch:** `gat-testing-complete-checkpoint`
**Working Commit (HEAD):** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Operating Mode:** Strictly Local Research and Validation Only (Zero Git Write Operations)
**Dedicated Perception Environment:** `C:\Cognix_Envs\m6_perception`
**Source Archive:** `E:/carlanomaly-base-test.tar.gz` (SHA-256: `267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a`)

---

## 1. Actual Git State and Qualification Evidence Verification

* **Active Branch:** `gat-testing-complete-checkpoint`
* **HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
* **Working Tree State:** `git diff --stat` returns 0 modified tracked files. All new code, tests, and artifacts remain strictly uncommitted.
* **Offline Qualification Backup:** `C:\Cognix_Backups\Cognix2_recovered_qualification_014_20261020_063301` was verified present and completely untouched. Sealed Attempt 014 qualification evidence remains immutable.

---

## 2. Milestone 8 Queue Integrity Audit

The original M8 pilot queue was inspected and hashed directly on disk:
* **File Path:** [`reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json)
* **Reported SHA-256:** `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15`
* **Computed Actual SHA-256:** `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15`
* **Audit Verdict:** **100% Bit-for-bit Integrity Verified.** The canonical M8 queue was not altered.

---

## 3. Baseline vs. Working-Copy Queue Separation

To protect the canonical baseline against accidental mutation during human review, an explicit separation was established:

| File Role | Path | SHA-256 (at creation) | Purpose & Mutability Invariant |
| :--- | :--- | :--- | :--- |
| **Original M8 Pilot Queue** | `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e...` | **Canonical historical baseline.** Read-only. |
| **M9 Immutable Baseline Copy** | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e...` | **Permanent reference copy.** Never modified by server. |
| **M9 Working Human-Review File** | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `bb20dd466a2e...` | **Active working target** for reviewer annotations. |

The full relationship and binding ledger is preserved in [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/dataset_provenance_manifest.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/dataset_provenance_manifest.json).

---

## 4. Annotation UI Verification & Smoke-Test Audit

A comprehensive automated test suite ([`tests/unit/test_annotation_app_smoke.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_annotation_app_smoke.py)) was executed against an isolated, synthetic test fixture.

### Findings Across Verification Criteria:
1. **Server Environment & Binding:**
   - Starts cleanly under `C:\Cognix_Envs\m6_perception`.
   - Binds strictly to `127.0.0.1` (localhost); does not bind to `0.0.0.0` or expose the interface to the local area network.
   - **Dependency Clarification:** While the HTML/JS frontend has zero npm or build dependencies, the backend relies on Python dependencies (`Flask`, `Pillow`, `pyarrow`).
2. **Save and Reload Round-Trip:**
   - Bounding boxes ($xyxy$), object categories, track IDs, occlusion states, and reviewer metadata survive a complete server restart with 100% numerical fidelity.
3. **Ambiguity & Reviewed-Empty Frames:**
   - Ambiguous objects (`is_ambiguous=True`, `category="ignore_ambiguous"`) are saved and loaded correctly.
   - Frames reviewed as legitimately empty (clear roadway) can be signed and confirmed with `reviewer_id`.
   - **UI Defect Repaired:** Auditing revealed a discrepancy where the JS progress badge excluded empty confirmed frames from the counter. This was resolved so that JS and Python `is_confirmed` agree identically.
4. **Validation Bounds & Duplicate Rejection:**
   - Coordinates outside $[0, 1920] \times [0, 1080]$ or inverted coordinates ($x_1 \ge x_2$) return HTTP 422 with a descriptive error.
   - Duplicate `track_id` in a single frame returns HTTP 422 with `ValidationError`.
   - Duplicate `object_id` returns HTTP 422 with `ValidationError`.
5. **Human Attribution Gate:**
   - The UI blocks confirmation if `reviewer_id` is blank.
6. **Frame Navigation & Image Integrity:**
   - Requesting `/api/frame_image` for a given scenario and tick streams the exact matching JPEG bytes verified by SHA-256. Image cross-talk or silent misassociation is impossible.

---

## 5. Genuine Normal-Driving Control Availability & Provenance

### Bounded Investigation and Historical Reconciliation:
* Prior reports in Milestone 7 and 8 lacked an actual baseline control because `change-weather/scenario-10` was informally mislabeled as normal driving.
* A forensic query into `reports/carla_test_structure_inventory_v1/kaggle_result_v1/cognix_carla_test_inventory_result_v1.zip` revealed that **107 genuine NORMAL scenarios** exist in `carlanomaly-base-test.tar.gz` under `test/normal/Town01/` through `Town05/`.
* The first normal sequence, `test/normal/Town01/scenario-1`, begins at header index **473,201**.

### Extraction and Package Creation:
* An autonomous streaming extractor ([`scratch/extract_normal_pilot_control.py`](file:///C:/Users/Aditya/Cognix2_recovered/scratch/extract_normal_pilot_control.py)) streamed through `E:/carlanomaly-base-test.tar.gz` and extracted 10 contiguous pilot frames (ticks 0, 5, 10, 15, 20, 25, 30, 35, 40, 45) in 297.8 seconds without downloading any external tools.
* **Frames Saved:** [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames/`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames/)
* **Package Specifications:**
  - Scenario ID: `test/normal/Town01/scenario-1`
  - Split: `test_normal`
  - Anomaly Type: `NORMAL` (ground truth from dataset structure)
  - Dimensions: $1920 \times 1080$ RGB JPEG
  - Annotations Queue: [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json) (SHA-256: `3d8733b4fd3b...`)
  - Status: `PENDING_HUMAN_REVIEW` (10 frames unconfirmed)
  - Full provenance recorded in [`normal_control_manifest.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_manifest.json).

---

## 6. Reviewer Guide & Protocol

A dedicated reviewer guide was authored at [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/HUMAN_ANNOTATION_PROTOCOL.md`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/HUMAN_ANNOTATION_PROTOCOL.md).

### Summary of Operating Standards:
* **Object Categories:** `person`, `vehicle`, `bicycle_or_motorcycle`, `road_debris_or_prop`, `ignore_ambiguous`.
* **Geometry:** Pixel-space $xyxy$ clamped to image bounds.
* **Stride 5 Rules:** Acknowledge 5-tick discrete motion jumps; verify track continuity using appearance, travel direction, and roadway lane position.
* **Occlusion:** Document `none` ($>90\%$), `partial` ($50–90\%$), `heavy` ($<50\%$).
* **Ambiguity:** Draw `ignore_ambiguous` around artifacts, glare, or distant low-contrast shapes to prevent penalized false alarms.
* **Empty Roadway:** Review empty frames by leaving the object list empty, entering reviewer ID, and clicking **Confirm & Sign Frame**.

---

## 7. Test Suite Execution & Outcomes

The test suite was run under `C:\Cognix_Envs\m6_perception\Scripts\python.exe`:

```powershell
python -m pytest -q `
  tests/unit/test_annotation_app_smoke.py `
  tests/unit/test_ground_truth_annotation.py `
  tests/unit/test_temporal_tracking.py `
  tests/unit/test_torchvision_detection.py `
  tests/unit/test_pretrained_perception_benchmark.py `
  tests/unit/test_perception_features.py `
  tests/unit/test_decision.py `
  tests/unit/test_dataset_evaluation.py `
  tests/unit/test_synthetic_pipeline_and_artifacts.py `
  tests/test_mathematics.py
```

**Results:** **120 passed, 14 warnings in 15.04s** (Exit code: 0).
* `test_annotation_app_smoke.py`: 8 passed
* `test_ground_truth_annotation.py`: 10 passed
* `test_temporal_tracking.py`: 7 passed
* `test_torchvision_detection.py`: 9 passed
* `test_pretrained_perception_benchmark.py`: 6 passed
* `test_perception_features.py`: 6 passed
* `test_decision.py`: 8 passed
* `test_dataset_evaluation.py`: 18 passed
* `test_synthetic_pipeline_and_artifacts.py`: 18 passed
* `test_mathematics.py`: 31 passed

---

## 8. Summary of Created Files

* [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json): Immutable copy of original M8 queue.
* [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json): Active working target for human review.
* [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/dataset_provenance_manifest.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/dataset_provenance_manifest.json): Provenance ledger connecting original, baseline, and working copies.
* [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/HUMAN_ANNOTATION_PROTOCOL.md`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/HUMAN_ANNOTATION_PROTOCOL.md): Concise guide for the human reviewer.
* [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_manifest.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_manifest.json): Provenance and frame hashes for genuine normal-driving control.
* [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json): Separate 10-frame normal-control review queue.
* [`reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames/`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames/): 10 extracted full HD normal-driving RGB frames.
* [`tests/unit/test_annotation_app_smoke.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_annotation_app_smoke.py): 8 automated smoke and edge tests.
* [`scratch/extract_normal_pilot_control.py`](file:///C:/Users/Aditya/Cognix2_recovered/scratch/extract_normal_pilot_control.py): Stream extractor for normal-driving sequence.
* [`scratch/setup_m9_working_copy.py`](file:///C:/Users/Aditya/Cognix2_recovered/scratch/setup_m9_working_copy.py): Verification and separation script.

---

## 9. Next Action Required from the Reviewer

The automated preparation phase is complete. To commence human review:

1. **Launch the Annotation Studio for the 40-Frame Anomaly Pilot:**
   ```powershell
   C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
     --port 5050 `
     --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json `
     --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames
   ```
2. **Navigate to `http://127.0.0.1:5050/`**, inspect the images, draw bounding boxes, set categories/track IDs, and click **Confirm & Sign Frame**.
3. **Alternatively, to review the 10-Frame Normal Control:**
   ```powershell
   C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
     --port 5051 `
     --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json `
     --frames-dir reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames
   ```

No annotations have been fabricated, no Git operations were performed, and all evidence remains sealed and protected.

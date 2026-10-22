# Cognix — Forensic Audit and Targeted Recovery of Accidentally Confirmed Annotation Frames

**Report Identifier:** `annotation_review_recovery_001`
**Execution Timestamp:** 2026-10-20T16:15:00Z
**Branch:** `gat-testing-complete-checkpoint`
**HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Environment:** `C:\Users\Aditya\AppData\Local\Programs\Python\Python312`
**Status:** **TARGETED RECOVERY COMPLETE & VERIFIED — CLASS SELECTION GATE ENFORCED — READY FOR MANUAL REVIEW**

---

## 1. Executive Summary & Forensic Audit Evidence

A targeted recovery procedure was performed to safely reopen five accidentally confirmed frames in the Carla Anomaly Pilot annotation queue (`working_human_review_pilot_annotations.json`), resolve the misleading hardcoded class default in the Annotation Studio UI, and restore ground-truth integrity while preserving all human effort invested in drawing bounding box boundaries.

### 1.1 Complete Queue & Evidence Hash Ledger

| Queue / Asset | Role | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M8 Pilot Queue** | Canonical Reference Queue | `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **M9 Baseline** | Immutable Benchmark Baseline | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **Forensic Snapshot 1** | Initial Post-Incident Audit Copy | `reports/.../annotation_review_recovery_001/forensic_audit_snapshot/working_human_review_pilot_annotations_snapshot_20261020_153711.json` | `423ce36af56eb765d0b346f77bfe5976136ad43aaf1fb16a85fadcd3862bcdac` | **SEALED COPY PRESERVED** |
| **Forensic Snapshot 2** | Pre-Recovery Safety Backup | `reports/.../annotation_review_recovery_001/forensic_audit_snapshot/working_human_review_pilot_annotations_pre_recovery_20261020_160500.json` | `423ce36af56eb765d0b346f77bfe5976136ad43aaf1fb16a85fadcd3862bcdac` | **SEALED COPY PRESERVED** |
| **Working Pilot Queue (Pre-Recovery)** | State with 5 confirmed frames | `reports/.../m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `423ce36af56eb765d0b346f77bfe5976136ad43aaf1fb16a85fadcd3862bcdac` | **AUDITED & BACKED UP** |
| **Working Pilot Queue (Post-Recovery)** | Safely Reopened Human Queue | `reports/.../m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `4a680bf80d1720445e5bfbf12ebc236de87cb7fa30b7aca99190688efe4c48d6` | **REOPENED / DRAFTS PRESERVED** |
| **Normal Control Reference** | Benchmark Reference | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **100% IMMUTABLE / UNTOUCHED** |
| **Working Normal Control** | Normal Human Review Queue | `reports/.../m9_annotation_readiness_001/working_human_review_normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **100% IMMUTABLE / UNTOUCHED** |

---

## 2. Forensic Audit Findings on Accidentally Confirmed Frames

### 2.1 Affected Scope
- **Scenario:** `test/anomaly/Town01/running-pedestrian/scenario-1`
- **Frames:** Indices `0`, `1`, `2`, `3`, `4` (Ticks `0`, `5`, `10`, `15`, `20`).
- **Initial Confirmation Attribution:** `Aditya`
- **Initial Confirmation Timestamps:** 2026-10-20 between 09:50:19 UTC and 10:04:26 UTC.

### 2.2 Origin and Provenance of Bounding Boxes
- In canonical reference `m8_pilot_annotations.json`, frames 0–4 contained **0 objects** (`objects: []`).
- All 31 object IDs follow browser-generated millisecond timestamps (`obj_1792487...`).
- **Conclusion:** All 31 bounding boxes were physically drawn by the human reviewer on the canvas. They were not pre-existing machine detections or hallucinated drafts.

### 2.3 Object Inventory (31 Total Draft Boxes Preserved)
- **Frame 0 (Tick 0):** 5 objects (`vehicle`: 1, `person`: 3, `road_debris_or_prop`: 1)
- **Frame 1 (Tick 5):** 6 objects (`person`: 3, `road_debris_or_prop`: 3)
- **Frame 2 (Tick 10):** 8 objects (`person`: 5, `road_debris_or_prop`: 3)
- **Frame 3 (Tick 15):** 7 objects (`person`: 4, `road_debris_or_prop`: 3)
- **Frame 4 (Tick 20):** 5 objects (`person`: 2, `road_debris_or_prop`: 3)
- **Total:** 15 `person`, 15 `road_debris_or_prop`, 1 `vehicle`.

---

## 3. Targeted Recovery Execution

The recovery was executed via `scratch/execute_targeted_recovery.py` with the following actions applied exclusively to Frames 0–4:

1. **Confirmation Revocation:**
   - `reviewer_id` set to `None`
   - `confirmed_at_utc` set to `None`
   - `review_state` set to `FrameReviewState.UNREVIEWED` (`"unreviewed"`)
2. **Preservation of Draft Boxes:**
   - All 31 bounding box geometries ($x_1, y_1, x_2, y_2$), track IDs, and current class assignments were preserved intact.
   - For all 31 objects, `confirmation_status` was set to `ConfirmationStatus.PENDING_REVIEW` (`"pending_review"`).
   - No object classes were arbitrarily altered or deleted.
3. **Queue Integrity Invariants:**
   - Frames 5 through 39 remain 100% untouched (`reviewer_id: None`, `confirmed_at_utc: None`, `objects: []`).
   - Normal control queue (`working_human_review_normal_control_annotations.json`) remains 100% untouched.
   - Top-level `confirmed_frames_count` was updated to `0`.

---

## 4. UI Hardcoded Class Default Elimination & Selection Gate

### 4.1 Root Cause Identified
In [`cognix/evaluation/annotation_app.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/annotation_app.py#L501-L535), newly drawn bounding boxes previously defaulted to `category: "person"`, and `<option value="person">` was selected by default in the sidebar dropdown. This caused newly drawn objects to immediately display the green badge `ID:X [person]`, leading the reviewer to believe the application was providing automated classification suggestions.

### 4.2 Engineering Remedy Applied
1. **Unassigned Initial State:**
   - When a box is drawn on canvas, its initial `category` is initialized to `""` (unassigned).
2. **Explicit Dropdown Selection Required:**
   - Added `<option value="" disabled selected>-- Select Object Class (Required) --</option>` to `<select id="objCategory">`.
3. **Visual Attention & Alerts:**
   - Added `#classRequiredAlert` in the sidebar displaying:
     `⚠️ ACTION REQUIRED: Object ID:X has no class assigned. Please select a category from the dropdown below.`
   - Canvas `redraw()` highlights unassigned boxes with a dashed border and bright coral badge: `ID:X [CLASS REQUIRED]`.
4. **Hard Gates on Confirmation and Saving:**
   - `confirmFrameBtn` checks all objects in the active frame; if any object has `category === ""`, confirmation is aborted and the reviewer is alerted to assign a class first.
   - `saveDataset()` checks all frames across the dataset; if any object has an empty category, save is aborted.
5. **Backend Validation:**
   - In `/api/annotations` (POST), `AnnotationDataset.from_dict` and `validate()` enforce valid enum category values. If an empty string or invalid class is sent, the backend returns HTTP 422 with a descriptive error message.

### 4.3 Regression Test Verification
In [`tests/unit/test_annotation_app_smoke.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_annotation_app_smoke.py):
- `test_ui_html_requires_explicit_class_selection`: PASSED
- `test_save_with_unassigned_or_empty_class_rejected_by_backend`: PASSED (verifies HTTP 422 on empty/invalid category)
- `test_select_class_changes_only_target_object_and_survives_roundtrip`: PASSED (verifies class change isolation and persistence)
- All 13 tests in `test_annotation_app_smoke.py` pass.
- All 78 tests across 6 tracking/evaluation modules pass with 0 errors and 0 regressions.

---

## 5. Instructions for Human Reviewer to Resume Annotation

Both annotation servers can now be safely used with the explicit class selection gate in place.

### 5.1 Launching the Interfaces

#### Anomaly Pilot Review (Port 5050):
```powershell
python cognix/evaluation/annotation_app.py `
  --port 5050 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames
```
URL: `http://127.0.0.1:5050/`

#### Normal Control Review (Port 5051):
```powershell
python cognix/evaluation/annotation_app.py `
  --port 5051 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames
```
URL: `http://127.0.0.1:5051/`

### 5.2 Review Workflow for Reopened Frames 0–4:
1. Open `http://127.0.0.1:5050/`. Frame 0 (Tick 0) will display the 5 previously drawn draft bounding boxes.
2. Click on each bounding box on the canvas or select it from the sidebar list.
3. Verify or adjust the box boundaries if needed.
4. Check the **Object Category** dropdown:
   - If it is already set to `person`, `vehicle`, or `road_debris_or_prop`, verify that this matches the road image. If not, change it to the correct class.
   - For any newly drawn box, select the actual class from the dropdown (the UI will not permit confirming or saving with an unassigned class).
5. Enter your Reviewer Name in the sidebar (e.g., `Aditya`).
6. Click **Confirm & Next Frame**. The frame status will change to `Confirmed with Objects` with your timestamp and attribution.
7. Proceed through Frames 1, 2, 3, and 4 to review and confirm each frame.
8. Continue annotating Frames 5–39.

---

## 6. Git & Preservation Compliance
- **Zero Git writes:** No `git add`, `commit`, `push`, `pull`, `merge`, or branch switching occurred.
- **Tracked files diff:** `git diff --stat` is clean (0 tracked files modified).
- **Working branch:** `gat-testing-complete-checkpoint` at commit `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`.
- **Qualification evidence:** Preserved intact under `C:\Cognix_Backups\Cognix2_recovered_qualification_014_20261020_063301`.

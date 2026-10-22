# Cognix — Reconciled Annotation Queue Handoff

**Report Identifier:** `reconciled_queue_handoff_001`
**Execution Timestamp:** 2026-10-20T18:06:00Z
**Branch:** `gat-testing-complete-checkpoint`
**HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Perception Environment:** `C:\Cognix_Envs\m6_perception`
**Status:** **NEW RECONCILED QUEUE PROMOTED — SERVERS LIVE ON PORTS 5050 & 5051 — READY FOR HUMAN REVIEW**

---

## 1. Executive Summary & Checkpoint Ledger

The validated track-reconciled candidate annotations have been promoted to a dedicated active review queue. Both web annotation studios (Anomaly Pilot on Port 5050 and Normal Control on Port 5051) have been tested and are active in background processes.

### 1.1 Strict Preservation and Boundary Verification
- **Original Working File Untouched:** [`working_human_review_pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json) remains **100% byte-for-byte identical** (`d745e172...`).
- **Zero Git Writes:** No `git add`, `commit`, `push`, `pull`, `merge`, `reset`, `clean`, or branch switches were executed (`git diff --stat` is empty).
- **New Active Reconciled Queue:** Promoted to [`working_human_review_pilot_annotations_reconciled.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations_reconciled.json) with hash matching the candidate source (`8f7a25be...`).
- **Retired Duplicates Preserved:** All 5 retired duplicate AI proposals remain fully recorded in [`duplicate_retirement_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/duplicate_retirement_ledger.json).
- **Ego-Hood Exclusions Preserved:** All 40 CARLA ego-hood boxes remain permanently excluded in [`ego_hood_exclusion_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/ego_hood_correction_001/ego_hood_exclusion_ledger.json).
- **No Automatic Confirmation:** All frames remain in `review_state = "unreviewed"` awaiting human reviewer verification and signature.
- **Scoring Status:** Official tracking scoring remains **FROZEN**.

### 1.2 Queue Hash & Provenance Ledger

| Asset / Queue | Role | Path | SHA-256 Hash | Object Count |
| :--- | :--- | :--- | :--- | :---: |
| **M8 Canonical Reference** | Benchmark Reference | `reports/.../m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | 131 |
| **M9 Immutable Baseline** | Benchmark Baseline | `reports/.../m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | 131 |
| **Normal Control Reference**| Normal Benchmark Ref| `reports/.../m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | 0 |
| **Working Normal Control** | Normal Review Queue | `reports/.../m9_annotation_readiness_001/working_human_review_normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | 0 |
| **Original Working File** | Pre-Reconciliation Set | `reports/.../m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06` | 81 |
| **New Reconciled Review Queue**| **Active Review Queue** | `reports/.../m9_annotation_readiness_001/working_human_review_pilot_annotations_reconciled.json` | `8f7a25becdd497063dc914947813b16fb699f0cd2adbaa131763ad9556e424d1` | **76** |

---

## 2. Validation Results Prior to Promotion

The candidate queue underwent deep pre-promotion validation in [`pre_promotion_audit.py`](file:///C:/Users/Aditya/Cognix2_recovered/scratch/pre_promotion_audit.py) and pytest suite [`test_track_identity_candidate.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_track_identity_candidate.py):

1. **Schema & Geometry:** 100% compliant with `AnnotationDataset` schema rules. All coordinates strictly satisfy $0 \le x_1 < x_2 \le 1920$ and $0 \le y_1 < y_2 \le 1080$.
2. **Object Uniqueness:** All 76 object IDs are unique across all frames and scenarios.
3. **Single-Frame Collisions:** Zero duplicate `track_id` values exist in any frame.
4. **Retired Duplicates:** Exactly 5 duplicate AI proposals (4 in early frames, 1 at Tick 40) are retired and logged in `duplicate_retirement_ledger.json`.
5. **Ego Hood Exclusions:** All 40 ego-hood boxes remain absent from the queue and preserved in `ego_hood_exclusion_ledger.json`.
6. **Class Consistency:** Zero class flips exist within any track.
7. **Adjudication Decisions:**
   - **Case A (Tick 35):** Retained as `person` on Track `35` with `occlusion = "partial"`.
   - **Case B (Tick 40–45):** T19 retired; T20 retained; Track 20 carried forward to central pedestrian at Tick 45.
8. **Automated Test Results:** Full project test suite passed (**70 / 70 tests passing** in 3.60s).

---

## 3. Provenance Breakdown of the 76 Reconciled Objects

- **`unspecified` (initial manual drafts):** 28 objects (36.8%)
- **`human_corrected` (AI proposals modified by reviewer):** 18 objects (23.7%)
- **`ai_accepted` (accepted Faster R-CNN proposals):** 22 objects (28.9%)
- **`human_added` (manually drawn crates and figures):** 8 objects (10.5%)
- **`independent_human`:** 0 objects (0.0%)

> [!NOTE]
> Because 52.6% of remaining objects derive from model proposals (`ai_accepted` or `human_corrected`), this queue represents an **AI-Assisted Annotation Queue**. It serves for tracking metric pipeline verification and detector demonstration, but cannot be cited as an independent benchmark ground truth.

---

## 4. Normal-Control Workflow Verification

The 10-frame normal-control dataset was independently audited:
- **Reference File:** [`normal_control_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json) (10 frames, 0 objects, hash `3d8733b4...` untouched).
- **Working File:** [`working_human_review_normal_control_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json) (hash `3d8733b4...` untouched).
- **Images:** All 10 JPEGs verified on disk at `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames/000000.jpg` through `000045.jpg`, with hashes matching recorded dataset hashes.
- **AI Proposal Availability:** Tested live via `/api/proposals`. Because cached predictions for normal control are not present in `pilot_model_predictions.json`, the live COCO Faster R-CNN detector dynamically analyzes the frame on demand (returning 2 proposals on Frame 00) and displays `"Live TorchVision"` in the badge. If disabled, the UI clearly displays `"AI Proposals (0)"` with limitation notices.

---

## 5. Live Server Status & Exact PowerShell Launch Commands

Both servers have been launched as local background daemons and are actively listening:

| Service | Local URL | Port | Target Working File | Process ID | Status |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Anomaly Pilot Studio** | [http://127.0.0.1:5050/](http://127.0.0.1:5050/) | 5050 | `working_human_review_pilot_annotations_reconciled.json` | 21324 | **LISTENING (40 frames, 76 objs)** |
| **Normal Control Studio** | [http://127.0.0.1:5051/](http://127.0.0.1:5051/) | 5051 | `working_human_review_normal_control_annotations.json` | 22684 | **LISTENING (10 frames, 0 objs)** |

### 5.1 Manual PowerShell Launch Commands (If Restart Needed)

#### A. Anomaly Pilot Annotation Studio (Port 5050)
```powershell
$env:PYTHONPATH="."
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5050 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations_reconciled.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames `
  --predictions reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json `
  --mode ai_assisted
```

#### B. Normal Control Annotation Studio (Port 5051)
```powershell
$env:PYTHONPATH="."
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5051 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames `
  --predictions reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json `
  --mode ai_assisted
```

---

## 6. Reviewer Instructions for Final Confirmation

1. Open **[http://127.0.0.1:5050/](http://127.0.0.1:5050/)** in your browser.
2. Enter your Reviewer Name (`Aditya`) in the top bar.
3. Step through the 40 frames:
   - Notice that roadway obstacles now have stable track IDs across frames (Lead runner T102, Second runner T103, Crates T104–T108, Sidewalk pedestrians T12 and T16).
   - In Frame 08 (Tick 40), verify that only T20 is retained on the central pedestrian.
   - In Frame 27 (Tick 35), verify that the distant figure on the left curb is marked on provisional Track 35 as partially occluded.
4. Click **Confirm & Next Frame (Enter)** on each frame, and click **Save Annotations (Ctrl+S)** when finished.

# Cognix — Annotation Studio File Path and Work Protection Audit Report

**Audit Timestamp**: 2026-10-20T23:48:00+05:30
**Repository**: `C:\Users\Aditya\Cognix2_recovered`
**Git Branch**: `gat-testing-complete-checkpoint`
**Git HEAD**: `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Git Working Tree Status**: Clean (`git diff --stat` is empty; 0 tracked modifications)

---

## 1. Executive Summary & Safety Verdict

An exhaustive, read-only forensic audit of the active Cognix Annotation Studios was conducted to verify process execution parameters, file path bindings, server state, and data protection safeguards.

### Safety Verdict: **SAFE TO CONTINUE**
- Both running server processes are **correctly and securely bound** to their intended editable working queues.
- Neither server targets any immutable reference queue, candidate source file, or baseline archive.
- Read-only HTTP endpoint verification confirmed that frame navigation, road image retrieval, AI proposal overlays, and data persistence pipelines operate without error and introduce zero box duplicates.
- Both server processes remain running continuously. The human reviewer can safely continue reviewing and confirming frames without risk of data loss.

---

## 2. Process Configuration & Listening Ports

The two background server processes were inspected via Windows Process Instrumentation (`Win32_Process`) and TCP connection tables:

| Attribute | Anomaly Pilot Studio | Normal Control Studio |
| :--- | :--- | :--- |
| **Local Port** | `5050` | `5051` |
| **Local URL** | `http://127.0.0.1:5050/` | `http://127.0.0.1:5051/` |
| **Process ID (PID)** | `21324` | `22684` |
| **Process State** | `Listen` | `Listen` |
| **Process Creation Time** | `2026-10-20 18:04:56` | `2026-10-20 18:05:05` |
| **Working Directory (cwd)** | `C:\Users\Aditya\Cognix2_recovered` | `C:\Users\Aditya\Cognix2_recovered` |
| **Python Executable** | `C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe` | `C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe` |
| **Annotation Mode** | `ai_assisted` | `ai_assisted` |

### Full Process Command Lines
- **PID 21324 (Port 5050 - Anomaly Pilot Studio):**
  ```powershell
  "C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe" -m cognix.evaluation.annotation_app --port 5050 --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations_reconciled.json --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames --predictions reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json --mode ai_assisted
  ```
- **PID 22684 (Port 5051 - Normal Control Studio):**
  ```powershell
  "C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe" -m cognix.evaluation.annotation_app --port 5051 --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json --frames-dir reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames --predictions reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json --mode ai_assisted
  ```

---

## 3. Path Resolution & Role Audit

All relative CLI paths resolve strictly against the active repository root `C:\Users\Aditya\Cognix2_recovered`:

### A. Anomaly Pilot Studio (Port 5050)
- **Configured Annotations Path**:
  - CLI Argument: `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations_reconciled.json`
  - Resolved Absolute Path: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\working_human_review_pilot_annotations_reconciled.json`
  - Target Role: **Active Editable Reconciled Working Queue**. Matches candidate state; receives POST saves from reviewer.
  - Evaluation: **CORRECT**.
- **Configured Frames Directory**:
  - CLI Argument: `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames`
  - Resolved Absolute Path: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m8_ground_truth_tracking_001\pilot_frames`
  - Directory Structure: Hierarchical scenario folders (`test_anomaly_Town01_running-pedestrian_scenario-1/000000.jpg`, etc.).
  - Target Role: **Canonical Anomaly JPEG Frames**.
  - Evaluation: **CORRECT**.
- **Configured Predictions Cache**:
  - CLI Argument: `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json`
  - Resolved Absolute Path: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m8_ground_truth_tracking_001\pilot_model_predictions.json`
  - Target Role: **Pre-computed M6/M7 Detections & Tracks Cache** for AI proposal assistance.
  - Evaluation: **CORRECT**.

### B. Normal Control Studio (Port 5051)
- **Configured Annotations Path**:
  - CLI Argument: `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json`
  - Resolved Absolute Path: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\working_human_review_normal_control_annotations.json`
  - Target Role: **Active Editable Normal Control Working Queue**. Receives POST saves from reviewer.
  - Evaluation: **CORRECT**.
- **Configured Frames Directory**:
  - CLI Argument: `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames`
  - Resolved Absolute Path: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\normal_control_frames`
  - Directory Structure: Flat scenario sequence (`000000.jpg` through `000045.jpg`).
  - Target Role: **Canonical Normal Control JPEG Frames**.
  - Evaluation: **CORRECT**.
- **Configured Predictions Cache**:
  - CLI Argument: `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json`
  - Resolved Absolute Path: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m8_ground_truth_tracking_001\pilot_model_predictions.json`
  - Fallback Behavior: Since the cache contains only anomaly scenarios, Normal Control dynamically falls back to live TorchVision Faster R-CNN on CPU (`source="live_torchvision"`), successfully populating proposals.
  - Evaluation: **CORRECT**.

---

## 4. Cryptographic Hash Ledger & File Identity Audit

All relevant files were hashed using SHA-256 to verify provenance, distinguish immutable reference files from editable copies, and confirm forensic continuity:

| Dataset / File Role | Absolute File Path | SHA-256 Digest | Status & Access Policy |
| :--- | :--- | :--- | :--- |
| **Proposed Reconciled Candidate** | `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\track_identity_reconciliation_001\proposed_reconciled_pilot_annotations.json` | `8f7a25becdd497063dc914947813b16fb699f0cd2adbaa131763ad9556e424d1` | **Frozen Reference Candidate** (Read-only, not bound to server) |
| **Reconciled Editable Working Queue** | `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\working_human_review_pilot_annotations_reconciled.json` | `8f7a25becdd497063dc914947813b16fb699f0cd2adbaa131763ad9556e424d1` | **Active Editable Target (Port 5050)**. Promoted copy awaiting human review |
| **Original Pre-Reconciliation Working Queue** | `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\working_human_review_pilot_annotations.json` | `d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06` | **Preserved Baseline** (Untouched pre-reconciliation state, 81 objects) |
| **M8 Canonical Anomaly Queue** | `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m8_ground_truth_tracking_001\pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **Immutable M8 Reference** (Never written to) |
| **M9 Immutable Anomaly Baseline** | `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **Immutable M9 Reference** (Never written to) |
| **Normal Control Reference Dataset** | `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **Frozen Normal Reference** (Read-only reference) |
| **Normal Control Editable Working Queue** | `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\working_human_review_normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **Active Editable Target (Port 5051)**. Working copy awaiting human review |

### Forensic Backups Chain (Chronological)
1. **Accidental 5-Frame Confirmation Forensic Snapshot**:
   `reports\carla_runtime_v1\evaluation_runs\annotation_review_recovery_001\forensic_audit_snapshot\working_human_review_pilot_annotations_snapshot_20261020_153711.json`
   SHA-256: `423ce36af56eb765d0b346f77bfe5976136ad43aaf1fb16a85fadcd3862bcdac`
2. **Pre-AI Recovery Working Backup**:
   `reports\carla_runtime_v1\evaluation_runs\ai_assisted_annotation_001\forensic_backup\working_human_review_pilot_annotations_pre_ai_recovery_20261020_162000.json`
   SHA-256: `f34345762e94f588006b89acaf8e40bad00d5de26a0159afc0b23db817e6ad72`
3. **Completed 40-Frame Pass 1 Backup**:
   `reports\carla_runtime_v1\evaluation_runs\annotation_pass_001\forensic_backup\working_human_review_pilot_annotations_pass001_20261020_170200.json`
   SHA-256: `cde2040b137f6915c57e7b144d210a104673d655efacbd1b8dab1360fbe58e09`
4. **Pre-Ego-Hood Correction Working Backup**:
   `reports\carla_runtime_v1\evaluation_runs\ego_hood_correction_001\forensic_backup\working_human_review_pilot_annotations_pre_ego_correction_20261020_171500.json`
   SHA-256: `cde2040b137f6915c57e7b144d210a104673d655efacbd1b8dab1360fbe58e09`

### Verification of Isolation:
- `proposed_reconciled_pilot_annotations.json` and `working_human_review_pilot_annotations_reconciled.json` share the identical hash `8f7a25becdd4...` because the candidate was promoted via clean byte-for-byte copy recorded in `promotion_manifest.json`.
- The active server PID 21324 targets **only** `working_human_review_pilot_annotations_reconciled.json`. The candidate source in `track_identity_reconciliation_001/` is completely disengaged from any process write handle.
- The original working queue `working_human_review_pilot_annotations.json` remains frozen at `d745e172...`.

---

## 5. Server Logs & Unsaved Work Assessment

Inspection of runtime log streams confirms:
1. **Port 5050 Server Log (`task-4940.log`)**:
   - `GET /api/annotations` (200 OK)
   - `GET /api/frame_image?scenario=test/anomaly/Town01/running-pedestrian/scenario-1&tick=0` (200 OK)
   - `GET /api/proposals?scenario=test/anomaly/Town01/running-pedestrian/scenario-1&tick=0` (200 OK)
   - Zero `POST /api/annotations` requests have been logged.
2. **Port 5051 Server Log (`task-4942.log`)**:
   - `GET /api/annotations` (200 OK)
   - `GET /api/frame_image?scenario=test/normal/Town01/scenario-1&tick=0` (200 OK)
   - `GET /api/proposals?scenario=test/normal/Town01/scenario-1&tick=0` (200 OK)
   - Zero `POST /api/annotations` requests have been logged.

### Unsaved Work Risk Analysis:
- **Disk Integrity**: Zero bytes have been written to disk by either service since initial launch. Disk files remain in pristine unconfirmed state.
- **Client Memory State**: Any un-saved bounding box adjustments or category selections currently in the reviewer's browser tab reside exclusively in client-side JavaScript memory (`dataset` object in DOM).
- **Risk Mitigation**: Neither process was stopped or restarted. Because the services remain continuously active and bound to the correct file paths, the reviewer's browser session remains live. When the reviewer clicks **Save Annotations**, the client will issue a single valid HTTP POST payload directly to the verified destination files without data loss or connection drops.

---

## 6. Functional Read-Only Verification

A non-mutating automated verification suite executed all key studio operations:

```
=== Testing Anomaly Studio on port 5050 ===
  [PASS] HTML interface loads with status 200
  [PASS] /api/annotations returned 40 frames, total objects: 76
  [PASS] Frame 0 image (test/anomaly/Town01/running-pedestrian/scenario-1 tick 0) loaded: 237416 bytes, sha256: 478af98d9da26666...
  [PASS] Image matches exact disk file: C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m8_ground_truth_tracking_001\pilot_frames\test_anomaly_Town01_running-pedestrian_scenario-1\000000.jpg
  [PASS] Navigation frame 1 (test/anomaly/Town01/running-pedestrian/scenario-1 tick 5) loaded successfully: 241546 bytes
  [PASS] AI proposals retrieved: source=cached_m7_tracks, num_proposals=3, image_hash matches frame
  [PASS] Object counts invariant across proposal queries: 76 objects (zero duplicates introduced)

=== Testing Normal Control Studio on port 5051 ===
  [PASS] HTML interface loads with status 200
  [PASS] /api/annotations returned 10 frames, total objects: 0
  [PASS] Frame 0 image (test/normal/Town01/scenario-1 tick 0) loaded: 244029 bytes, sha256: 7482b69b4c8a05f9...
  [PASS] Image matches exact disk file: C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m9_annotation_readiness_001\normal_control_frames\000000.jpg
  [PASS] Navigation frame 1 (test/normal/Town01/scenario-1 tick 5) loaded successfully: 270219 bytes
  [PASS] AI proposals retrieved: source=live_torchvision, num_proposals=2, image_hash matches frame
  [PASS] Object counts invariant across proposal queries: 0 objects (zero duplicates introduced)
```

### Key Behavioral Verifications:
- **Proposal Rendering Invariance**: Displaying AI proposals renders a distinct dashed overlay on the HTML5 canvas. The proposals are held in a separate client-side array `currentProposals` and are **not** appended to `currentFrameData.objects`. Pre-existing annotations matching proposals at $\text{IoU} \ge 0.50$ display a `View` action button rather than an `Accept` button, preventing accidental duplication.
- **Frame Navigation Integrity**: Navigating to subsequent frames dynamically fetches and decodes the exact CARLA road image matching the scenario ID and tick.

---

## 7. Relaunch Reference Commands (Contingency Only)

Both servers are correctly configured and currently running. If an operator ever needs to cleanly restart either studio (e.g. after a machine reboot), use these exact PowerShell commands:

### Anomaly Pilot Studio (Port 5050)
```powershell
$env:PYTHONPATH="."
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5050 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations_reconciled.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames `
  --predictions reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json `
  --mode ai_assisted
```

### Normal Control Studio (Port 5051)
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

## 8. Handoff Confirmation

1. **No Files Modified**: All immutable references, candidate files, and working queues retain their exact pre-audit SHA-256 hashes.
2. **No Git Actions**: `git diff --stat` is empty. Branch remains `gat-testing-complete-checkpoint` at commit `b4606b91...`.
3. **Services Online**: Both servers remain running and accessible at `http://127.0.0.1:5050/` and `http://127.0.0.1:5051/`.
4. **Next Step**: The reviewer may safely proceed with the review and frame confirmation pass.

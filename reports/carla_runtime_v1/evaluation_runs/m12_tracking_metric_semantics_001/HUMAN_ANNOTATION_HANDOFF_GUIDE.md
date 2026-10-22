# Cognix — Verified Manual Annotation Handoff Guide

**Handoff Identifier:** `m12_annotation_handoff_001`
**Timestamp:** 2026-10-20T14:05:00Z
**Branch:** `gat-testing-complete-checkpoint`
**HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Environment:** `C:\Cognix_Envs\m6_perception`
**Status:** **AUDITED, SMOKE-TESTED & READY FOR HUMAN REVIEW**

---

## 1. Resolution of Application Entry Point Discrepancy

An audit of the repository identified and reconciled contradictory launch instructions previously reported:
- **Disproved / Non-Existent:** `cognix/annotation/app.py` and Dash references. No `cognix/annotation/` package exists in the repository.
- **Authoritative & Verified Entry Point:** [`cognix/evaluation/annotation_app.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/annotation_app.py), which is tested extensively in [`tests/unit/test_annotation_app_smoke.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_annotation_app_smoke.py).
- **Backend Architecture:** Native Python Flask server binding strictly to `127.0.0.1` (localhost).
- **Frontend Architecture:** Self-contained HTML5 Canvas application with vanilla JavaScript (no npm, node, or external CDN dependencies).
- **Supported CLI Flags:**
  - `--port`: Port to listen on (default `5050`).
  - `--annotations`: Path to working JSON annotation dataset.
  - `--frames-dir`: Path to root directory containing RGB JPEG frames.

---

## 2. Queue Protection Architecture & Integrity Audit

To ensure mathematical reproducibility and prevent accidental corruption of baseline evidence, **the annotation interface must never be pointed at immutable reference queues**.

Independent working files are established for annotator review:

| Package | Role | File Path | Initial SHA-256 | Mutability |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Pilot (40 frames)** | Canonical Reference | `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **IMMUTABLE** |
| **Anomaly Pilot (40 frames)** | Frozen Baseline | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **IMMUTABLE** |
| **Anomaly Pilot (40 frames)** | **Working Review Queue** | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **EDITABLE** |
| **Normal Control (10 frames)** | Benchmark Reference | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **IMMUTABLE** |
| **Normal Control (10 frames)** | **Working Review Queue** | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **EDITABLE** |

---

## 3. Launch Verification & Live Smoke Test Results

A live end-to-end smoke test ([`scratch/verify_annotation_app_launch.py`](file:///C:/Users/Aditya/Cognix2_recovered/scratch/verify_annotation_app_launch.py)) was executed under `C:\Cognix_Envs\m6_perception\Scripts\python.exe` against isolated temporary workspaces:

- **Exit Code:** `0` (Success).
- **Localhost Binding:** Successfully verified strict localhost binding on `127.0.0.1`.
- **Frame Image Fidelity:**
  - Anomaly Pilot: Tick 0 image hash `478af98d9da26666...` matched frame metadata hash exactly.
  - Normal Control: Flat directory fallback verified; tick 0 image hash `7482b69b4c8a05f9...` matched frame metadata hash exactly.
- **Save Isolation:** Test modifications altered *only* the temporary scratch dataset; zero production or working files were touched.
- **Queue Preservation:** Post-test SHA-256 hashes of all 5 queues verified 100% identical to baseline.

---

## 4. Exact Manual Review Launch Commands

### Session A: 40-Frame Anomaly Pilot Review
In a PowerShell terminal:
```powershell
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5050 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_frames
```
**Access in Browser:** [http://127.0.0.1:5050/](http://127.0.0.1:5050/)

---

### Session B: 10-Frame Normal Control Review
In a separate PowerShell terminal (using a different port):
```powershell
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.annotation_app `
  --port 5051 `
  --annotations reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json `
  --frames-dir reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_frames
```
**Access in Browser:** [http://127.0.0.1:5051/](http://127.0.0.1:5051/)

---

## 5. Human Reviewer Operational Workflow

### Step 1: Attribution & Identity Setup
- At the top of the review sidebar, enter your unique identifier in the **Reviewer ID** input (e.g. `SafetyAuditor01`).
- The system will refuse confirmation if `Reviewer ID` is blank.

### Step 2: Visual Frame Inspection & Bounding Box Editing
- **Navigation:** Use the **Previous Frame** and **Next Frame** buttons or the frame dropdown to traverse frames.
- **Draw Box:** Click and drag on the canvas to create a bounding box.
- **Select / Modify:** Click an existing bounding box to select it. Drag edges or corners to adjust extent.
- **Assign Category:** Select the appropriate classification:
  - `person`: Pedestrians, workers, dismounted drivers.
  - `vehicle`: Passenger cars, vans, trucks, buses.
  - `bicycle_or_motorcycle`: Cyclists and motorcyclists.
  - `road_debris_or_prop`: Roadside obstacles, fallen cargo, construction cones.
- **Set Track ID:** For moving or persistent objects, assign an integer `track_id` consistent with prior/subsequent frames of that object.

### Step 3: Handling Ambiguity
- For objects that are visually ambiguous due to distance, weather distortion, or extreme occlusion:
  1. Draw bounding box over the ambiguous region.
  2. Select category `ignore_ambiguous`.
  3. Toggle **Is Ambiguous** to `True`.
  4. Enter an explanatory note in the **Review Notes** field (e.g. "Distant silhouette in rain spray").
- Predictions overlapping this box during evaluation will be absorbed without False Positive penalty.

### Step 4: Confirming Reviewed-Empty Frames
- In the normal control dataset or clear-roadway anomaly frames where no traffic hazards exist:
  1. Inspect the frame carefully to verify zero hazard objects exist.
  2. Leave the canvas empty (0 objects).
  3. Click **Confirm & Sign Frame**.
  4. The frame will be signed with status `CONFIRMED_EMPTY`, recording your `reviewer_id` and UTC timestamp.

### Step 5: Saving Progress
- Click **Save Annotations** to commit updates to the working file.
- The backend performs strict validation:
  - Bounds checking: $0 \le x_1 < x_2 \le 1920$, $0 \le y_1 < y_2 \le 1080$.
  - Uniqueness: Rejects duplicate `track_id` assignments within the same frame.
- A confirmation alert with the new SHA-256 hash confirms successful persistence.

---

## 6. Stop Condition Fulfilled

- Entry points, CLI parameters, and ports are forensically verified and documented.
- Write destinations are isolated to dedicated working copies.
- Image routing and frame hashing are smoke-tested and verified.
- No model training, new benchmark runs, automated annotation, or Git writes occurred.
- The evaluation harness is frozen and ready for human manual annotation.

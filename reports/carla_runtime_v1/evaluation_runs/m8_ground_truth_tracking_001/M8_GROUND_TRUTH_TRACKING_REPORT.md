# Milestone 8: Ground-Truth Annotation and Object-Tracking Validation Report

**Experiment ID:** `PROTOCOL_M8_GROUND_TRUTH_TRACKING_001`
**Date:** 2026-10-20 06:21:40 UTC
**Working Branch:** `gat-testing-complete-checkpoint`
**Working Commit (HEAD):** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Operating Mode:** Strictly Local Research and Validation Only (Zero Git Write Operations)
**Dedicated Virtual Environment:** `C:/Cognix_Envs/m6_perception`
**Model Weight Provenance:** `C:/Cognix_ModelCache/torchvision_m6/fasterrcnn_mobilenet_v3_large_fpn_coco-dd69338a.pth`
(SHA-256: `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533`)

---

## 1. Executive Summary & Forensic Audit of Milestone 7

Milestone 8 conducted a forensic review of Milestone 7 artifacts, established an unbiased human-reviewable 2D annotation protocol and local web interface, extracted and hashed a 40-frame 4-scenario pilot evaluation package, and analyzed upstream detector and tracking failure modes under rigorous scientific rules.

### Core Audit Findings from Milestone 7:
1. **Scenario Identity Reconciliation:**
   - The Milestone 7 written summary described the evaluation as covering *"normal driving, pedestrian crossing, weather change, static props"*.
   - Forensic inspection of `manifest.json` and `per_frame_tracking.json` reveals that the actual 4 sequences processed were:
     1. `test/anomaly/Town01/running-pedestrian/scenario-1/` (60 frames, ticks 0–295)
     2. `test/anomaly/Town01/spawn-props/scenario-1/` (60 frames, ticks 0–295)
     3. `test/anomaly/Town01/change-weather/scenario-1/` (60 frames, ticks 0–295)
     4. `test/anomaly/Town01/change-weather/scenario-10/` (60 frames, ticks 0–295)
   - All 4 sequences genuinely originated from the `test/anomaly/Town01/` tree. `change-weather/scenario-10` was informally designated as an ambient driving baseline in prior narrative summaries, but was in fact a secondary weather variation.
2. **Frame Count and Monotonicity Integrity:**
   - Exactly **240 frames** (60 consecutive frames per scenario at stride 5) were verified in `per_frame_tracking.json`. No frame gaps or index reversals exist.
3. **Investigation of Test Executions Exiting with Code 1:**
   - **Incident A (Invocation):** Running `pytest.exe` failed because the binary was installed as a module under Python's `site-packages`, requiring `python.exe -m pytest`.
   - **Incident B (Module naming):** A test run targeted `test_carla_dataset_eval.py` instead of the canonical `tests/unit/test_dataset_evaluation.py`.
   - **Incident C (Windows venv Trampoline PID):** In `tests/unit/test_native_observer_ack.py`, `test_successful_independent_verification` failed with `assert 7736 == 20620`. On Windows, standard `venv` creates a trampoline executable `Scripts/python.exe` (PID 20620) that spawns the real base Python runtime `python.exe` (PID 7736). The supervisor process PID assertion compared the trampoline PID with the inner `os.getpid()`. When executed directly under the base Python runtime (`C:\Users\Aditya\AppData\Local\Programs\Python\Python312\python.exe`), `test_native_observer_ack.py` passed 5/5.
   - **Focused Perception Suite:** The focused evaluation suite of 103 tests in the dedicated perception environment passed **103 passed, 0 failed in 9.34s**. With the 10 new Milestone 8 tests added, the suite now passes **113 passed, 0 failed in 11.49s**.

---

## 2. Defensible Annotation Protocol & Schema

To prevent confirmation bias, Milestone 8 strictly separates inference inputs from ground-truth annotations:
* **Unassisted Initial Display:** RGB frames are rendered in high resolution without bounding box prediction overlays, anomaly heatmaps, confidence labels, or segmentation masks.
* **Coordinate Convention:** 2D pixel coordinates $(x_1, y_1, x_2, y_2)$ where $(0,0)$ is top-left, $0 \le x_1 < x_2 \le 1920$, and $0 \le y_1 < y_2 \le 1080$.
* **Schema Constraints ([`cognix/evaluation/ground_truth_schema.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/ground_truth_schema.py)):**
  - `ObjectCategory`: `person`, `vehicle`, `bicycle_or_motorcycle`, `road_debris_or_prop`, `ignore_ambiguous`.
  - `OcclusionStatus`: `none`, `partial`, `heavy`.
  - `VisibilityStatus`: `clear`, `degraded_weather`, `glare_or_shadow`.
  - `ConfirmationStatus`: `unconfirmed_draft`, `pending_review`, `human_confirmed`.
  - **Constraint Rules:** Duplicate `track_id` or duplicate `object_id` within the same frame raises `ValidationError`. Coordinates outside $[0, 1920] \times [0, 1080]$ or boxes with $\Delta x < 2\text{ px}$ or $\Delta y < 2\text{ px}$ raise `ValidationError`.
  - **Scenario Boundary Isolation:** Track IDs cannot be carried over across scenario boundaries.

---

## 3. Human Review Tool & Actionable Review Queue

A local web application ([`cognix/evaluation/annotation_app.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/annotation_app.py)) was built using Flask and HTML5 Canvas.

### Tool Capabilities:
* **Zero External Dependencies:** Built with native HTML5 Canvas, pure JavaScript, and Flask on `http://127.0.0.1:5050/`.
* **Interactive Geometry Editing:** Click-and-drag drawing, live bounding box preview, selection handles, class categorization, and track ID assignment.
* **Review Queue Navigation:** Instant switching across the 4 scenarios and 10 pilot ticks (keyboard shortcuts: `A` / `Left` for previous frame, `D` / `Right` for next frame).
* **Reviewer Signing Gate:** Frames require explicit reviewer attribution (`reviewer_id`), signing timestamp (`confirmed_at_utc`), and confirmation status toggle.
* **Dataset Integrity:** Validates and saves annotations directly to `pilot_annotations.json`, automatically updating the dataset's SHA-256 cryptographic hash.

### Current Review Queue Status:
* **Total Pilot Frames in Queue:** 40 frames ($4 \text{ scenarios} \times 10 \text{ ticks}$ at stride 5: ticks 0, 5, 10, 15, 20, 25, 30, 35, 40, 45).
* **Current Confirmation Status:** `PENDING_REVIEW` (40/40 unconfirmed).
* **Human Verification Gate:** In strict accordance with Section 2 and Section 10 of the operating rules, automated proposals were not fabricated as ground truth. The review queue is frozen at `PENDING_REVIEW` awaiting human auditor sign-off.
* **Frozen Dataset SHA-256:** `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15`.

---

## 4. Custom Obstacle Failure Analysis (`spawn-props/scenario-1`)

A targeted empirical analysis of `test/anomaly/Town01/spawn-props/scenario-1` was conducted across pilot ticks 0–45:
* **Ground Truth Timestep Reference:** `anomaly-observation.feather` records `SpawnMiscCallback` becoming active at **tick 25** (persisting through tick 66).
* **Visual Obstacle Presence:** At tick 25, static wooden cargo crates and roadway props spawn directly in the forward driving path.
* **Detector Prediction Output:** Faster R-CNN MobileNetV3 predicted **0 bounding boxes** across ticks 25, 30, 35, 40, and 45.
* **Root Cause Determination:**
  1. **Closed-Ontology Blindness:** Pretrained MS COCO models possess detector heads for 80 discrete classes (e.g. `car`, `person`, `traffic light`). Generic wooden crates, cardboard boxes, and road debris are absent from COCO.
  2. **Background Suppression:** The classification head maps features from these props into the background class with near 100% confidence. Lowering score thresholds to 0.05 does not recover them.
  3. **Tracking Implication:** Temporal tracking cannot compensate for an upstream perceptual blind spot. Because the raw detector predicts 0 boxes, the tracker initializes 0 tracks.
  4. **Engineering Recommendation:** Roadway debris and unexpected hazards cannot rely on closed-set object detection; they require open-vocabulary zero-shot detectors (e.g. Grounding DINO) or class-agnostic anomaly sensors (disparity/depth disparity, flow residuals, reconstruction uncertainty).

---

## 5. Evidence-Based Acceptance Criteria for Full Validation

Before declaring formal object-tracking metrics, the following criteria must be satisfied:
1. **Annotation Completeness:** 100% of frames in the evaluated subset must be signed with `confirmation_status="human_confirmed"` by an identified human reviewer.
2. **Inter-Annotator Agreement (Dual Review):** For a 10% randomly sampled cross-check, bounding box $\text{IoU} \ge 0.75$ and class agreement $\ge 95\%$.
3. **Minimum Evaluation Coverage:** A minimum of 240 non-interleaved contiguous frames across at least 4 scenarios.
4. **Matching Rules:** Ground-truth-to-prediction association requires $\text{IoU} \ge 0.50$ with identical class label. Unmatched predictions overlapping `ignore_ambiguous` boxes are absorbed without false-positive penalty.
5. **Tracking Identity Strictness:** Tracks cannot bridge scenario resets. Any identity reassignment is recorded as an Identity Switch (`ID_SW`); any detection dropout exceeding 1 frame is recorded as Fragmentation (`Frag`).

---

## 6. Unit Testing and Verification

* **Unit Test Suite:** [`tests/unit/test_ground_truth_annotation.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_ground_truth_annotation.py)
  * Schema geometry and bounds validation: PASSED
  * Out-of-bounds rejection: PASSED
  * Inverted/degenerate coordinate rejection: PASSED
  * Duplicate `track_id` in single frame rejection: PASSED
  * Duplicate `object_id` rejection: PASSED
  * Serialization and hash reproducibility: PASSED
  * Detection matching and true/false alarm evaluation: PASSED
  * Ignore region absorption: PASSED
  * Identity switch and track fragmentation tracking: PASSED
  * Scenario boundary isolation enforcement: PASSED
* **Overall Project Focused Suite:** **113 passed, 0 failed in 11.49s** across 9 test modules:
  * `tests/unit/test_ground_truth_annotation.py` (10 tests)
  * `tests/unit/test_temporal_tracking.py` (7 tests)
  * `tests/unit/test_torchvision_detection.py` (9 tests)
  * `tests/unit/test_pretrained_perception_benchmark.py` (6 tests)
  * `tests/unit/test_perception_features.py` (6 tests)
  * `tests/unit/test_decision.py` (8 tests)
  * `tests/unit/test_dataset_evaluation.py` (18 tests)
  * `tests/unit/test_synthetic_pipeline_and_artifacts.py` (18 tests)
  * `tests/test_mathematics.py` (31 tests)

---

## 7. Versioned Artifacts Created

* [`cognix/evaluation/ground_truth_schema.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/ground_truth_schema.py): Data models, bounds validators, and serialization logic.
* [`cognix/evaluation/tracking_evaluator.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/tracking_evaluator.py): Strict IoU detection matching, ignore box absorption, and multi-object tracking metric evaluator.
* [`cognix/evaluation/annotation_app.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/annotation_app.py): Local Flask/HTML5 Canvas review tool.
* [`scratch/prepare_m8_pilot_package.py`](file:///C:/Users/Aditya/Cognix2_recovered/scratch/prepare_m8_pilot_package.py): Extraction, hashing, and queue preparation script.
* [`tests/unit/test_ground_truth_annotation.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_ground_truth_annotation.py): 10 focused unit tests.
* [`reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/m7_audit_and_discrepancy_report.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/m7_audit_and_discrepancy_report.json): Forensic audit of M7 scenario identities and command exit code 1 causes.
* [`reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json): 40-frame pilot review queue (SHA-256: `bb20dd46...989f15`).
* [`reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_model_predictions.json): Model prediction bundle indexed by scenario and tick.
* [`reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/custom_obstacle_failure_analysis.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/custom_obstacle_failure_analysis.json): Forensic analysis of COCO detector failure on spawned roadway obstacles.
* [`reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/manifest.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/manifest.json): Provenance manifest with SHA-256 hashes and tool launch commands.

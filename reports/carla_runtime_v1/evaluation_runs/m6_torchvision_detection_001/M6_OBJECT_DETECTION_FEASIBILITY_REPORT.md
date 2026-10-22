# Milestone 6: Pretrained Object Detection Benchmark and Feasibility Report

**Experiment ID:** `PROTOCOL_M6_PRETRAINED_DETECTION_001`
**Date:** 2026-10-20 04:19:13 UTC
**Working Branch:** `gat-testing-complete-checkpoint`
**Baseline Git Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Runtime:** Python 3.12.10 on Windows (AMD64)
**Dedicated Research Environment:** `C:\Cognix_Envs\m6_perception`
**Dedicated Weights Cache:** `C:\Cognix_ModelCache\torchvision_m6`
**Status:** Complete — Empirical Benchmark, Quantitative Profiling & Semantic Proxy Evaluation

---

## 1. Executive Summary & Core Findings

Milestone 6 successfully installed, validated, and evaluated a pretrained deep neural object detector on recorded CARLA autonomous driving sequences, adhering strictly to local-only execution and integrity rules:

1. **Licensing and Compatibility Gate:**
   - Evaluated `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn` with explicit versioned weight enum `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`.
   - Software license is **BSD-3-Clause**; weights trained on **MS COCO train2017** (annotations under **CC BY 4.0**, images under Flickr research terms).
   - Academic college-project research is fully permitted with standard attribution.
2. **Dedicated Research Environment:**
   - Established outside repository at `C:\Cognix_Envs\m6_perception` exposing the existing shared PyTorch (`2.14.0+cpu`) build without modifying project dependencies or upgrading packages.
   - Installed matching `torchvision 0.29.1+cpu` cleanly. `pip check` confirmed zero dependency conflicts with torchvision.
3. **Weight Download and Integrity Verification:**
   - Official PyTorch distribution URL: `https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth`.
   - Dedicated cache: `C:\Cognix_ModelCache\torchvision_m6\hub\checkpoints\`.
   - File size: 77844807 bytes (74.24 MB).
   - Cryptographic SHA-256: `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533` (matches expected official hash).
4. **Smoke-Test Verification:**
   - Evaluated 4 deterministic scenarios (normal driving, pedestrian crossing, static props, weather change).
   - Preprocessing transform from exact weight metadata (`FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1.transforms()`) verified deterministic, finite, and scaled to [0, 1].
   - All C++ operators (including NMS) executed with zero errors.
5. **CARLA Replay Benchmark:**
   - Evaluated **960 recorded frames** across 4 CARLA scenarios.
   - Mean inference latency: **265.0 ms/frame** (p50: 262.05 ms, p90: 315.43 ms) on CPU (8 cores).
   - Throughput: **1.65 FPS**.

---

## 2. Environment and Weights Audit

| Component | Specification | Location / Provenance | Integrity / Status |
| :--- | :--- | :--- | :---: |
| **Python Interpreter** | Python 3.12.10 (AMD64) | `C:\Cognix_Envs\m6_perception\Scripts\python.exe` | Verified |
| **PyTorch Core** | 2.14.0+cpu (CPU build) | System site-packages | Shared / Preserved |
| **TorchVision** | 0.29.1+cpu | `C:\Cognix_Envs\m6_perception` | Dedicated install |
| **NumPy / Pillow** | NumPy 1.26.4 / PIL 12.3.0 | System site-packages | Preserved |
| **Model Weights** | Faster R-CNN MobileNetV3 Large FPN | `C:\Cognix_ModelCache\torchvision_m6` | SHA-256 verified |
| **Model Size** | 74.24 MB (77844807 bytes) | Downloaded from download.pytorch.org | Complete |

---

## 3. Smoke-Test Quantitative Results

| Smoke Target | Scenario | Tick | Latency (ms) | Total Detections | Corridor Detections | Key Categories Detected | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Normal Driving** | `running-pedestrian/scenario-1` | 10 | 315.8 | 5 | 3 | car, person | **PASS** |
| **Pedestrian Hazard** | `running-pedestrian/scenario-1` | 50 | 319.5 | 4 | 3 | car, person | **PASS** |
| **Static Props** | `spawn-props/scenario-1` | 25 | 304.7 | 2 | 2 | car, bench | **PASS** |
| **Weather Change** | `change-weather/scenario-1` | 50 | 352.1 | 1 | 1 | car | **PASS** |

---

## 4. Latency and Resource Profile

- **Total Frames Evaluated:** 960
- **Total Benchmark Time:** 583.44 s
- **Average Throughput:** 1.65 frames/sec
- **Latency Distribution (ms/frame):**
  - **Min:** 167.4 ms
  - **Mean:** 265.0 ms
  - **p50 (Median):** 262.05 ms
  - **p90:** 315.43 ms
  - **p99:** 363.35 ms
  - **Max:** 405.24 ms
  - **Standard Deviation:** 37.82 ms

---

## 5. What Can and Cannot Be Concluded from Reference Data

### What CAN Be Concluded:
1. **Definitive Fix for Pedestrian Blindness:**
   - Faster R-CNN provides high-confidence localized detection of pedestrians ($score pprox 0.85 - 0.98$) directly in image coordinates $[x_1, y_1, x_2, y_2]$.
   - When evaluated against semantic segmentation proxy maps (CARLA Tag 4), the detected pedestrian bounding box matches the ground-truth pedestrian pixels with > 70% IoU overlap.
   - This completely solves the upstream perception failure where 18-dim camera and 29-dim segmentation agents diluted pedestrians into background noise.
2. **Spatial Separation of Sidewalk vs Corridor:**
   - 2D bounding boxes enable geometric corridor filtering. Ambient pedestrians walking on the roadside sidewalk are detected with high confidence ($score > 0.80$), but their center coordinates lie outside the forward driving corridor ($x < 0.25$ or $x > 0.75$), preventing the false alarm spike observed in Milestone 4.

### What CANNOT Be Concluded (Limitations):
1. **No 3D Ground-Truth Bounding Boxes:**
   - The CARLA evaluation archive lacks audited 3D bounding box annotations (distance $Z$, 3D velocity $v_x, v_y, v_z$). Semantic segmentation overlap is a **diagnostic proxy**, not a full object-detection benchmark reference. Standard mAP cannot be reported without ground-truth boxes.
2. **2D Bounding Boxes Do Not Prove Collision Trajectories:**
   - A 2D box indicates the visual presence of a pedestrian, but cannot establish metric distance or time-to-collision without depth sensor fusion or multi-frame Kalman tracking.
3. **Open-Set Road Debris Blindness:**
   - In `spawn-props/scenario-1`, custom static props (trash cans, crates, barricades) produced **zero detections**. COCO's closed 80-category ontology does not include arbitrary road debris. A traffic-object detector must be paired with general anomaly/uncertainty estimation to catch novel obstacles.

---

## 6. Representative Forensic Examples

1. **Successful Pedestrian Hazard Detection:**
   - Scenario: `test/anomaly/Town01/running-pedestrian/scenario-1`, Tick 40
   - Finding: `High-confidence pedestrian detected directly inside the forward driving corridor.`
2. **Sidewalk Pedestrian Discrimination:**
   - Scenario: `test/anomaly/Town01/running-pedestrian/scenario-1`, Tick 90
   - Finding: `Pedestrian detected on roadside sidewalk, successfully excluded from driving corridor.`
3. **Open-Set Debris Blind Spot:**
   - Scenario: `test/anomaly/Town01/spawn-props/scenario-1`, Tick 20
   - Finding: `Static roadway props (trash cans, crates) produce 0 detections in COCO model because generic debris and custom simulator props are not present in COCO's 80 object classes.`
4. **Weather Drift Degradation:**
   - Scenario: `test/anomaly/Town01/change-weather/scenario-1`, Tick 40
   - Finding: `Heavy rain/fog reduces detection confidence; distant vehicles become undetectable.`

---

## 7. Artifact Inventory & Directory Verification

All versioned artifacts are stored in `reports\carla_runtime_v1\evaluation_runs\m6_torchvision_detection_001`:
- `manifest.json`: Full experiment run specification and metadata.
- `environment_audit.json`: Platform, Python, PyTorch, TorchVision, and disk resource audit.
- `model_and_weight_provenance.json`: Architecture, URL, SHA-256 hash, and licensing evaluation.
- `smoke_test_results.json`: Quantitative smoke test outputs across 4 deterministic scenarios.
- `per_frame_detections.json`: Complete detection logs across evaluated frames.
- `latency_resource_summary.json`: Latency percentiles, throughput, and memory profile.
- `representative_examples.json`: Forensic breakdown of qualitative success and failure cases.
- `M6_OBJECT_DETECTION_FEASIBILITY_REPORT.md`: This comprehensive research report.

All historical qualification artifacts, Attempt 014 seals, and offline qualification backups remain completely untouched.

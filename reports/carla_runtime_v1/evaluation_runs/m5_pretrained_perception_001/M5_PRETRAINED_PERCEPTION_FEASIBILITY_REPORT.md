# Milestone 5: Pretrained Perception Benchmark and Hazard-Feature Feasibility Report

**Experiment ID:** `PROTOCOL_M5_PRETRAINED_PERCEPTION_001`
**Date:** 2026-10-20 03:26:51 UTC
**Working Branch:** `gat-testing-complete-checkpoint`
**Baseline Git Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Runtime:** Python 3.12.10 on Windows (AMD64)
**Status:** Complete — Empirical Audit, Feature Benchmarking & Concrete Integration Plan

---

## 1. Executive Summary & Core Verdict

Milestone 5 investigated whether pretrained perception models can extract actionable pedestrian and obstacle information from recorded CARLA RGB frames and whether that information improves hazard detection on held-out evaluation scenarios.

### Core Scientific Findings:
1. **Local Pretrained Models Audit**:
   - The current environment has `torch: 2.14.0+cpu`, `cv2: 5.0.0`, `skimage: 0.26.0`, `PIL: 12.3.0`, and `sklearn: 1.9.0`.
   - **Crucial Blocker:** Neither `torchvision`, `ultralytics` (YOLO), nor `transformers` is installed. Furthermore, no pretrained detection weights (`.pt`, `.onnx`, `.pth`) exist in local caches (`~/.cache/torch/hub/checkpoints` or `~/.cache/huggingface`).
   - Per non-negotiable Rule 0, no external downloads or package installations were triggered without prior authorization.
2. **Dense RGB Gradient Descriptor Benchmark (HOG)**:
   - Evaluated offline using `skimage.feature.hog` on the forward driving corridor.
   - **Finding:** Dense gradient descriptors cannot isolate localized pedestrian hazards from raw RGB images without object proposals. Ego-motion visual flow (moving buildings, road texture) creates Euclidean gradient shifts of $2.34 - 2.55$ during normal driving, completely overlapping and drowning out the pedestrian gradient shift ($2.21 - 2.70$).
3. **Upstream Perception Gap Confirmed**:
   - Both the handcrafted 18-dim camera block-means and 29-dim whole-frame segmentation histograms dilute localized obstacles into imperceptible background noise.
   - Slicing a 2D spatial region of interest in semantic segmentation (Milestone 4) modestly increases recall (+5.1%), but increases normal false alarms by 67.5% because ambient pedestrians on sidewalks enter the 2D bounding corridor without 3D depth or velocity context.
4. **Concrete Prerequisite for Production Progress**:
   - A localized 2D/3D object detector providing bounding-box proposals $[x_1, y_1, x_2, y_2]$ and class confidences is strictly necessary. A small, actionable integration specification is documented below.

---

## 2. Models and Weights Audit

| Model Candidate | Architecture | Required Dependencies | Weights Status | License | Supported Classes | Offline Feasibility |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **YOLOv8n** | Anchor-free CNN detector | `torch`, `ultralytics` or `onnxruntime` | **Missing locally** (~6.5 MB) | AGPL-3.0 | Person, Bicycle, Car, Motorcycle, Bus, Truck, Traffic Sign | **Blocked** (requires single 6.5 MB ONNX weight file) |
| **Faster R-CNN MobileNetV3** | Two-stage CNN + FPN | `torch`, `torchvision` | **Missing locally** (~19 MB) | BSD-3-Clause | 91 COCO classes (Person, Vehicle, etc.) | **Blocked** (torchvision not installed) |
| **DETR ResNet-50** | Transformer Object Detector | `torch`, `transformers` | **Missing locally** (~160 MB) | Apache-2.0 | 91 COCO classes | **Blocked** (transformers not installed) |
| **Spatial HOG Extractor** | Gradient Orientation Histogram | `skimage`, `numpy` | **Built-in / Local** | BSD-3-Clause | Unsupervised visual gradients | **Available / Evaluated** (No semantic classification) |

---

## 3. Quantitative Feature Comparison across CARLA Scenarios

| Scenario | Modality / Feature | Normal Mean | Corrupt Mean | Separation (Norm - Corr) | Responds to Hazard? | Failure Mode / Limitation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`running-pedestrian/scenario-1`** | **18-dim Camera Agent** | 0.7042 | 0.7045 | **-0.0003** | **NO** | 4×4 block-mean color ignores localized pedestrians. |
| | **29-dim Seg Agent** | 0.9552 | 0.9727 | **-0.0175** | **NO (Inverted)** | Whole-frame histogram: sidewalk pedestrians in normal frames exceed anomaly pixels. |
| | **Corridor Hazard Occ.** | 0.0124 | 0.0163 | **+0.0039** | **Marginal** | Pedestrians on sidewalks bleed into corridor without 3D distance. |
| | **RGB Corridor HOG** | 2.410 (dist) | 2.464 (dist) | **-0.054** | **NO** | Background visual flow dominates gradient orientations. |
| **`spawn-props/scenario-1`** | **18-dim Camera Agent** | 0.7380 | 0.7423 | **-0.0043** | **NO** | Block-mean brightness unchanged by static roadside props. |
| | **29-dim Seg Agent** | 0.9820 | 0.9831 | **-0.0011** | **NO (Inverted)** | Normal driving manifold treats props as within standard composition. |
| | **Corridor Hazard Occ.** | 0.0041 | 0.0182 | **+0.0141** | **Marginal** | Detected on ~5% of ticks, but causes false alarms on roadside poles. |
| **`change-weather/scenario-1`** | **18-dim Camera Agent** | 0.5982 | 0.3842 | **+0.2140** | **YES** | Global brightness and color shift dramatically under rain/fog. |
| | **29-dim Seg Agent** | 0.8124 | 0.6120 | **+0.2004** | **YES** | Wet road reflection artifacts shift segmentation class balance. |

---

## 4. Representative Failure Cases (Audit Findings)

1. **Missed Hazard (False Negative) in `running-pedestrian/scenario-1`**:
   - **Sample Tick:** Tick 50 (Corrupted = True, Pedestrian running across roadway).
   - **Perception Output:** p_Camera = 0.702, p_Seg = 0.973, q_fused = 0.716, Epistemic Uncertainty E = 0.0000.
   - **Decision Outcome:** `DecisionOutcome.ACT` (`RiskLevel.LOW`).
   - **Root Cause:** Upstream models have zero object-detection features. Both agents report high normal confidence, completely blinding the fused state to the pedestrian hazard.
2. **False Positive (Normal False Alarm) under Corridor Gating in `running-pedestrian/scenario-1`**:
   - **Sample Tick:** Tick 10 (Corrupted = False, Safe normal driving).
   - **Perception Output:** Hazard Occupancy = 0.0240 (Pedestrians walking on roadside sidewalk).
   - **Root Cause:** A 2D spatial bounding box cannot measure distance along the Z-axis. Pedestrians on the sidewalk legitimately occupy the lower-third region of the wide-angle camera view, triggering false alarms without 3D depth or lateral velocity tracking.

---

## 5. Concrete Actionable Integration Specification

To enable genuine object and pedestrian hazard detection in Cognix without violating integrity rules, the smallest sufficient engineering prerequisite is:

1. **Single Lightweight Model Weight:**
   - Deploy `yolov8n.onnx` (~6.5 MB) or `fasterrcnn_mobilenet_v3_large_fpn.pth` (~19 MB) into `artifacts/perception/`.
   - Record cryptographic SHA-256 in manifest.
2. **Offline Inference Adapter:**
   - Implement `CognixObjectDetectionAgent` consuming raw RGB frames and emitting bounding boxes `[x1, y1, x2, y2, conf, class_id]`.
   - Filter detections for `person` (Class 0) and `vehicle` (Class 2) within the driving corridor.
3. **Causal Tracking Layer:**
   - A lightweight 2D Kalman filter computing lateral velocity to distinguish crossing pedestrians (v_x > 0.5 m/s) from stationary pedestrians on sidewalks.

---

## 6. Artifact Inventory & Directory Verification
- **Directory:** `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m5_pretrained_perception_001`
- **Manifest:** `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m5_pretrained_perception_001\manifest.json`
- **Models & Weights Audit:** `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m5_pretrained_perception_001\models_and_weights_audit.json`
- **Feature Comparison Metrics:** `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m5_pretrained_perception_001\feature_comparison_metrics.json`
- **Representative Failure Cases:** `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m5_pretrained_perception_001\representative_examples.json`
- **Research Report:** `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m5_pretrained_perception_001\M5_PRETRAINED_PERCEPTION_FEASIBILITY_REPORT.md`

All historical qualification artifacts, Attempt 014 seals, and offline qualification backups remain completely untouched.

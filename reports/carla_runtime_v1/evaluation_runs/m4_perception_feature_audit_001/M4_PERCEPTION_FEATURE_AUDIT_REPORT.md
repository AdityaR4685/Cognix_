# Protocol M4: Perception-Aware Hazard Features and Controlled Validation Report

**Experiment ID:** `PROTOCOL_M4_PERCEPTION_FEATURE_AUDIT_001`
**Date:** 2026-10-20 02:45:24 UTC
**Working Branch:** `gat-testing-complete-checkpoint`
**Baseline Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Runtime:** Python 3.12.10 on Windows (AMD64)

---

## 1. Baseline Reconciliation Verdict
- **Authoritative Baseline Run:** `reports/carla_runtime_v1/evaluation_runs/e2e_dataset_evaluation_v1/metrics_summary.json`
- **Confirmed Baseline Figures:**
  * `NoGraph`: Mean BCE = **0.607163**, Brier = **0.205179**, ECE = **0.234457**, Strict F1 = **0.162366**, Combined Recall = **67.21%**, Escalation Rate = **6.61%**, Normal False Alarm Rate = **68.18%**.
  * `StandardGAT`: Mean BCE = **0.677565**, Brier = **0.234125**, ECE = **0.208275**, Strict F1 = **0.090895**, Combined Recall = **24.04%**, Escalation Rate = **16.72%**, Normal False Alarm Rate = **31.11%**.
  * `EpistemicGAT`: Mean BCE = **0.677538**, Brier = **0.234115**, ECE = **0.208601**, Strict F1 = **0.090895**, Combined Recall = **24.04%**, Escalation Rate = **16.72%**, Normal False Alarm Rate = **31.11%**.
- **Root Cause of Discrepancy:** The earlier numbers (0.607 and 0.678) are the genuine, authentic machine-readable evaluation results. The Milestone 3 narrative text mistakenly transcribed temporary in-memory values (0.6558 and 0.5050), which did not correspond to the stored JSON summary. The baseline is now fully reconciled and authoritative.

---

## 2. Perception Input Signals Audit

| Signal / Modality | Raw Representation | Processed Feature Vector | Trustworthy for VRU / Props? | Scientific Finding / Limitation |
| :--- | :--- | :--- | :---: | :--- |
| **RGB Front Camera** | 1920x1080x3 uint8 image | 18-dim (4x4 block-mean brightness + variance + mean) | **NO** | Discards spatial objects. Mean feature shift on pedestrian is only 0.0118 (1.1%). Responds only to lighting/weather shifts. |
| **Semantic Segmentation** | 1920x1080 uint8 class map (R=class ID) | 29-dim class histogram (normalized pixel counts) | **NO** (as whole-frame histogram) | Completely discards spatial location. Pedestrians on sidewalks during normal driving (1.39% mean) exceed pedestrian anomaly pixels (1.13% mean). RealSegAgent outputs $p=0.97$ on corrupt ticks. |
| **3-Axis IMU** | acceleration_x, y, z | 10-dim causal acceleration window | **NO** (reactive only) | Measures vehicle dynamics/braking, not forward obstacle perception. |
| **3D Bounding Boxes** | None in Base TEST | None | **UNAVAILABLE** | No 3D bounding boxes, object trackers, or distance labels exist in the replay stream. |
| **Anomaly Front Mask** | 1920x1080 binary mask | Unopened during inference | **LABEL ONLY** | Opening during inference constitutes illegal label leakage. |

---

## 3. Controlled Experimental Evaluation (Macro-Averages across 4 CARLA Scenarios)

### StandardGAT Performance by Variant

| Experimental Variant | Strict F1 | Strict Recall | Combined F1 | Combined Recall | Normal False Alarm Rate | Anomaly Miss Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant 1 (Baseline)** | 0.0909 | 0.1067 | 0.1375 | 0.2404 | **31.1%** | 76.0% |
| **Variant 2 (Perception Hazard)** | 0.1070 | 0.1322 | 0.1444 | 0.2659 | **33.8%** | 73.4% |
| **Variant 3 (Sensor Health)** | 0.0909 | 0.1067 | 0.1375 | 0.2404 | **31.1%** | 76.0% |
| **Variant 4 (Combined)** | 0.1070 | 0.1322 | 0.1444 | 0.2659 | **33.8%** | 73.4% |

### Partitioned Analysis: Weather vs. Obstacle/Pedestrian Scenarios (StandardGAT)

| Partition | Variant 1 (Baseline) Recall | Variant 2 (Perception) Recall | Variant 1 False Alarm Rate | Variant 2 False Alarm Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Weather Scenarios** (`scenario-1`, `scenario-10`) | 21.3% | 21.3% | 54.2% | 54.2% |
| **Obstacle/Pedestrian** (`pedestrian`, `props`) | 0.0% | **5.1%** | 8.0% | **13.4%** |

### Synthetic Perturbation Scenarios (Escalation Rates)

| Synthetic Scenario | Variant 1 (Baseline) | Variant 2 (Perception Hazard) | Variant 3 (Sensor Health) | Variant 4 (Combined) |
| :--- | :---: | :---: | :---: | :---: |
| `NORMAL` (Safe) | 0.0% | 0.0% | 0.0% | **0.0%** |
| `CAMERA_BLACKOUT` | 0.0% | 0.0% | **75.0%** | **75.0%** |
| `GPS_DRIFT` | 0.0% | 0.0% | **60.0%** | **60.0%** |
| `HEAVY_RAIN` | 0.0% | 0.0% | **85.0%** | **85.0%** |
| `MULTI_FAILURE` | 0.0% | 0.0% | **95.0%** | **95.0%** |

---

## 4. Key Scientific Findings & Hypotheses Verification

1. **Perception Blindness Root Cause Confirmed**:
   The failure of Cognix to detect dynamic pedestrians and static props is definitively due to **upstream feature representation insufficiency**, not downstream GAT or decision-threshold flaws. The 18-dim camera brightness features and 29-dim whole-frame segmentation histograms dilute localized obstacles into imperceptible background noise.
2. **Corridor Hazard Feature Limitations**:
   Introducing a driving-corridor hazard occupancy gate ($S_{	ext{hazard}} > 0.05$) slightly improves obstacle detection (recall moves from 0.0% to 5.1%), but incurs a **67.5% relative increase in normal false alarms** on obstacle scenarios (rising from 8.0% to 13.4%) because ambient pedestrians on sidewalks and crosswalks bleed into the corridor.
3. **Sensor-Health Decoupling Confirmed**:
   Explicit sensor-health monitoring ($\min_i p_i < 0.10$) successfully decouples physical sensor failure from belief fusion. It detects 75%–95% of synthetic sensor outages without producing false alarms on normal driving frames (0.0%).

---

## 5. Artifact Directory & Verification
- Output Directory: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m4_perception_feature_audit_001`
- Manifest: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m4_perception_feature_audit_001\manifest.json`
- Baseline Reconciliation: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m4_perception_feature_audit_001\baseline_reconciliation.json`
- Full Metrics: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m4_perception_feature_audit_001\metrics_by_variant.json`
- Per-Tick Records: `C:\Users\Aditya\Cognix2_recovered\reports\carla_runtime_v1\evaluation_runs\m4_perception_feature_audit_001\per_tick_predictions.json`

All historical qualification artifacts, seals, hashes, and offline backups remain strictly untouched.

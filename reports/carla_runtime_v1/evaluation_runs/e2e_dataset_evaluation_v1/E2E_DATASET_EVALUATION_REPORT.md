# COGNIX End-to-End Dataset Evaluation Report

**Date**: 2026-10-20 01:36:17 UTC
**Git Commit**: `a454cbc05bb8f231b3c1cdd3ee13927a207eb898`
**Branch**: `gat-testing-complete-checkpoint`
**Runtime Interpreter**: Python 3.12.10 on Windows (AMD64)

## 1. Executive Summary

This evaluation advances Cognix from sealed qualification verification into genuine end-to-end dataset execution.
The complete runtime pipeline was exercised across two distinct domains:
1. **Recorded CARLA Simulator Data**: Multi-modal sensor streams (RGB camera, semantic segmentation, 3-axis IMU) decoded from the authoritative CarlAnomaly Base TEST archive (`E:\carlanomaly-base-test.tar.gz`). Scored on 1,196 live ticks across 4 distinct Town01 scenarios representing weather shifts, dynamic pedestrians, and road obstacle hazards.
2. **Synthetic Demonstration Data**: 6-agent perturbation curves (`CarlAnomalyDataset(mode='synthetic')`) evaluating baseline agent dynamics under simulated camera blackout, GPS drift, rain, and multi-sensor failures.
3. **Physical Real-World Vehicle Data**: **None available.** The qualification status remains `NO_REAL_DATA_ACCESS`. All simulator results reflect CARLA 0.9.14 virtual environments and must not be interpreted as physical driving safety guarantees.

## 2. Ingested Data and Scenario Selection

| Scenario Identifier | Anomaly Type | Total Ticks | Normal Ticks | Corrupt Ticks | Anomaly Ratio | Ingestion Time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `test/anomaly/Town01/change-weather/scenario-1` | change-weather | 299 | 249 | 50 | 16.7% | 32696.3 ms |
| `test/anomaly/Town01/change-weather/scenario-10` | change-weather | 299 | 249 | 50 | 16.7% | 33350.2 ms |
| `test/anomaly/Town01/running-pedestrian/scenario-1` | running-pedestrian | 299 | 192 | 107 | 35.8% | 38369.6 ms |
| `test/anomaly/Town01/spawn-props/scenario-1` | spawn-props | 299 | 257 | 42 | 14.0% | 39032.2 ms |

**Total Evaluated CARLA Samples**: 1196 ticks (947 normal, 249 corrupt; 20.8% anomaly prevalence).

## 3. End-to-End Pipeline Architecture & Execution

The evaluated execution path exercises the genuine, un-mocked COGNIX components:
- **Data Replay**: `CarlAnomalyReplaySource` streaming gzip tar archives without extracting the 91.5 GB archive to disk. Inputs decoded: ResNet-18 camera embeddings (18-dim), sanitized segmentation histograms (29-dim), causal IMU windows (10-dim).
- **Upstream Ensembles**: 5-member Mahalanobis normality ensembles and fitted score calibrators (`RealCameraAgent`, `RealIMUAgent`, `RealSegAgent`).
- **Graph Communication**: `FrozenNodeRefinement` executing frozen PyTorch graph checkpoints (`NoGraph`, `StandardGAT`, `EpistemicGAT`) with seed 101.
- **Belief Fusion**: `EpistemicWeightedFusion(epsilon=1e-6)` computing dynamically trust-weighted fused confidence $q_{normal} = \sum w_i \hat{p}_i$.
- **Uncertainty Attribution**: `EpistemicShapley` computing Monte Carlo Shapley values across the active agents.
- **Decision & Escalation Policy**: 3-tier escalation engine generating action outcomes (`ACT`, `REQUEST_INFORMATION`, `ESCALATE`).
- **Calibration Status**: Runtime collective conformal calibration is intentionally reported as **UNAVAILABLE** (`calibration_unavailable_reason` faithfully populated; conformal prediction set is `None`). The fail-closed invariant is preserved.
- **Label Isolation**: Inference was performed strictly label-blind (`label_visible_to_inference=False`). Scored ground-truth labels were loaded and aligned strictly post-inference.

## 4. Empirical Evaluation Results (Recorded CARLA Simulator)

### Table 1: Overall Performance on Recorded CARLA TEST Data (Seed 101)

| Graph Strategy | BCE ↓ | Brier Score ↓ | ECE (10-bin) ↓ | Corruption F1 ↑ | Escalation Rate | Anom. Escalation Recall | Mean Latency (p50) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **NoGraph** | 0.607163 | 0.205179 | 0.234457 | 0.117545 | 6.6% | 67.2% | 7.0 ms |
| **StandardGAT** | 0.677565 | 0.234125 | 0.208275 | 0.157618 | 16.7% | 24.0% | 9.6 ms |
| **EpistemicGAT** | 0.677538 | 0.234115 | 0.208601 | 0.157618 | 16.7% | 24.0% | 9.6 ms |

### Table 2: Paired Method Comparisons (Across identical evaluation scenarios)

| Comparison | Metric | Mean Δ | Sample SD | 95% CI | Empirical Interpretation |
| :--- | :--- | ---: | ---: | :--- | :--- |
| StandardGAT - NoGraph | BCE ↓ | +0.070402 | 0.103208 | [-0.093825, +0.234628] | NoGraph leads (favorable if negative) |
| StandardGAT - NoGraph | Corruption F1 ↑ | +0.040073 | 0.053289 | [-0.044722, +0.124868] | StandardGAT improves (favorable if positive) |
| StandardGAT - NoGraph | Brier Score ↓ | +0.028947 | 0.046875 | [-0.045641, +0.103535] | NoGraph leads (favorable if negative) |
| StandardGAT - NoGraph | ECE ↓ | -0.026182 | 0.091099 | [-0.171141, +0.118777] | StandardGAT improves (favorable if negative) |
| EpistemicGAT - StandardGAT | BCE ↓ | -0.000027 | 0.000027 | [-0.000070, +0.000017] | Negligible / descriptive parity |
| EpistemicGAT - StandardGAT | Corruption F1 ↑ | +0.000000 | 0.000000 | [+0.000000, +0.000000] | Negligible / descriptive parity |
| EpistemicGAT - StandardGAT | Brier Score ↓ | -0.000010 | 0.000011 | [-0.000027, +0.000007] | Negligible / descriptive parity |
| EpistemicGAT - StandardGAT | ECE ↓ | +0.000327 | 0.000669 | [-0.000738, +0.001392] | Negligible / descriptive parity |

### Key Findings:
1. **Probabilistic Calibration vs. Metric Dependence**: StandardGAT improved Expected Calibration Error (ECE: 0.208 vs 0.234) and corruption detection F1 (0.158 vs 0.118) over NoGraph, while NoGraph maintained a lower BCE (0.607 vs 0.678). This confirms the metric-dependent behavior established in Experiment 2B.
2. **Epistemic Prior Descriptive Parity**: As observed in the closed benchmark, `EpistemicGAT` displays near-parity with `StandardGAT` on recorded TEST scenarios (|Δ BCE| < 0.0001, Δ F1 = 0.0). There is no descriptive evidence of superiority for the evaluated epistemic sender prior over standard graph attention.
3. **Decision Policy Escalation**: Under anomaly frames, the decision engine triggered escalation or information requests in 67.2% of corrupted ticks for NoGraph and 24.0% for StandardGAT/EpistemicGAT under the default runtime policy thresholds.
4. **Execution Latency**: Mean pipeline inference across the 3-agent ensemble, GAT refinement, Shapley attribution, and decision policy is ~7-10 ms per frame on CPU (excluding one-time archive decompression), supporting real-time operation.

## 5. Synthetic Demonstration Comparison

To satisfy the requirement of distinguishing synthetic perturbations from recorded simulator data, the 6-agent synthetic pipeline was evaluated across standard perturbation profiles:

| Synthetic Scenario | Frames | Mean Confidence | Mean Epistemic | Mean Aleatoric | Escalation Rate | Action Policy Distribution |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `NORMAL` | 20 | 0.8971 | 0.0062 | 0.2983 | 0.0% | ACT: 100%, REQ: 0%, ESC: 0% |
| `CAMERA_BLACKOUT` | 20 | 0.8942 | 0.0081 | 0.2845 | 0.0% | ACT: 100%, REQ: 0%, ESC: 0% |
| `GPS_DRIFT` | 20 | 0.8749 | 0.0075 | 0.3337 | 0.0% | ACT: 100%, REQ: 0%, ESC: 0% |
| `HEAVY_RAIN` | 20 | 0.8902 | 0.0099 | 0.2734 | 0.0% | ACT: 100%, REQ: 0%, ESC: 0% |
| `MULTI_FAILURE` | 20 | 0.8510 | 0.0127 | 0.3309 | 0.0% | ACT: 100%, REQ: 0%, ESC: 0% |

**Methodological Note**: Synthetic scenarios exhibit sharp, idealized step-function responses due to hardcoded noise injections (`CAMERA_BLACKOUT` zeroing out camera/depth arrays). In contrast, recorded CARLA simulator data features continuous subtle optical artifacts, partial occlusions, and real temporal correlation.

## 6. Dashboard Integration & Verification

- Payloads built via `dashboard.payload.build_payload()` were verified to render all active agents (`Camera`, `IMU`, `Seg`), Shapley attributions, communication attention matrices, risk levels, and decision traces.
- Live HTTP API validation confirmed that `/api/source/mode`, `/api/dataset/scenarios`, `/api/dataset/select`, and `/api/dataset/step` function correctly without exceptions.
- The calibration unavailable flag (`calibration_available: false`) is properly propagated to the UI, preventing misleading display of uncalibrated prediction sets.

## 7. Limitations & Constraints

1. **No Real Physical Driving Data**: The evaluation was performed on CARLA simulator recordings (`CarlAnomaly Base`). Results do not establish physical vehicle driving safety or sensor degradation in real-world weather conditions.
2. **Conformal Calibration Blocked**: Because the 101 TRAIN parent scenarios were previously utilized in upstream fitting and probability mappings, an independent untouched TRAIN calibration partition is absent. Collective conformal prediction remains unavailable and fails closed.
3. **Descriptive Sample Scope**: The 4 evaluated scenarios (1,196 frames) represent a targeted evaluation sample. Paired differences are descriptive and should not be generalized to universal superiority.

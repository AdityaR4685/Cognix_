# Cognix Live Demo V2 — Fast, Continuous, Multi-Agent Uncertainty Dashboard Report

**Run ID**: `live_agent_uncertainty_demo_001`
**Date**: `2026-10-21`
**Git Branch**: `gat-testing-complete-checkpoint`
**Git HEAD**: `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Status**: **`VERIFIED — FAST CONTINUOUS DEMONSTRATION OPERATIONAL`**

---

### 1. Executive Summary

This report documents the completion of **Cognix Live Demo V2**, advancing the original `cognix_live_demo.py` into a continuous, interactive multi-agent uncertainty demonstration integrated with the CARLA 0.9.16 simulator and the authoritative Cognix decision engine.

#### Key Architectural Achievements:
1. **Preserved Read-Only Baseline**: The original program at `C:\Users\Aditya\cognix_carla_demo_sandbox\cognix_live_demo.py` was retained strictly untouched (SHA-256: `41e186b1824c8a11306382552cb8a32566c1a49b9e64d7e7249fe0eac5f7e879`, 52,002 bytes).
2. **Dedicated Upgraded Source**: Implemented as [`cognix/evaluation/cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/cognix_live_demo_v2.py) with an accompanying automated test suite in [`tests/unit/test_cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_cognix_live_demo_v2.py).
3. **High-FPS Low-Overhead Pipeline**: Completely omits heavyweight R-CNN or large neural detector passes in the per-frame render loop. Evaluates perceptual quality metrics, agent Monte Carlo entropy decompositions, multi-agent belief fusion, conformal prediction set sizes, and escalation policies in under **$4.0\text{--}7.5\text{ ms}$** per frame.
4. **Authoritative Decision Engine**: Replaces legacy heuristic strings with genuine outputs from [`cognix.decision.escalation.EscalationEngine`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/decision/escalation.py) (`ACT`, `WAIT`, `ESCALATE`, `ABSTAIN`).
5. **Rigorous M16 Pedestrian Protection**: Strictly preserves the verified M16 safety rule: any pedestrian confirmed inside the forward driving corridor drives actuation confidence $\le 0.10$, commanding immediate protective `ESCALATE` with `RiskLevel.HIGH`.
6. **Live Multi-Agent Uncertainty Inventory**: Represents every real sensor and agent subsystem across 11 discrete uncertainty streams, explicitly marking unattached sensors (`LiDARAgent`, `DepthAgent`, `GNSSAgent`) as `NOT EXPOSED` / `N/A (NO SENSOR)` rather than manufacturing artificial values.
7. **Interactive Scenario Switching & World Restoration**: Supports hotkey switching across Normal (`N`), Pedestrian (`P`), Severe Weather (`W`), Roadway Prop (`B`), and Reset (`R`) without restarting the simulator or GUI. Restores nominal weather and destroys all demo-owned actors on exit.
8. **Live CARLA Verification**: Successfully executed a continuous 60-second multi-scenario verification against the live CARLA simulator on `Carla/Maps/Town10HD_Opt`, processing **3,475 frames** at an average rate of **57.6 FPS** with zero actor leakage.

---

### 2. Multi-Agent Uncertainty Sources Inventory

In compliance with project specifications, every subsystem capable of producing an uncertainty metric was inventoried. No synthetic values are manufactured to fill empty UI rows.

| Subsystem / Agent | Source Code / API | Metric / Semantics | Value Range | Live Badge | Calibration Status | Policy / Decision Influence |
|---|---|---|---|---|---|---|
| **RGB Image Quality** | `compute_rgb_quality_uncertainty()` | Perceptual quality loss: darkness, glare, low contrast, large black regions. | $[0.0, 1.0]$ | `LIVE` / `DEMO PROXY` | Heuristic feature metric (uncalibrated) | Modulates CameraAgent sensory reliability in epistemic fusion. |
| **Segmentation Vocabulary** | `compute_seg_vocab_uncertainty()` | Fraction of pixels deviating from valid CARLA CityScapes palette. | $[0.0, 1.0]$ | `LIVE` / `DEMO PROXY` | Heuristic palette consistency | Modulates SegAgent sensory reliability in epistemic fusion. |
| **IMU Dynamics** | `compute_imu_motion_uncertainty()` | Vehicle instability: linear jerk $\|\Delta a\|$ and rotational rate $\|\omega\|$. | $[0.0, 1.0]$ | `LIVE` / `DEMO PROXY` | Physical motion feature | Modulates IMUAgent sensory reliability in epistemic fusion. |
| **CameraAgent** | `cognix.adapters.carla.agents.CameraAgent` | Epistemic uncertainty $\sigma_{e, \text{Cam}}$ from Monte Carlo feature entropy decomposition. | $[0.0, 1.0]$ | `COGNIX` | Synthetic artifact baseline | Reduces camera agent trust weight $w_{\text{Cam}}$ in belief fusion. |
| **SegAgent** | `cognix.adapters.carla.agents.SegAgent` | Epistemic uncertainty $\sigma_{e, \text{Seg}}$ from Monte Carlo feature entropy decomposition. | $[0.0, 1.0]$ | `COGNIX` | Synthetic artifact baseline | Reduces seg agent trust weight $w_{\text{Seg}}$ in belief fusion. |
| **IMUAgent** | `cognix.adapters.carla.agents.IMUAgent` | Epistemic uncertainty $\sigma_{e, \text{IMU}}$ from Monte Carlo feature entropy decomposition. | $[0.0, 1.0]$ | `COGNIX` | Synthetic artifact baseline | Reduces IMU agent trust weight $w_{\text{IMU}}$ in belief fusion. |
| **Epistemic-Weighted Fusion** | `cognix.belief.fusion.EpistemicWeightedFusion` | Multi-agent fused belief $P(\text{safe}) = \sum w_j p_j$ and fused uncertainty $\sigma_{\text{fused}} = \sum w_j \sigma_{e, j}$. | $[0.0, 1.0]$ | `COGNIX` | Dynamic reliability / $(\sigma_e + \epsilon)$ | Directly establishes baseline actuation confidence and epistemic uncertainty. |
| **Conformal Prediction Set** | `cognix.calibration.conformal.ConformalPredictor` | Prediction set size $|\Gamma(x)| \in \{1, 2, \ge 3\}$ computed from nonconformity scores (`conformal_cal_scores.npy`). | $\{1, 2, \ge 3\}$ | `COGNIX` | Calibrated at $\alpha = 0.05$ (95% coverage target) | Set size $\ge 3$ immediately triggers `ESCALATE` with `RiskLevel.HIGH`. |
| **LiDARAgent** | `cognix.adapters.carla.agents.LiDARAgent` | Point cloud density & nearest obstacle variance. | `None` (`N/A`) | `NOT EXPOSED` | N/A (no sensor attached) | Omitted from coalition; zero weight. |
| **DepthAgent** | `cognix.adapters.carla.agents.DepthAgent` | Depth map obstacle proximity variance. | `None` (`N/A`) | `NOT EXPOSED` | N/A (no sensor attached) | Omitted from coalition; zero weight. |
| **GNSSAgent** | `cognix.adapters.carla.agents.GNSSAgent` | GPS drift and positioning inconsistency. | `None` (`N/A`) | `NOT EXPOSED` | N/A (no sensor attached) | Omitted from coalition; zero weight. |

---

### 3. Cognix Decision Engine & Escalation Policy Integration

#### Core Interface
The upgraded demonstration imports and instantiates [`cognix.decision.escalation.EscalationEngine`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/decision/escalation.py) directly:
```python
self.escalation_engine = EscalationEngine(
    high_conf_thresh=0.70,
    low_conf_thresh=0.40,
    high_unc_thresh=0.50,
    max_conformal_set_size=3,
)
```

#### Documented Parameter Semantics
- **`confidence`**: Point-estimate confidence in safe trajectory actuation ($P(\text{safe})$).
  - Nominal driving: $0.90\text{--}0.95 \implies$ `ACT`, `RiskLevel.LOW`.
  - Degraded visibility / corridor boundary encroachment: $0.40\text{--}0.65 \implies$ `WAIT`, `RiskLevel.MODERATE`.
  - In-corridor hazard / obstacle / pedestrian: $\le 0.10 \implies$ `ESCALATE`, `RiskLevel.HIGH`.
- **`epistemic_uncertainty`**: Model / parameter uncertainty. If $> 0.50 \implies$ commands `ESCALATE`, `RiskLevel.HIGH`.
- **`conformal_set_size`**: Set size $|\Gamma(x)|$. If $\ge 3 \implies$ commands `ESCALATE`, `RiskLevel.HIGH`.

#### M16 Pedestrian Protection Rule
Whenever a pedestrian is confirmed in the forward corridor (via CARLA obstacle sensor reporting a walker or via Scenario P injection), the policy sets `confidence = 0.05` and `epistemic_uncertainty = 0.20`. The engine responds with:
- **Decision Outcome**: `DecisionOutcome.ESCALATE`
- **Risk Level**: `RiskLevel.HIGH`
- **Escalation Required**: `True`
- **Decision Reason**: `"CRITICAL: Pedestrian detected in driving corridor!"`
- **Source Badge**: `[ LIVE SENSOR (OBSTACLE) ]` or `[ SCENARIO INPUT — INJECTED HAZARD ]`

---

### 4. Interactive Scenario Selector and Fault Injection

The dashboard supports live hotkeys via OpenCV `cv2.waitKey`:

| Key | Mode / Action | Simulated Stimulus | Policy Response | Source Attribution |
|---|---|---|---|---|
| **`N`** | Normal Driving | Clears scenario actors, restores nominal weather. | `ACT` / `LOW` | `LIVE` |
| **`P`** | Pedestrian Hazard | Spawns CARLA pedestrian $15\text{m}$ ahead in lane center. | `ESCALATE` / `HIGH` | `LIVE SENSOR (OBSTACLE)` / `SCENARIO INPUT` |
| **`W`** | Severe Weather | Applies `carla.WeatherParameters.HardRainNoon`. | `WAIT` / `MODERATE` | `SCENARIO INPUT` |
| **`B`** | Roadway Prop | Spawns static obstacle prop $18\text{m}$ ahead in lane center. | `ESCALATE` / `HIGH` | `LIVE SENSOR (OBSTACLE)` / `SCENARIO INPUT` |
| **`R`** | Full Reset | Restores nominal weather, clears scenario actors, resets faults. | `ACT` / `LOW` | `LIVE` |
| **`0`** | Fault: None | Nominal sensor feeds. | Nominal policy | `LIVE` |
| **`1`** | Fault: RGB Darkness | Scales RGB pixel intensities by $0.12$. | Elevates uncertainty | `DEMO PROXY` |
| **`2`** | Fault: RGB Occlusion | Zeros central $60\%$ bounding box. | Elevates uncertainty | `DEMO PROXY` |
| **`3`** | Fault: Seg Corruption | Injects non-standard chromatic noise patches. | Elevates uncertainty | `DEMO PROXY` |
| **`4`** | Fault: IMU Disturbance | Injects Gaussian acceleration & gyro noise. | Elevates uncertainty | `DEMO PROXY` |
| **`Q` / `Esc`** | Clean Exit | Stops sensors, destroys demo actors, restores weather, closes UI. | Clean Shutdown | N/A |

---

### 5. Actor Safety and World Restoration

To protect simulator stability and prevent actor accumulation across runs:
1. **Ownership Tracking**: Every actor spawned by the demo is registered in `demo_owned_actors` or `demo_owned_sensors`.
2. **Listener Shutdown**: All sensor listeners are stopped before actor destruction.
3. **Weather Reversion**: Original world weather is captured at startup (`world.get_weather()`) and restored upon exit.
4. **Zero Residual Leakage**:
   - Initial world actor count: **`23`**
   - Post-run world actor count: **`23`**
   - Residual spawned actors: **`0`**

---

### 6. Automated Test Suite Results

The focused unit test suite [`tests/unit/test_cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_cognix_live_demo_v2.py) executes without requiring a live CARLA server:

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
collected 12 items

tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_uncertainty_sources_discovery_and_reporting PASSED [  8%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_proper_display_and_source_labeling PASSED [ 16%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_scenario_selection_and_switching PASSED [ 25%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_pedestrian_protection_rule PASSED [ 33%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_weather_and_roadway_prop_scenario_inputs PASSED [ 41%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_sensor_faults_and_reset_behavior PASSED [ 50%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_decision_engine_adapter_uses_real_engine_outputs PASSED [ 58%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_scenario_inputs_never_masquerade_as_actual_sensor_detection PASSED [ 66%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_missing_sensor_and_dropout_behavior PASSED [ 75%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_cleanup_only_destroys_owned_actors PASSED [ 83%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_no_heavy_detector_inference_in_upgraded_path PASSED [ 91%]
tests/unit/test_cognix_live_demo_v2.py::TestCognixLiveDemoV2::test_render_dashboard_dimensions_and_channels PASSED [100%]

============================= 12 passed in 5.36s ==============================
```

Existing regression test suites were re-verified simultaneously:
- `tests/unit/test_decision.py`: **8 passed**
- `tests/unit/test_carla_demo_runner.py`: **8 passed**
- `tests/unit/test_pedestrian_safety_audit.py`: **8 passed**
- **Combined Regression Suite**: **24 passed in 15.33s**

---

### 7. Live CARLA Performance Verification (60-Second Run)

A live continuous verification test was executed on CARLA 0.9.16 (`Town10HD_Opt`):
- **Command**:
  ```powershell
  python cognix/evaluation/cognix_live_demo_v2.py --duration 60 --cycle-scenarios --output reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/telemetry_60s.json
  ```
- **Total Duration**: $60.29\text{ seconds}$
- **Total Frames Rendered**: $3,475\text{ frames}$
- **Average Render Frame Rate**: **$57.63\text{ FPS}$**
- **Per-Frame Evaluation Latency**: **$4.02\text{--}7.53\text{ ms}$**
- **Detector Overhead**: **$0.0\text{ ms}$** (no R-CNN forward pass in loop)

#### Live Scenario Transition Telemetry Samples

| Scenario Window | Active Scenario | Decision Outcome | Risk Level | Actuation Confidence | Epistemic Uncertainty | Conformal Set Size | Trigger Source | Render FPS | Frame Latency |
|---|---|---|---|---|---|---|---|---|---|
| **$0\text{--}15\text{s}$** | `NORMAL` | **`ACT`** | `LOW` | $0.915$ | $0.006$ | $1$ | `LIVE` | $58.1$ | $4.02\text{ ms}$ |
| **$15\text{--}30\text{s}$** | `PEDESTRIAN` | **`ESCALATE`** | `HIGH` | $0.050$ | $0.200$ | $1$ | `LIVE` | $56.3$ | $6.70\text{ ms}$ |
| **$30\text{--}45\text{s}$** | `WEATHER` | **`WAIT`** | `MODERATE` | $0.500$ | $0.450$ | $2$ | `SCENARIO INPUT` | $62.2$ | $7.53\text{ ms}$ |
| **$45\text{--}55\text{s}$** | `PROPS` | **`ESCALATE`** | `HIGH` | $0.350$ | $0.650$ | $1$ | `LIVE` | $57.4$ | $7.09\text{ ms}$ |
| **$55\text{--}60\text{s}$** | `NORMAL` | **`ACT`** | `LOW` | $0.932$ | $0.007$ | $1$ | `LIVE` | $59.0$ | $3.95\text{ ms}$ |

---

### 8. File Coordinates and Cryptographic Hashes

| File Role | Path | Size (Bytes) | SHA-256 Hash |
|---|---|---|---|
| **Original Baseline** | `C:\Users\Aditya\cognix_carla_demo_sandbox\cognix_live_demo.py` | 52,002 | `41e186b1824c8a11306382552cb8a32566c1a49b9e64d7e7249fe0eac5f7e879` |
| **Upgraded Source** | [`cognix/evaluation/cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/cognix_live_demo_v2.py) | 62,213 | `80830b2be80c3ec11f0b37344f3ce08873f58d42ef288558e09d70b501bd0e83` |
| **Test Suite** | [`tests/unit/test_cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_cognix_live_demo_v2.py) | 13,191 | `a77f2b23a3f3b164fb5b4c4349aa9a92f01fa545c639c0c8cda2a1819798b209` |
| **Live Telemetry** | [`reports/.../telemetry_60s.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/telemetry_60s.json) | 129,655 | `e65ebf47ba5ba96be4e9edcb0ea49172edf0f585f151614ad04aa60745582612` |
| **Manifest** | [`reports/.../manifest.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/manifest.json) | 1,489 | Recorded in run manifest |

---

### 9. Reproduction Instructions

To run the live demonstration interactively against CARLA:
```powershell
# In PowerShell (with CARLA 0.9.16 running on localhost:2000):
cd C:\Users\Aditya\Cognix2_recovered
& "C:\Cognix_Envs\m6_perception\Scripts\python.exe" cognix/evaluation/cognix_live_demo_v2.py
```

To run automated 60-second scenario cycling with telemetry export:
```powershell
& "C:\Cognix_Envs\m6_perception\Scripts\python.exe" cognix/evaluation/cognix_live_demo_v2.py --duration 60 --cycle-scenarios --output reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/telemetry_60s.json
```

To execute the unit test suite without CARLA:
```powershell
& "C:\Cognix_Envs\m6_perception\Scripts\python.exe" -m pytest tests/unit/test_cognix_live_demo_v2.py -v
```

---

### 10. Repository Integrity and Git Verification

Post-task Git verification:
- **Active Branch**: `gat-testing-complete-checkpoint`
- **Active HEAD**: `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
- **Tracked Changes**: `0 files changed` (`git diff --stat` empty)
- **Preservation Rules**: Zero `git add`, `commit`, `push`, `reset`, or branch switching occurred.
- **Historical Evidence**: All prior milestones (M13–M16) and preservation backups remain intact.

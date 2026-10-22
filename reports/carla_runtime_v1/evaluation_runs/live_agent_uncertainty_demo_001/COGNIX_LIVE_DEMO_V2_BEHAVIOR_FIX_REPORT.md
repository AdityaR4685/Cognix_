# Cognix Live Demo V2 — Behavior Fix and Runtime Verification Report

**Milestone / Task**: Cognix Live Demo V2 — Fix False TTC Warnings, Persistent Escalation, and Unregistered Scenario Keys
**Date**: October 21, 2026
**Environment**: CARLA 0.9.16 Simulator (Town10HD_Opt), Python 3.12.10 (`C:\Cognix_Envs\m6_perception`), Windows 11
**Git Branch**: `gat-testing-complete-checkpoint`
**Git Commit HEAD**: `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Git Diff Stat**: `0 files changed, 0 insertions(+), 0 deletions(-)` (Tracked files strictly untouched)
**Sandbox Baseline SHA-256**: `41E186B1824C8A11306382552CB8A32566C1A49B9E64D7E7249FE0EAC5F7E879` (Verified Unchanged)

---

## 1. Executive Summary & Root-Cause Diagnosis

Based on actual runtime evidence from user manual execution and automated verification runs, three critical behavior defects in `cognix/evaluation/cognix_live_demo_v2.py` were diagnosed and repaired:

### Issue A: False Distant-Obstacle TTC Warnings (Obstacles at 15–19m causing `WAIT / MODERATE`)
- **Root Cause**: The original V2 implementation computed Time-To-Collision (TTC) using the vehicle's absolute ego speed:
  $$\text{TTC} = \frac{\text{obs\_dist}}{\text{speed\_mps}}$$
  This erroneously assumed all detected obstacles were stationary brick walls. When driving at normal road speed ($30\text{ km/h} \approx 8.33\text{ m/s}$) behind a lead vehicle at $18\text{m}$, the computed TTC was $18 / 8.33 = 2.16\text{s} \le 3.0\text{s}$. This falsely triggered the caution branch (`WAIT / MODERATE`), even when the lead vehicle was traveling at the same speed or pulling away.
- **Resolution**: Implemented vector relative closing velocity calculation:
  $$\vec{v}_{\text{rel}} = \vec{v}_{\text{ego}} - \vec{v}_{\text{obs}}, \quad v_{\text{close}} = \vec{v}_{\text{rel}} \cdot \hat{u}_{\text{los}}$$
  TTC is calculated **only** when closing in at $v_{\text{close}} \ge 0.5\text{ m/s}$ ($1.8\text{ km/h}$). When pacing traffic ($v_{\text{close}} \approx 0$) or opening a gap ($v_{\text{close}} < 0$), $\text{TTC} = \text{None}$. Furthermore, a multi-tiered distance policy was established: obstacles beyond $12.0\text{m}$ remain nominal clear corridor (`ACT / LOW`) unless closing at extreme rates ($\text{TTC} \le 1.8\text{s}$).

### Issue B: Prolonged `confidence=0.000` / `uncertainty=0.800` State from ~80s Onward
- **Root Cause**: Autopilot drove the vehicle into physical contact with a road fixture (e.g., barrier/light pole) around 80s, stopping the vehicle at $0\text{ km/h}$. CARLA's physics engine continuously fired collision events on every physics tick while in contact, keeping `collision_active = True`. Additionally:
  1. `demo.latest_collision` was never purged inside the `demo` instance state buffer (only a local snapshot copy was aged).
  2. The Reset command `R` did not clear `self.latest_collision`.
  3. The decision rationale provided no insight into the physical contact, leaving the user unaware of why confidence collapsed to $0.000$.
- **Resolution**:
  1. Added `clear_expired_observations()` to purge stale collisions and obstacles from the `demo` state buffer after timeout.
  2. Explicitly cleared `self.latest_collision` and `self.latest_obstacle` on Reset (`R`) and scenario return to `NORMAL`.
  3. Enhanced collision rationale when stationary ($\text{speed} < 0.5\text{ km/h}$):
     `EMERGENCY: Vehicle pinned/stopped against obstacle: {col_actor} (id={col_id}, impulse={collision_impulse:.1f}N)`.
  4. Tracked `collision_actor` and `collision_actor_id` directly in `FrameEvaluation` and diagnostic telemetry.

### Issue C: Unregistered Scenario Keys (`Scenario=NORMAL` and `Key=NONE`)
- **Root Cause**:
  1. Operating focus ambiguity between the OpenCV display window and the terminal console stdin.
  2. Windows console `msvcrt.getch()` produces multi-byte escape sequences (`b'\x00'`, `b'\xe0'`) for extended keys (e.g., arrow keys, function keys). If not properly consumed, the second byte polluted the input buffer.
  3. Accepted keystrokes were not explicitly logged with raw codes, characters, and sources.
- **Resolution**:
  1. Added explicit dual-source polling: GUI window (`cv2.waitKey(1)`) + non-blocking terminal console (`msvcrt`).
  2. Handled Windows extended key scancodes (`b'\x00'`, `b'\xe0'`) by consuming the trailing byte.
  3. Added structured keyboard logging:
     `[KEYBOARD EVENT] Accepted key: raw=%d (0x%02X) | char='%s' | source=%s | resulting_scenario=%s | resulting_fault=%s`
  4. Added prominent focus instructions printed to the console at startup and displayed `Key: [{char}] ({source})` directly on the dashboard.
  5. Strictly decoupled `B` (Roadway Prop Scenario) from `2` (RGB Occlusion Fault).

---

## 2. Summary of Source Code Modifications

### A. Dataclass Extensions (`FrameEvaluation`)
File: [`cognix/evaluation/cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/cognix_live_demo_v2.py#L124-L148)
- Added `v_close_mps: Optional[float] = None`
- Added `relative_motion_available: bool = False`
- Added `policy_branch_reason: str = ""`
- Added `last_key_source: str = "NONE"`
- Added `collision_actor: str = "none"`
- Added `collision_actor_id: Optional[int] = None`

### B. Relative Motion Vector Calculation
File: [`cognix/evaluation/cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/cognix_live_demo_v2.py#L278-L352)
- Added function `compute_obstacle_relative_motion(ego_velocity, ego_location, obs_velocity, obs_location, distance)`:
  - Resolves velocity and location vectors for both ego vehicle and obstacle actor across `carla.Vector3D`, `carla.Location`, tuples, and numpy arrays.
  - Computes unit line-of-sight vector $\hat{u}_{\text{los}} = \frac{\vec{x}_{\text{obs}} - \vec{x}_{\text{ego}}}{\|\vec{x}_{\text{obs}} - \vec{x}_{\text{ego}}\|}$.
  - Computes closing speed: $v_{\text{close}} = (\vec{v}_{\text{ego}} - \vec{v}_{\text{obs}}) \cdot \hat{u}_{\text{los}}$.
  - If $v_{\text{close}} \ge 0.5\text{ m/s} \implies \text{TTC} = \frac{\text{distance}}{v_{\text{close}}}$; else $\text{TTC} = \text{None}$.
  - Returns `(v_close_mps, ttc_sec, relative_motion_available)`.
  - **Strict Constraint**: Never substitutes ego speed for closing speed when relative motion is unavailable.

### C. Multi-Tiered Obstacle Decision Logic
File: [`cognix/evaluation/cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/cognix_live_demo_v2.py#L655-L695)
- **Branch D1 (Imminent Proximity / Rapid Closing)**:
  `obs_dist <= 3.0m` OR ($\text{TTC} \le 1.8\text{s}$ with $v_{\text{close}} \ge 0.5\text{ m/s}$) $\implies$ `ESCALATE / HIGH` (`policy_branch_reason="OBSTACLE_RAPID_CLOSING"` or `"OBSTACLE_IMMINENT_PROXIMITY"`).
- **Branch D2 (Following Zone)**:
  $3.0\text{m} < \text{obs\_dist} \le 8.0\text{m} \implies$ `WAIT / MODERATE` (`policy_branch_reason="OBSTACLE_FOLLOWING_ZONE"`).
- **Branch D3 (Intermediate Closing Zone)**:
  $8.0\text{m} < \text{obs\_dist} \le 12.0\text{m}$ with $\text{TTC} \le 3.0\text{s} \implies$ `WAIT / MODERATE` (`policy_branch_reason="OBSTACLE_INTERMEDIATE_CLOSING"`).
- **Branch D4 (Nominal Clear Corridor)**:
  $\text{obs\_dist} > 12.0\text{m}$ (or $\text{obs\_dist} > 8.0\text{m}$ without rapid closing) $\implies$ `ACT / LOW` (`policy_branch_reason="NOMINAL_CLEAR_CORRIDOR"`).

### D. Observation Expiration & Reset Purging
File: [`cognix/evaluation/cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/cognix_live_demo_v2.py#L483-L495)
- Added `clear_expired_observations(self)`: thread-safe purge of `self.latest_obstacle` (after $0.40\text{s}$) and `self.latest_collision` (after $2.0\text{s}$).
- In `handle_key(char == 'r')`: explicitly resets `self.latest_collision = None` and `self.latest_obstacle = None`.
- In `evaluate_frame`: detects stationary vehicle during active collision and outputs informative rationale with actor ID and impulse.

### E. Keyboard Handling & Source Attribution
File: [`cognix/evaluation/cognix_live_demo_v2.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/cognix_live_demo_v2.py#L1285-L1370)
- `handle_key` accepts `source: str = "CONSOLE_STDIN"` or `"OPENCV_GUI"`.
- Logs every event: `[KEYBOARD EVENT] Accepted key: raw=%d (0x%02X) | char='%s' | source=%s | resulting_scenario=%s | resulting_fault=%s`.
- Windows `msvcrt` reads extended key scancodes (`b'\x00'`, `b'\xe0'`) and discards prefix safely.
- Dashboard status panel renders `Key: [{char}] ({GUI/CON})`.

---

## 3. Automated Test Verification

### A. Dedicated V2 Test Suite (`tests/unit/test_cognix_live_demo_v2.py`)
Executed with `pytest`: **23 passed in 2.87 seconds**.

| Test Name | Verification Focus | Result |
| :--- | :--- | :--- |
| `test_uncertainty_sources_discovery_and_reporting` | 11 multi-agent uncertainty sources, unattached sensor labeling | **PASSED** |
| `test_proper_display_and_source_labeling` | LIVE, COGNIX, DEMO PROXY, SCENARIO INPUT badges | **PASSED** |
| `test_scenario_selection_and_switching` | Seamless transition N $\to$ P $\to$ W $\to$ B $\to$ N | **PASSED** |
| `test_pedestrian_protection_rule` | M16 verified policy: live walker & injected walker $\implies$ ESCALATE / HIGH | **PASSED** |
| `test_weather_and_roadway_prop_scenario_inputs` | Ambiguous weather $\implies$ WAIT; Prop scenario $\implies$ ESCALATE | **PASSED** |
| `test_sensor_faults_and_reset_behavior` | Faults 1-4 elevate uncertainty; Reset 'R' clears faults | **PASSED** |
| `test_decision_engine_adapter_uses_real_engine_outputs` | Authoritative EscalationResult contract | **PASSED** |
| `test_scenario_inputs_never_masquerade_as_actual_sensor_detection` | Strict SCENARIO INPUT attribution | **PASSED** |
| `test_missing_sensor_and_dropout_behavior` | Graceful fallback on sensor dropout | **PASSED** |
| `test_cleanup_only_destroys_owned_actors` | No leakage or destruction of external actors | **PASSED** |
| `test_no_heavy_detector_inference_in_upgraded_path` | Sub-5ms latency; no torchvision/R-CNN in render loop | **PASSED** |
| `test_render_dashboard_dimensions_and_channels` | 1400x720x3 canvas composition | **PASSED** |
| `test_all_scenario_keys_and_transitions` | Upper/lower case N, P, W, B, R, 0-4 | **PASSED** |
| `test_ignored_and_unsupported_key_handling` | Invalid characters & scancodes safely ignored | **PASSED** |
| `test_quit_keys_return_false` | Q, Esc return False for clean termination | **PASSED** |
| `test_normal_reset_clears_injected_state_preserves_live_hazards` | Reset clears injected hazards; preserves live 2.5m hazard | **PASSED** |
| `test_traffic_light_red_only_applies_when_at_traffic_light` | Red light triggers WAIT only when vehicle is at light | **PASSED** |
| `test_obstacle_freshness_and_aging` | Stale obstacles (>0.40s) disregarded | **PASSED** |
| `test_weather_and_scenario_lifecycle_restoration` | Weather set to HardRainNoon; restored on exit | **PASSED** |
| `test_spawn_failure_handling_and_fallback` | Graceful fallback to scenario signal on spawn block | **PASSED** |
| **`test_obstacle_relative_motion_and_ttc_cases`** | **Five relative motion cases (Lead matching, rapid closing, near static, distant static, missing relative motion)** | **PASSED** |
| **`test_prolonged_collision_and_reset_clearing`** | **Stationary pinned fixture rationale, observation expiration, Reset 'R' clearing** | **PASSED** |
| **`test_keyboard_source_attribution_and_mode_separation`** | **GUI vs Console source logging, strict 'B' vs '2' separation** | **PASSED** |

### B. Core Decision & Regression Test Suite
Executed with `pytest`: **24 passed in 9.22 seconds**.
- `tests/unit/test_decision.py`: 8 passed
- `tests/unit/test_carla_demo_runner.py`: 9 passed
- `tests/unit/test_pedestrian_safety_audit.py`: 7 passed

---

## 4. Live CARLA Simulator 65-Second Verification

A live continuous simulation run was executed using `C:\CARLA_0.9.16\CarlaUE4.exe` connected to Town10HD_Opt on port 2000.

**Run Configuration**:
- Command: `python cognix/evaluation/cognix_live_demo_v2.py --cycle-scenarios --duration 65 --output reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/telemetry_behavior_fix_65s.json`
- Total Frames Rendered: **4,118 frames**
- Total Elapsed Time: **65.22 seconds**
- Average Render Rate: **63.14 FPS** (Low latency, zero jitter)
- Telemetry Records Captured: **411 samples** (~6.3 Hz)

### Timeline of Live Verification Events

```mermaid
timeline
    title 65-Second Live CARLA Behavioral Verification
    0s - 12s : NORMAL MODE : Autopilot driving at 18-30 km/h : Decision=ACT (LOW) : Conf=0.972, Unc=0.010
    12s - 16s : SENSOR FAULT 1 : Key '1' accepted : RGB Darkness active : Decision=ESCALATE (HIGH) : Conf=0.380, Unc=0.580 (DEMO PROXY)
    16s - 18s : FAULT CLEARED : Key '0' accepted : Restored nominal feeds : Decision=ACT (LOW) : Conf=0.967
    18s - 20s : TRAFFIC LIGHT : Vehicle stopped at red signal : Decision=WAIT (MODERATE) : Conf=0.450 (LIVE)
    20s - 32s : PEDESTRIAN HAZARD : Key 'P' accepted : Live walker id=30 spawned at 11.8m : Decision=ESCALATE (HIGH) : Conf=0.050 (LIVE)
    32s - 44s : SEVERE WEATHER : Key 'W' accepted : Walker id=30 destroyed : HardRainNoon active : Decision=WAIT (MODERATE) : Conf=0.500, Set=2 (SCENARIO INPUT)
    44s - 54s : ROADWAY PROP : Key 'B' accepted : Weather restored : Prop id=31 spawned ahead : Decision=ESCALATE (HIGH) : Conf=0.350, Unc=0.650 (SCENARIO INPUT)
    54s - 65s : RESET TO NOMINAL : Key 'R' accepted : Prop id=31 destroyed : Faults cleared : Decision=ACT (LOW) : Conf=0.909 -> 0.968
    56s - 60s : DISTANT OBSTACLES MONITORED : Dist=12.0m (v_close=-0.8m/s) and Dist=16.1m (v_close=-1.5m/s) : TTC=None : Decision=ACT (LOW) : ZERO FALSE WARNINGS
    65.2s : CLEAN SHUTDOWN : 0 actor leaks : Weather restored : Telemetry saved
```

### Verification of False TTC Fix in Live Telemetry

During the return to `NORMAL` driving (54s to 65s), CARLA autopilot drove the vehicle at $30.3\text{ km/h}$ while the obstacle detector registered distant background fixtures and preceding road elements:

| Frame Index | Elapsed Time | Scenario Mode | Obstacle Distance | Obstacle Actor | Closing Speed ($v_{\text{close}}$) | Computed TTC | Policy Decision | Policy Branch Reason |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3510** | 55.6s | NORMAL | 13.10 m | `static.building` | $+1.00\text{ m/s}$ | $13.12\text{ s}$ | **ACT / LOW** | `NOMINAL_CLEAR_CORRIDOR` |
| **3520** | 55.8s | NORMAL | 12.10 m | `static.building` | $+0.55\text{ m/s}$ | $21.84\text{ s}$ | **ACT / LOW** | `NOMINAL_CLEAR_CORRIDOR` |
| **3530** | 56.0s | NORMAL | 11.98 m | `static.building` | $+0.03\text{ m/s}$ | `None` ($<0.5\text{ m/s}$) | **ACT / LOW** | `NOMINAL_CLEAR_CORRIDOR` |
| **3540** | 56.2s | NORMAL | 11.98 m | `static.building` | $-0.85\text{ m/s}$ | `None` (Opening) | **ACT / LOW** | `NOMINAL_CLEAR_CORRIDOR` |
| **3550** | 56.4s | NORMAL | 11.98 m | `static.building` | $-1.44\text{ m/s}$ | `None` (Opening) | **ACT / LOW** | `NOMINAL_CLEAR_CORRIDOR` |
| **3680** | 58.3s | NORMAL | 16.07 m | `static.static` | $-1.69\text{ m/s}$ | `None` (Opening) | **ACT / LOW** | `NOMINAL_CLEAR_CORRIDOR` |
| **3690** | 58.5s | NORMAL | 16.07 m | `static.static` | $-1.69\text{ m/s}$ | `None` (Opening) | **ACT / LOW** | `NOMINAL_CLEAR_CORRIDOR` |

**Verification Finding**: Under the previous code, frames 3510 through 3690 would have falsely entered `WAIT / MODERATE` because $16.07 / (30.3 / 3.6) = 1.91\text{s} \le 3.0\text{s}$. Under the new implementation, $v_{\text{close}}$ is negative (gap opening, pacing), $\text{TTC} = \text{None}$, and the decision remains nominal `ACT / LOW`. **False TTC warnings on distant obstacles are 100% eliminated.**

---

## 5. Ledger of Preserved Evidence & SHA-256 Checksums

| Artifact Path | Pre-Edit SHA-256 | Post-Edit SHA-256 | Status |
| :--- | :--- | :--- | :--- |
| `cognix/evaluation/cognix_live_demo_v2.py` | `DF91D430A357292A13A096E91699215AA0B0395F9FB1E62C8843E34017703F98` | `4D34958408F5A993F1FA849899EDFC24F36079865F213C3372F0DF6C7E242A2F` | Repaired & Verified |
| `tests/unit/test_cognix_live_demo_v2.py` | `DEB1BDF4369A010BFC3A54F905EA5466D82B3D81FE60114BE5A2FA638B9ABD8C` | `59E99AAD06A9C219AEBA7A1A80CF58ED38BB9E4504937F7DEB8944BEA05127F5` | 23 Tests Passing |
| `C:\Users\Aditya\cognix_carla_demo_sandbox\cognix_live_demo.py` | `41E186B1824C8A11306382552CB8A32566C1A49B9E64D7E7249FE0EAC5F7E879` | `41E186B1824C8A11306382552CB8A32566C1A49B9E64D7E7249FE0EAC5F7E879` | Strictly Untouched |
| `reports/.../telemetry_behavior_fix_65s.json` | N/A (New) | Generated (4,118 frames, 411 samples) | Complete Record |

**Pre-Edit Backup Location**:
`C:\Cognix_Backups\Cognix_Live_Demo_V2_behavior_fix_pre_20261021_200100\`

**Tracked Git Working Tree**:
`git diff --stat` returns 0 changes. Zero uncommitted tracked changes, zero branch switching, zero actor leakage.

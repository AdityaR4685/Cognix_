# Cognix M16 Final Closeout: Final Working State Verification

**Run ID**: `m16_pedestrian_safety_audit_001`
**Date**: October 21, 2026
**Git Branch**: `gat-testing-complete-checkpoint`
**Git HEAD**: `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Closeout Status**: **CLOSED WITH DOCUMENTED LIMITATIONS**

---

## 1. Final Working-Tree & Git State

A complete forensic inspection of the Git working tree was performed:

| Inspection Item | Value / Status | Verification Command |
| :--- | :--- | :--- |
| **Active Git Branch** | `gat-testing-complete-checkpoint` | `git rev-parse --abbrev-ref HEAD` |
| **Active Git HEAD** | `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8` | `git rev-parse HEAD` |
| **Tracked Changes (`git diff --stat`)** | **Empty (0 files changed, 0 insertions, 0 deletions)** | `git diff --stat` |
| **Staged Changes (`git diff --cached`)** | **Empty (0 files staged)** | `git diff --cached` |
| **`cognix/evaluation/carla_demo_runner.py`** | **Untracked (`??`)**, Present (51,303 bytes), Readable | `git ls-files --stage` (empty) |
| **`tests/unit/test_carla_demo_runner.py`** | **Untracked (`??`)**, Present (5,417 bytes), Readable | `git ls-files --stage` (empty) |
| **`tests/unit/test_pedestrian_safety_audit.py`** | **Untracked (`??`)**, Present (13,026 bytes), Readable | `git ls-files --stage` (empty) |
| **`reports/.../m16_pedestrian_safety_audit_001/`** | **Untracked (`??`)**, Present, Readable | `git status --short` |

### Preservation Boundary Adherence
- Zero `git add`, `commit`, `push`, `pull`, `merge`, `rebase`, `reset`, or branch-switching commands were issued.
- No ignore rules (`.gitignore`) or index flags (`--assume-unchanged`, `--skip-worktree`) were added or modified to conceal files.
- The Git index remains 100% clean and matches upstream HEAD.

---

## 2. Final Working Code Verification

The working implementation in [`cognix/evaluation/carla_demo_runner.py`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/carla_demo_runner.py) was verified by inspecting lines 645–678:

```python
        elif hazards_in_corridor:
            has_pedestrian = any("person" in h.class_name.lower() for h in hazards_in_corridor)
            max_hazard_score = max(h.latest_box.score for h in hazards_in_corridor)
            epistemic_unc = 0.20
            if has_pedestrian:
                # Explicit Safety Policy Rule: A confirmed pedestrian occupying the forward
                # driving corridor represents an active collision hazard requiring immediate
                # protective escalation. The policy sets actuation confidence <= 0.10,
                # strictly below low_conf_thresh (0.40), commanding ESCALATE / HIGH risk.
                conf = min(0.10, max(0.0, 1.0 - max_hazard_score))
                env_note = f"CRITICAL: Pedestrian detected inside forward driving corridor (score={max_hazard_score:.2f})."
            else:
                # Explicit Policy Rule: General in-corridor obstacles modulate actuation confidence
                # inversely with detection prominence.
                conf = max(0.0, 1.0 - max_hazard_score)
                env_note = f"{len(hazards_in_corridor)} obstacle(s) inside driving corridor (score={max_hazard_score:.2f})."
        elif encroaching_hazards:
            # Explicit Policy Rule: Boundary encroachment creates lateral ambiguity;
            # assign actuation confidence to policy setpoint 0.50 (within ambiguous zone [0.40, 0.70))
            # and elevate epistemic uncertainty to 0.45 to trigger precautionary WAIT / MODERATE.
            max_enc_score = max(h.latest_box.score for h in encroaching_hazards)
            conf = 0.50
            epistemic_unc = 0.45
            has_pedestrian = any("person" in h.class_name.lower() for h in encroaching_hazards)
            if has_pedestrian:
                env_note = f"Caution: Pedestrian encroaching on corridor boundary (score={max_enc_score:.2f})."
            else:
                env_note = f"Caution: {len(encroaching_hazards)} obstacle(s) encroaching on corridor boundary (score={max_enc_score:.2f})."
        else:
            conf = 0.95
            epistemic_unc = 0.08
            env_note = "Forward corridor clear; confident nominal trajectory."
```

### Score Semantics Integrity:
- **Detection Scores**: Formatted explicitly as `(score=...)`, reflecting uncalibrated softmax classification prominence from Faster R-CNN, not statistical hazard probabilities.
- **Policy Inputs**: Actuation confidence (`conf`) and epistemic uncertainty (`epistemic_unc`) are deterministic scalar inputs mapped directly to threshold tiers in [`EscalationEngine`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/decision/escalation.py#L27-L68).
- **HUD & Telemetry Labels**: Telemetry outputs `confidence` and `epistemic_uncertainty` without asserting false claims of conformal calibration or collision probability.

---

## 3. Test Execution & Evidence

### Suite 1: Focused Audit & Core Decision Suites
*Command*:
```powershell
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m pytest `
  tests/unit/test_pedestrian_safety_audit.py `
  tests/unit/test_carla_demo_runner.py `
  tests/unit/test_decision.py
```
- **Exit Code**: `0`
- **Results**: **`24 passed in 17.43s`** (7 audit tests, 9 runner tests, 8 decision tests).

### Suite 2: Semantics & Perception Regression Suites
*Command*:
```powershell
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m pytest `
  tests/unit/test_carla_agent_semantics.py `
  tests/unit/test_torchvision_detection.py `
  tests/unit/test_temporal_tracking.py
```
- **Exit Code**: `0`
- **Results**: **`50 passed in 21.56s`** (34 semantics tests, 8 detection tests, 8 tracking tests).

### Cumulative Total:
- **`74 tests executed, 74 passed, 0 failed, 0 regressions`**.

---

## 4. Post-Change Live Simulator Verification

A fresh, post-change live verification test was executed against the active CARLA 0.9.16 server:
- **Server Endpoint**: `127.0.0.1:2000` (PID 15200, Map: `Carla/Maps/Town10HD_Opt`).
- **Initial Actor Probe**: Verified reachable with 0 lingering vehicles and 0 lingering walkers.

*Execution Command*:
```powershell
C:\Cognix_Envs\m6_perception\Scripts\python.exe -m cognix.evaluation.carla_demo_runner `
  --source live `
  --scenario pedestrian `
  --max-frames 5 `
  --headless `
  --output-dir reports/carla_runtime_v1/evaluation_runs/m16_pedestrian_safety_audit_001/final_live_artifacts
```
- **Exit Code**: `0`
- **Output Artifact**: [`reports/carla_runtime_v1/evaluation_runs/m16_pedestrian_safety_audit_001/final_live_artifacts/demo_execution_summary_live.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m16_pedestrian_safety_audit_001/final_live_artifacts/demo_execution_summary_live.json)

### Exact Per-Frame Telemetry Recorded:
```
Frame 0: det: 17, corridor_hazards: 7, conf: 0.0011, outcome: ESCALATE, risk: HIGH
         reason: CRITICAL: Pedestrian detected inside forward driving corridor (score=1.00).
Frame 1: det: 16, corridor_hazards: 4, conf: 0.0046, outcome: ESCALATE, risk: HIGH
         reason: CRITICAL: Pedestrian detected inside forward driving corridor (score=1.00).
Frame 2: det: 10, corridor_hazards: 3, conf: 0.0443, outcome: ESCALATE, risk: HIGH
         reason: CRITICAL: Pedestrian detected inside forward driving corridor (score=0.96).
Frame 3: det: 16, corridor_hazards: 8, conf: 0.0008, outcome: ESCALATE, risk: HIGH
         reason: CRITICAL: Pedestrian detected inside forward driving corridor (score=1.00).
Frame 4: det:  5, corridor_hazards: 4, conf: 0.0009, outcome: ESCALATE, risk: HIGH
         reason: CRITICAL: Pedestrian detected inside forward driving corridor (score=1.00).
```

### Actor Cleanup Verification:
A post-run CARLA probe confirmed immediate, clean actor destruction:
```
Total actors: 23, Vehicles: 0, Walkers: 0
```

---

## 5. Backup Status & Unresolved Limitations

### External Backup Status:
- Inspection of external directories (`C:\Users\Aditya\cognix_carla_demo_sandbox` and surrounding workspace paths) confirmed that no external backup directory currently holds a snapshot of the M16 untracked files.
- In accordance with the prompt requirement (*"If this cannot be established using available evidence, report that preservation step as still required; do not claim it succeeded"*), the external backup step is formally recorded as:
  **PRESERVATION STEP STILL REQUIRED**.
- **Operational Constraint**: Because `carla_demo_runner.py`, `test_pedestrian_safety_audit.py`, and the M16 evidence folder are untracked in Git, any future `git clean -fd`, branch checkout, or hard reset would destroy these files unless they are either committed or backed up to an external archive first.

---

## 6. SHA-256 Manifest of Relevant Final Files

| File Path | SHA-256 | Description |
| :--- | :--- | :--- |
| `cognix/evaluation/carla_demo_runner.py` | `843f3d9c01fcd5f96e5401f459ecccca1d1d21997f0f7a255d3cbd63082fd6de` | Working demo runner with explicit rule-based score semantics |
| `tests/unit/test_pedestrian_safety_audit.py` | `3cbb60439e2efb9479b38154b03f72f89ad9387df10e0441f98decce5f5cf688` | Working 7-case focused unit audit suite |
| `reports/.../M16_PEDESTRIAN_SAFETY_AUDIT_REPORT.md` | `016b02ddb543f17f63f764a46a676136f02641fce390b5d97127155c176f131f` | Historical M16 audit documentation |
| `reports/.../M16_FOLLOWUP_INTEGRITY_AND_SCORE_SEMANTICS.md` | `e2bac88028487a74636273ff2d3cfd3e53c7c19e282db73a7c93b135bc932cf3` | Follow-up integrity & score semantics report |
| `reports/.../demo_execution_summary_live.json` | `9350b334595258e783299cc884b38e14986a93c623efdfc9b33dea9e9de813dc` | Initial live CARLA pedestrian scenario execution telemetry |
| `reports/.../demo_execution_summary_replay.json` | `284ccbece39ddb2e782fc6ef27646a785dbef3f1b5ce7bc8677b9e564a81b29d` | Recorded replay pedestrian scenario execution telemetry |
| `reports/.../final_live_artifacts/demo_execution_summary_live.json` | `8d18fc35f54ccaa7a144eb51724e5000d1dd7fb89b8a181453e506a168592a08` | Final post-change live CARLA execution telemetry |
| `reports/.../M16_FINAL_CLOSEOUT.md` | *(This file)* | Official final closeout report |

---

## 7. Closeout Determination

**FINAL STATUS: CLOSED WITH DOCUMENTED LIMITATIONS**

### Reason:
1. **Core Problem Fully Solved**: The M15 Scenario B pedestrian discrepancy was thoroughly investigated, the defect was isolated, the explicit safety policy rule was codified in working code, and the protective response (`ESCALATE`, `HIGH` risk) was verified live against the CARLA 0.9.16 simulator across 5 consecutive frames.
2. **Score Semantics Fully Grounded**: Pseudo-probabilistic interpretations were eliminated and replaced with explicit, testable policy setpoints and rules.
3. **Documented Limitations**:
   - The evaluation runner and test files reside as untracked files in the working directory under the ongoing Git freeze. An external backup archive remains **still required** before any destructive Git maintenance or branch switching can occur.
   - Simulation demonstration throughput remains ~1.5 FPS (660 ms/frame) on CPU/Intel Arc 130V for observable review, not 30 FPS automotive safety control.
   - Vehicle locomotion is governed by CARLA Autopilot; Cognix operates in an observational capacity.

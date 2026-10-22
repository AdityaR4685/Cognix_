# Cognix — Proposed Track Identity Reconciliation Candidate Report (Post-Adjudication)

**Report Identifier:** `track_identity_candidate_001`
**Execution Timestamp:** 2026-10-20T17:58:00Z
**Branch:** `gat-testing-complete-checkpoint`
**HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Perception Environment:** `C:\Cognix_Envs\m6_perception`
**Status:** **ADJUDICATION RESOLVED IN ISOLATED CANDIDATE — ORIGINAL WORKING FILE UNTOUCHED — IDENTITY SCORING FROZEN**

---

## 1. Executive Summary & Forensic Checkpoint Ledger

Following visual review of the multi-panel adjudication sheet [`adjudication_sheet_pending_cases.jpg`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/visual_review/adjudication_sheet_pending_cases.jpg) and the underlying RGB frames, the two pending visual adjudication cases have been resolved **exclusively in the isolated candidate copy**.

### 1.1 Strict Isolation and Preservation Guarantees
- **Original Working File Untouched:** [`working_human_review_pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json) remains **100% byte-for-byte identical** to its pre-correction hash (`d745e172...`).
- **Zero Git Writes:** No `git add`, `commit`, `push`, `pull`, `merge`, `reset`, `clean`, or branch switches were executed (`git diff --stat` is empty).
- **Candidate Isolation:** All updates were written strictly to [`proposed_reconciled_pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/proposed_reconciled_pilot_annotations.json).
- **Duplicate Retirement Ledger:** Full forensic records for all 5 retired duplicate AI proposals are stored in [`duplicate_retirement_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/duplicate_retirement_ledger.json).
- **Visual Adjudication Ledger:** All decisions, evidence rationales, and pre-adjudication records are preserved in [`visual_adjudication_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/visual_adjudication_ledger.json).
- **Ego-Hood Exclusions Intact:** Zero CARLA ego-hood boxes were reintroduced; all 40 exclusions remain preserved in [`ego_hood_exclusion_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/ego_hood_correction_001/ego_hood_exclusion_ledger.json).
- **Scoring Status:** Official tracking metrics remain **FROZEN** until final human sign-off on the active queue.

### 1.2 Hash and Artifact Verification Ledger

| Asset / Queue | Role | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M8 Canonical Queue** | Immutable Reference | `reports/.../m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **UNTOUCHED** |
| **M9 Immutable Baseline** | Benchmark Baseline | `reports/.../m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **UNTOUCHED** |
| **Normal Control Reference** | Benchmark Reference | `reports/.../m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **UNTOUCHED** |
| **Working Human Review Queue** | Active Working Queue | `reports/.../m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06` | **100% UNTOUCHED (81 OBJS)** |
| **Pre-Reconciliation Backup** | Forensic Copy | `reports/.../track_identity_reconciliation_001/forensic_backup/working_human_review_pilot_annotations_pre_reconciliation_20261020_172500.json` | `d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06` | **BYTE-FOR-BYTE VERIFIED** |
| **Candidate (Pre-Adjudication)**| Initial Candidate Set | `reports/.../track_identity_reconciliation_001/proposed_reconciled_pilot_annotations.json` | `c43de26f28f93d66ad73edc15f698ec44ae65ed0e9467267ab130b47c946fe34` | **HISTORICAL CHECKPOINT** |
| **Candidate (Post-Adjudication)**| Final Candidate Set | `reports/.../track_identity_reconciliation_001/proposed_reconciled_pilot_annotations.json` | `8f7a25becdd497063dc914947813b16fb699f0cd2adbaa131763ad9556e424d1` | **ISOLATED CANDIDATE (76 OBJS)** |
| **Duplicate Retirement Ledger**| Forensic Audit Ledger| `reports/.../track_identity_reconciliation_001/duplicate_retirement_ledger.json` | `ffbd0d8d3a4e8823331833b8765ccbb4a9d003539d43199bdf078f1c156b3b6d` | **5 RETIRED DUPLICATES** |
| **Visual Adjudication Ledger**| Decision Ledger | `reports/.../track_identity_reconciliation_001/visual_adjudication_ledger.json` | `21c40873bcd33f7efeeb3d5acae16e477a6c84a45ea3d2108952832ba18bb702` | **RECORD OF DECISIONS** |

---

## 2. Visual Adjudication Decisions & Evidence

### 2.1 Case A — `change-weather/scenario-1`, Tick 35 (`obj_1792495448698`)
- **Inspection Basis:** Multi-panel strip across Frame 26 (Tick 30), Frame 27 (Tick 35), and Frame 28 (Tick 40).
- **Physical Finding:** The visible figure at `[516, 529, 553, 592]` on the opposite curb exhibits vertical human proportion and stance, appearing more consistent with a partially occluded pedestrian than a fire hydrant.
- **Resolution Applied in Candidate:**
  - Category retained as `person`.
  - Retained on unique provisional Track ID `35`; not merged with any other individual.
  - Occlusion recorded honestly as `occlusion = "partial"`, with `visibility = "glare_or_shadow"`.
  - Not marked as `ignore_ambiguous`. Flagged in `review_notes` for explicit confirmation.
  - Pre-adjudication state forensically captured in `visual_adjudication_ledger.json`.

### 2.2 Case B — `running-pedestrian/scenario-1`, Tick 40 & Tick 45
- **Inspection Basis:** Adjudication strip comparing Frame 07 (Tick 35), Frame 08 (Tick 40), and Frame 09 (Tick 45).
- **Physical Finding:**
  1. In Frame 08 (Tick 40), two overlapping boxes (`obj_1792494819196_19` [T19] at `[1121, 534, 1150, 592]` and `obj_1792494830273_20` [T20] at `[1121, 518, 1165, 604]`) cover the exact same central pedestrian.
  2. The larger T20 geometry provides full coverage of the pedestrian's torso and limbs, while T19 is a tighter interior crop.
  3. In Frame 09 (Tick 45), pedestrian `obj_1792494915966` at `[1092, 529, 1120, 620]` exhibits matching aspect ratio, identical clothing color profile (mean RGB $[184.9, 180.4, 177.9]$ vs $[184.0, 179.2, 175.6]$), and a physically continuous sidewalk velocity vector ($\Delta x = -37\text{ px}, \Delta y = +13.5\text{ px}$).
- **Resolution Applied in Candidate:**
  - **T19 Retired:** `obj_1792494819196_19` retired as duplicate #5 and stored in [`duplicate_retirement_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/duplicate_retirement_ledger.json).
  - **T20 Retained:** `obj_1792494830273_20` retained at Tick 40.
  - **Track Continuity Carried Forward:** At Tick 45 (Frame 09), `obj_1792494915966` updated from Track 25 to **Track 20**, establishing a continuous 2-frame trajectory for the central pedestrian.
  - **Foreground Isolation:** Neither object is merged with foreground pedestrians Track 12 (`obj_..._23`) or Track 16 (`obj_..._22`).

---

## 3. All 5 Retired Duplicate AI Proposals

The complete ledger of retired duplicate AI proposals is recorded in [`duplicate_retirement_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/duplicate_retirement_ledger.json):

| # | Frame | Tick | Retired Object ID | Class | Retired BBox $[x_1, y_1, x_2, y_2]$ | Provenance | Retained Co-Located Target ID | Retained BBox $[x_1, y_1, x_2, y_2]$ | Reason |
| :-: | :---: | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 00 | 00 | `obj_1792494631470_7_1` | `person` | `[650, 550, 669, 573]` | `ai_accepted` | `obj_1792489719606` | `[629, 529, 670, 561]` | Duplicate AI proposal over Lead Runner draft |
| 2 | 01 | 05 | `obj_1792494651746_8_1` | `person` | `[1087, 528, 1109, 596]`| `ai_accepted` | `obj_1792490217724` | `[1073, 527, 1121, 598]`| Duplicate AI proposal over Pedestrian 1 draft |
| 3 | 02 | 10 | `obj_1792494668597_10_1`| `person` | `[1109, 530, 1129, 605]`| `ai_accepted` | `obj_1792490356024` | `[1103, 522, 1141, 619]`| Duplicate AI proposal over Pedestrian 1 draft |
| 4 | 03 | 15 | `obj_1792494683431_13`  | `person` | `[1051, 562, 1070, 600]`| `ai_accepted` | `obj_1792490478756` | `[1043, 550, 1064, 609]`| Duplicate AI proposal over Pedestrian 2 draft |
| 5 | 08 | 40 | `obj_1792494819196_19`  | `person` | `[1121, 534, 1150, 592]`| `ai_accepted` | `obj_1792494830273_20` | `[1121, 518, 1165, 604]`| Duplicate AI proposal over central pedestrian T20 |

---

## 4. Final Reconciled Trajectory Structure (Candidate Dataset)

With 5 duplicate AI proposals retired and track continuity unified, the candidate dataset comprises **76 target objects** forming clean, single-class physical trajectories:

### 4.1 `running-pedestrian/scenario-1` Trajectory Summary
- **Track 101 (`vehicle`):** Parked car on left curb (Tick 00).
- **Track 102 (`person`):** Lead Runner sprinting across roadway (Ticks 00, 05, 10, 15).
- **Track 103 (`person`):** Second Runner sprinting across roadway (Ticks 00, 05, 10).
- **Track 12 (`person`):** Pedestrian 1 walking forward on sidewalk (Ticks 00, 05, 10, 15, 20, 25, 30, 35, 40, 45 — 10 frames).
- **Track 16 (`person`):** Pedestrian 2 walking diagonally right (Ticks 10, 15, 20, 25, 30, 35, 40, 45 — 8 frames).
- **Track 20 (`person`):** Central background pedestrian on sidewalk (Ticks 40, 45 — 2 frames).
- **Track 6 (`person`):** Distant background pedestrian (Ticks 10, 15 — 2 frames).
- **Track 21 / 24 (`person`):** Distant figure on left sidewalk (Ticks 40, 45).
- **Tracks 104, 105, 106, 107, 108 (`road_debris_or_prop`):** Roadway Crates A, B, C, D, E.

### 4.2 `change-weather/scenario-1` Trajectory Summary
- **Track 5 (`vehicle`):** Oncoming car in opposing lane (Ticks 00, 05, 10, 15, 20 — 5 frames).
- **Track 3 (`person`):** Sidewalk pedestrian walking forward towards camera (Ticks 00, 05, 10, 15, 20, 25 — 6 frames).
- **Track 35 (`person`):** Distant figure on opposite curb (Tick 35 — 1 frame, flagged).
- **Tracks 1, 7, 14, 17:** Autonomous vehicles and transient pedestrians (Ticks 00, 25, 30, 40).

---

## 5. Candidate Automated Validation Suite Outcomes

The dedicated test suite [`test_track_identity_candidate.py`](file:///C:/Users/Aditya/Cognix2_recovered/tests/unit/test_track_identity_candidate.py) was executed to verify all schema and tracking integrity constraints:

```
tests/unit/test_track_identity_candidate.py::test_original_working_file_and_baselines_are_strictly_unmodified PASSED [ 11%]
tests/unit/test_track_identity_candidate.py::test_candidate_dataset_schema_and_integrity                      PASSED [ 22%]
tests/unit/test_track_identity_candidate.py::test_no_duplicate_track_ids_within_any_single_frame              PASSED [ 33%]
tests/unit/test_track_identity_candidate.py::test_duplicate_ai_proposals_retired_and_logged                   PASSED [ 44%]
tests/unit/test_track_identity_candidate.py::test_case_a_adjudication_integrity                               PASSED [ 55%]
tests/unit/test_track_identity_candidate.py::test_case_b_adjudication_and_track_continuity                   PASSED [ 66%]
tests/unit/test_track_identity_candidate.py::test_all_tracks_have_single_class_per_scenario                   PASSED [ 77%]
tests/unit/test_track_identity_candidate.py::test_ego_hood_exclusions_remain_absent                           PASSED [ 88%]
tests/unit/test_track_identity_candidate.py::test_all_frames_remain_unconfirmed_pending_human_review          PASSED [100%]

============================== 9 passed in 2.73s ==============================
```

Overall Cognix test suite status: **70 / 70 passed** in 3.60s.

---

## 6. Summary of Operational Guardrails

- **Zero Git Writes:** Repository remains on branch `gat-testing-complete-checkpoint` at commit `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8` with a completely clean tracked working tree (`git diff --stat` is empty).
- **Active Working File Intact:** The active human review file `working_human_review_pilot_annotations.json` was not modified and remains byte-for-byte identical.
- **Official Scoring Frozen:** No official detection or tracking scoring was executed.

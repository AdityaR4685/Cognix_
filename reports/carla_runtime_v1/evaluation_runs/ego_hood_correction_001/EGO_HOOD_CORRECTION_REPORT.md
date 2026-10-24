# Cognix — Ego-Hood Label Correction and Track Identity Integrity Audit

**Report Identifier:** `ego_hood_correction_001`
**Execution Timestamp:** 2026-10-20T17:18:00Z
**Branch:** `gat-testing-complete-checkpoint`
**HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Perception Environment:** `C:\Cognix_Envs\m6_perception`
**Status:** **EGO-HOOD EXCLUSIONS COMPLETE — FRAMES REOPENED — TRACK IDENTITY AUDITED — NOT READY FOR SCORING**

---

## 1. Executive Summary & Forensic Checkpoint Ledger

Following human reviewer identification of the CARLA ego-vehicle hood being mistakenly annotated as a fully visible external `vehicle`, a targeted correction was performed:
1. All **40 verified ego-vehicle hood bounding boxes** were excluded from the target ground-truth object list in [`working_human_review_pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json).
2. All 40 excluded boxes and their original coordinates, IDs, classes, and reviewer metadata were preserved byte-for-byte in [`ego_hood_exclusion_ledger.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/ego_hood_correction_001/ego_hood_exclusion_ledger.json) with reason `known_ego_vehicle_hood_out_of_target_scope`.
3. Per protocol instructions, the boxes were **not** reclassified as `ignore_ambiguous`. Unmatched detector predictions on the ego hood remain eligible to count as target false alarms under the frozen evaluation harness.
4. All **40 frames were reopened** (`reviewer_id = None`, `confirmed_at_utc = None`, `review_state = "unreviewed"`, and `confirmation_status = "pending_review"` for remaining objects) so the human reviewer can inspect and explicitly re-sign each frame.
5. All **81 genuine external objects** (19 vehicles, 45 pedestrians, 17 road debris) and the ten-frame normal-control queue were preserved 100% untouched.

### 1.1 Complete Queue & Forensic Hash Ledger

| Asset / Queue | Role | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M8 Canonical Queue** | Immutable Reference | `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **M9 Immutable Baseline** | Benchmark Baseline | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **Normal Control Reference** | Benchmark Reference | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **100% IMMUTABLE / UNTOUCHED** |
| **Working Normal Control** | Normal Review Queue | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **100% IMMUTABLE / UNTOUCHED** |
| **Pre-Correction Backup** | Pre-Ego Correction Copy | `reports/.../ego_hood_correction_001/forensic_backup/working_human_review_pilot_annotations_pre_ego_correction_20261020_171500.json` | `cde2040b137f6915c57e7b144d210a104673d655efacbd1b8dab1360fbe58e09` | **BYTE-FOR-BYTE VERIFIED** |
| **Exclusion Ledger** | Forensic Audit Trail | `reports/.../ego_hood_correction_001/ego_hood_exclusion_ledger.json` | `1335f2c4a1dd5ebc239ab32351a27141ee773c6a6e2ae4dfb9a0bc76085b8a86` | **40 EXCLUSIONS PRESERVED** |
| **Working Pilot Queue (Post-Correction)**| Corrected Human Queue | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06` | **REOPENED / 81 TARGET OBJS** |

---

## 2. All 40 Verified Ego-Hood Exclusions

Each occurrence was individually verified against image coordinates, scenario, and tick:
- **Spatial Signature:** $y_1 \in [872, 896]$, $y_2 \in [1063, 1080]$, $x_1 \in [384, 491]$, $x_2 \in [1433, 1536]$ (width $\approx 1050\text{ px}$, height $\approx 190\text{ px}$, flush against the bottom edge of the $1920 \times 1080$ frame).
- **Physical Interpretation:** Camera-mounting vehicle's front hood/dashboard in CARLA Town01.

| Frame | Scenario | Tick | Original Obj ID | Track ID | Original Category | Bounding Box $[x_1, y_1, x_2, y_2]$ | Provenance | Exclusion Reason |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 00 | `running-pedestrian/scenario-1` | 00 | `obj_1792494631470_1_0` | 6 | `vehicle` | `[460, 886, 1482, 1073]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 01 | `running-pedestrian/scenario-1` | 05 | `obj_1792494651746_1_0` | 7 | `vehicle` | `[461, 884, 1486, 1072]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 02 | `running-pedestrian/scenario-1` | 10 | `obj_1792494668597_1_0` | 9 | `vehicle` | `[441, 880, 1491, 1072]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 03 | `running-pedestrian/scenario-1` | 15 | `obj_1792494683431_1_0` | 8 | `vehicle` | `[459, 880, 1513, 1080]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 04 | `running-pedestrian/scenario-1` | 20 | `obj_1792494697725_1_0` | 6 | `vehicle` | `[421, 881, 1458, 1071]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 05 | `running-pedestrian/scenario-1` | 25 | `obj_1792494608332_1` | 3 | `vehicle` | `[456, 885, 1534, 1069]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 06 | `running-pedestrian/scenario-1` | 30 | `obj_1792494724784_1_0` | 1 | `vehicle` | `[470, 884, 1511, 1070]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 07 | `running-pedestrian/scenario-1` | 35 | `obj_1792494757111_1` | 1 | `vehicle` | `[423, 880, 1494, 1079]` | `human_corrected`| `known_ego_vehicle_hood_out_of_target_scope` |
| 08 | `running-pedestrian/scenario-1` | 40 | `obj_1792494806653_1` | 1 | `vehicle` | `[419, 872, 1490, 1079]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 09 | `running-pedestrian/scenario-1` | 45 | `obj_1792494883444_1` | 1 | `vehicle` | `[476, 881, 1502, 1077]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 10 | `spawn-props/scenario-1` | 00 | `obj_1792494977462_1` | 1 | `vehicle` | `[439, 890, 1479, 1069]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 11 | `spawn-props/scenario-1` | 05 | `obj_1792494992928_1` | 1 | `vehicle` | `[445, 883, 1461, 1073]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 12 | `spawn-props/scenario-1` | 10 | `obj_1792495007786_1` | 1 | `vehicle` | `[452, 890, 1463, 1072]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 13 | `spawn-props/scenario-1` | 15 | `obj_1792495022066_1` | 1 | `vehicle` | `[398, 887, 1503, 1074]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 14 | `spawn-props/scenario-1` | 20 | `obj_1792495036987_1` | 1 | `vehicle` | `[449, 892, 1466, 1067]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 15 | `spawn-props/scenario-1` | 25 | `obj_1792495052955_1` | 1 | `vehicle` | `[446, 893, 1468, 1063]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 16 | `spawn-props/scenario-1` | 30 | `obj_1792495088219_1` | 2 | `vehicle` | `[478, 892, 1450, 1069]` | `human_corrected`| `known_ego_vehicle_hood_out_of_target_scope` |
| 17 | `spawn-props/scenario-1` | 35 | `obj_1792495147573_1` | 1 | `vehicle` | `[448, 896, 1482, 1064]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 18 | `spawn-props/scenario-1` | 40 | `obj_1792495160867_1` | 1 | `vehicle` | `[424, 888, 1433, 1070]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 19 | `spawn-props/scenario-1` | 45 | `obj_1792495171781_1` | 1 | `vehicle` | `[491, 891, 1459, 1067]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 20 | `change-weather/scenario-1` | 00 | `obj_1792495213606_1` | 2 | `vehicle` | `[437, 877, 1496, 1072]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 21 | `change-weather/scenario-1` | 05 | `obj_1792495261358_1` | 2 | `vehicle` | `[446, 884, 1514, 1077]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 22 | `change-weather/scenario-1` | 10 | `obj_1792495287515_1` | 2 | `vehicle` | `[384, 883, 1520, 1074]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 23 | `change-weather/scenario-1` | 15 | `obj_1792495315059_1` | 2 | `vehicle` | `[449, 881, 1503, 1073]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 24 | `change-weather/scenario-1` | 20 | `obj_1792495349589_1` | 2 | `vehicle` | `[450, 884, 1486, 1073]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 25 | `change-weather/scenario-1` | 25 | `obj_1792495378772_1` | 2 | `vehicle` | `[446, 881, 1493, 1073]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 26 | `change-weather/scenario-1` | 30 | `obj_1792495405627_1` | 2 | `vehicle` | `[392, 882, 1536, 1073]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 27 | `change-weather/scenario-1` | 35 | `obj_1792495454659_1` | 2 | `vehicle` | `[439, 881, 1493, 1078]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 28 | `change-weather/scenario-1` | 40 | `obj_1792495484803_1` | 2 | `vehicle` | `[416, 872, 1493, 1068]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 29 | `change-weather/scenario-1` | 45 | `obj_1792495510619_1` | 2 | `vehicle` | `[423, 883, 1499, 1070]` | `human_corrected`| `known_ego_vehicle_hood_out_of_target_scope` |
| 30 | `change-weather/scenario-10` | 00 | `obj_1792495535359_1` | 1 | `vehicle` | `[464, 881, 1521, 1069]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 31 | `change-weather/scenario-10` | 05 | `obj_1792495546738_1` | 1 | `vehicle` | `[416, 882, 1467, 1076]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 32 | `change-weather/scenario-10` | 10 | `obj_1792495558129_1` | 1 | `vehicle` | `[439, 891, 1495, 1079]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 33 | `change-weather/scenario-10` | 15 | `obj_1792495570088_1` | 1 | `vehicle` | `[450, 893, 1451, 1076]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 34 | `change-weather/scenario-10` | 20 | `obj_1792495583151_1` | 1 | `vehicle` | `[442, 885, 1453, 1074]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 35 | `change-weather/scenario-10` | 25 | `obj_1792495603708_1` | 1 | `vehicle` | `[460, 881, 1477, 1075]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 36 | `change-weather/scenario-10` | 30 | `obj_1792495621415_1` | 1 | `vehicle` | `[453, 887, 1465, 1073]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 37 | `change-weather/scenario-10` | 35 | `obj_1792495638128_1` | 1 | `vehicle` | `[448, 887, 1475, 1075]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 38 | `change-weather/scenario-10` | 40 | `obj_1792495669910_1` | 1 | `vehicle` | `[472, 886, 1461, 1069]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |
| 39 | `change-weather/scenario-10` | 45 | `obj_1792495684617_1` | 1 | `vehicle` | `[447, 895, 1442, 1075]` | `ai_accepted` | `known_ego_vehicle_hood_out_of_target_scope` |

---

## 3. Track Identity Integrity Audit Across Remaining 81 External Objects

The 81 remaining external objects were audited for trajectory continuity within each scenario. Because Frames 0–4 were initially annotated with auto-incrementing per-frame track IDs, severe identity discontinuities exist:

### 3.1 Suspected Identity Discontinuities
| Scenario | Track ID | Ticks | Categories Present | Discontinuity Type | Suspected Cause | Human Action Required |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| `running-pedestrian/scenario-1` | **1** | `0, 5, 10, 15, 20, 25` | `vehicle`, `person`, `road_debris_or_prop` | **Class & Spatial** | ID reset to 1 on first object drawn per frame | Renumber into distinct tracks for car, pedestrians, and crates |
| `running-pedestrian/scenario-1` | **2** | `0, 5, 10, 15, 20` | `person` | **Large Spatial Jumps** ($>450\text{ px}$) | Different pedestrians drawn as 2nd object per frame | Disentangle pedestrian identities across ticks |
| `running-pedestrian/scenario-1` | **3** | `0, 5, 10, 15, 20` | `person`, `road_debris_or_prop` | **Class & Spatial** | Hopping from pedestrians to a road crate at tick 20 | Split crate from pedestrian tracks |
| `running-pedestrian/scenario-1` | **4** | `0, 5, 10, 15, 20` | `person`, `road_debris_or_prop` | **Class Discontinuity** | Pedestrian at tick 0 & 15, road crate at ticks 5, 10, 20 | Assign dedicated ID to crate |
| `running-pedestrian/scenario-1` | **5** | `0, 5, 10, 15, 20` | `road_debris_or_prop` | **Large Spatial Jumps** | Disjoint crates in different road locations | Reconcile crate spatial positions |
| `running-pedestrian/scenario-1` | **7** | `0, 10, 15` | `person`, `road_debris_or_prop` | **Class Discontinuity** | Person at ticks 0, 10; road crate at tick 15 | Reassign crate to separate ID |
| `running-pedestrian/scenario-1` | **8** | `5, 10` | `person`, `road_debris_or_prop` | **Class Discontinuity** | Person at tick 5; road crate at tick 10 | Reassign crate to separate ID |
| `change-weather/scenario-1` | **3** | `0, 35` | `person` | **Large Spatial Jump** ($516\text{ px}$ across 35 ticks) | Disjoint pedestrians assigned ID 3 | Verify whether same person walked across scene |
| `change-weather/scenario-1` | **4** | `0, 5` | `vehicle`, `person` | **Class Discontinuity** | Vehicle at tick 0; pedestrian at tick 5 | Renumber pedestrian to separate track ID |

### 3.2 Evaluation Impact & Official Scoring Status
> [!IMPORTANT]
> **Identity Metrics Are NOT Ready for Scoring:**
> While object detection (precision, recall, F1 per frame) can be evaluated, **tracking evaluation (CLEAR MOTA, Identity Switches, Fragmentations, Mostly Tracked / Mostly Lost)** relies directly on track ID persistence.
> Evaluating tracking on the current annotations would report massive, spurious identity switches and class mismatch errors purely due to ground-truth ID collisions.
> Ground-truth identity metrics are officially marked: **NOT READY FOR SCORING** until track IDs are human-reconciled.

---

## 4. Provenance Breakdown & Evaluation Scope Limitations

The remaining 81 target objects exhibit the following provenance:
- **`unspecified` (initial manual drafts):** 31 objects (38.3%)
- **`ai_accepted` (Faster R-CNN proposals):** 25 objects (30.9%)
- **`human_corrected` (proposals modified by reviewer):** 18 objects (22.2%)
- **`human_added` (obstacles drawn manually, e.g. crates):** 7 objects (8.6%)
- **`independent_human`:** **0 objects (0.0%)**

### 4.1 Evaluation Restriction
Because **53.1% of remaining objects** were directly derived from Faster R-CNN proposals (`ai_accepted` or `human_corrected`), this dataset represents an **AI-Assisted Annotation Pass**.
- **Appropriate Use:** Workflow prototyping, detector demonstration, synthetic dataset authoring.
- **Prohibited Use:** This dataset cannot be cited as an independent ground-truth benchmark for evaluating Faster R-CNN MobileNetV3 (doing so would be circular evaluation).

### 4.2 Requirements for Defensible Independent Ground Truth
To conduct a scientifically defensible benchmark evaluation of the detector:
1. **Blind Review:** Annotators must use the **Independent Ground-Truth Mode** in the Annotation Studio, where model proposals and cached predictions are strictly hidden.
2. **Fresh Evaluation Split:** Annotations must be collected on a fresh, held-out sample (such as the normal-control dataset or unseen CARLA scenario seeds).
3. **Independent Reviewer:** The reviewer must not have previewed model proposals on the evaluated frames.

---

## 5. Remaining Workflow to Finalize Human Ground Truth

1. **Track Identity Reconciliation (Human Review):**
   - Reviewer opens the Annotation Studio on `http://127.0.0.1:5050/`.
   - Inspects the 9 suspect tracks identified in Section 3.1 and updates `Track ID` fields so each physical obstacle retains a unique, unchanging integer across time.
2. **Re-Confirmation of All 40 Frames:**
   - After inspecting target boxes and verifying track continuity, the reviewer inputs their Reviewer Name (`Aditya`) and clicks **Confirm & Next Frame** for each of the 40 frames.
3. **Execution Compliance:**
   - No Git commits, merges, resets, or destructive cleanup were performed.
   - Tracked git diff is 100% clean (`git diff --stat` is empty).
   - All immutable queues (`pilot_annotations.json`, `baseline_pilot_annotations_immutable.json`, `normal_control_annotations.json`) remain untouched.

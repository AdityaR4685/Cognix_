# Cognix — Forensic Audit and Verification Report: 40-Frame Annotation Pass

**Report Identifier:** `annotation_pass_001`
**Execution Timestamp:** 2026-10-20T17:08:00Z
**Branch:** `gat-testing-complete-checkpoint`
**HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Perception Environment:** `C:\Cognix_Envs\m6_perception`
**Status:** **READ-ONLY AUDIT & PRESERVATION COMPLETE — PROPOSED CORRECTION DOCUMENTED — AWAITING USER DIRECTION**

---

## 1. Executive Summary & Forensic Checkpoint Ledger

A read-only audit was performed on the completed 40-frame anomaly-pilot annotation pass in [`working_human_review_pilot_annotations.json`](file:///C:/Users/Aditya/Cognix2_recovered/reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json). All 40 frames were verified against the canonical M8 manifest and physical disk images. Before inspection, a byte-for-byte timestamped backup was created and cryptographically verified.

### 1.1 Complete Evidence & Queue Hash Ledger

| Asset / Queue | Role | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M8 Canonical Queue** | Immutable Reference | `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **M9 Immutable Baseline** | Benchmark Baseline | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **Normal Control Reference** | Benchmark Reference | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **100% IMMUTABLE / UNTOUCHED** |
| **Working Normal Control** | Normal Review Queue | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **100% IMMUTABLE / UNTOUCHED** |
| **Forensic Snapshot 1** | Prior Session Backup | `reports/.../annotation_review_recovery_001/forensic_audit_snapshot/working_human_review_pilot_annotations_snapshot_20261020_153711.json` | `423ce36af56eb765d0b346f77bfe5976136ad43aaf1fb16a85fadcd3862bcdac` | **SEALED COPY PRESERVED** |
| **Forensic Snapshot 2** | Pre-Recovery Backup | `reports/.../annotation_review_recovery_001/forensic_audit_snapshot/working_human_review_pilot_annotations_pre_recovery_20261020_160500.json` | `423ce36af56eb765d0b346f77bfe5976136ad43aaf1fb16a85fadcd3862bcdac` | **SEALED COPY PRESERVED** |
| **Forensic Snapshot 3** | Re-Confirmed Backup | `reports/.../annotation_review_recovery_001/forensic_audit_snapshot/working_human_review_pilot_annotations_reconfirmed_20261020_162000.json` | `f34345762e94f588006b89acaf8e40bad00d5de26a0159afc0b23db817e6ad72` | **SEALED COPY PRESERVED** |
| **Forensic Snapshot 4** | Pre-AI Recovery Backup | `reports/.../ai_assisted_annotation_001/forensic_backup/working_human_review_pilot_annotations_pre_ai_recovery_20261020_162000.json` | `f34345762e94f588006b89acaf8e40bad00d5de26a0159afc0b23db817e6ad72` | **SEALED COPY PRESERVED** |
| **New Audit Pass Backup** | Current Pass-001 Backup | `reports/.../annotation_pass_001/forensic_backup/working_human_review_pilot_annotations_pass001_20261020_170200.json` | `cde2040b137f6915c57e7b144d210a104673d655efacbd1b8dab1360fbe58e09` | **BYTE-FOR-BYTE VERIFIED** |
| **Working Pilot Queue (Current)** | Human Reviewed Queue | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `cde2040b137f6915c57e7b144d210a104673d655efacbd1b8dab1360fbe58e09` | **AUDITED & PRESERVED** |

---

## 2. Comprehensive 40-Frame Audit Findings

### 2.1 Manifest Alignment & Frame Image Hashes
- **Total Frames Present:** Exactly 40 frame entries across the 4 canonical anomaly pilot scenarios.
- **Manifest Alignment:** 100% matched against M8 canonical scenario IDs and tick intervals (`0, 5, 10, 15, 20, 25, 30, 35, 40, 45`).
- **Disk Image Hashes:** All 40 frames matched their physical disk JPEG images byte-for-byte with zero hash discrepancies.

### 2.2 Review and Confirmation Status
- **Frames with Saved Objects:** **40 / 40 (100%)**
- **Human-Confirmed (Signed) Frames:** **39 / 40 (97.5%)**
  - Confirmed by: `Aditya`
  - Confirmation timestamps: Session executed on 2026-10-20 between `11:10:43 UTC` and `11:28:08 UTC`.
- **Unconfirmed / Pending Review Frames:** **1 / 40 (2.5%)**
  - **Frame 6 (Tick 30 of `test/anomaly/Town01/running-pedestrian/scenario-1`):**
    - `reviewer_id`: `null`
    - `confirmed_at_utc`: `null`
    - `review_state`: `"unreviewed"`
    - Contains 3 saved objects with `confirmation_status = "pending_review"`:
      1. Object ID `obj_1792494724784_1_0`: `vehicle` (CARLA ego-vehicle hood, Track ID 1)
      2. Object ID `obj_1792494724784_12_1`: `person` (Pedestrian, Track ID 12)
      3. Object ID `obj_1792494724784_16_2`: `person` (Pedestrian, Track ID 16)
    - **Finding:** Objects were saved to disk via "Save Annotations", but "Confirm & Next Frame" was not clicked for Frame 6.

### 2.3 Object Inventory, Geometry, and Integrity Checks
- **Total Objects in Dataset:** **121 objects**
- **Unassigned Categories:** **0** (All 121 objects have valid classes; required class selection gate held).
- **Invalid Bounding Boxes:** **0** (All coordinates are within $[0, 1920] \times [0, 1080]$, non-negative, non-inverted, and non-degenerate).
- **Duplicate Track IDs in Same Frame:** **0** (Uniqueness constraint satisfied across all 40 frames).
- **Duplicate Object IDs in Same Frame:** **0** (Global UUIDs unique).

### 2.4 Breakdown by Hazard Class
| Object Category | Count | Percentage |
| :--- | :--- | :--- |
| **`vehicle`** | 59 | 48.8% |
| *(of which CARLA Ego Vehicle Hood)* | *(40)* | *(33.1%)* |
| *(of which External Genuine Vehicles)* | *(19)* | *(15.7%)* |
| **`person`** | 45 | 37.2% |
| **`road_debris_or_prop`** | 17 | 14.0% |
| **`bicycle_or_motorcycle`** | 0 | 0.0% |
| **`ignore_ambiguous`** | 0 | 0.0% |
| **Total** | **121** | **100.0%** |

### 2.5 Breakdown by Provenance
| Provenance | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **`ai_accepted`** | 62 | 51.2% | Faster R-CNN proposals accepted directly by reviewer |
| **`unspecified`** | 31 | 25.6% | The 31 manual drafts from earlier recovery pass |
| **`human_corrected`** | 21 | 17.4% | Model proposals modified in class or coordinates |
| **`human_added`** | 7 | 5.8% | Obstacles drawn manually from scratch (e.g. road crates) |
| **`independent_human`** | 0 | 0.0% | Objects created in unbiased independent mode |
| **Total AI-Assisted Fraction** | **83 / 121 (68.6%)** | — | $(62\text{ accepted} + 21\text{ corrected}) / 121$ |

### 2.6 Metadata Attributes (Visibility & Occlusion)
- **Visibility:** `clear`: 70 (57.9%), `degraded_weather`: 49 (40.5%), `glare_or_shadow`: 2 (1.7%).
- **Occlusion:** `none`: 113 (93.4%), `partial`: 6 (5.0%), `heavy`: 2 (1.7%).
- **Confirmation Status:** `human_confirmed`: 118 (97.5%), `pending_review`: 3 (2.5%, all in Frame 6).

---

## 3. Detailed Audit: CARLA Ego Vehicle (Track ID 1)

### 3.1 Spatial and Physical Profile of the Ego Vehicle
Inspection of the bounding box coordinates across all 40 frames identified a prominent, recurring bounding box positioned at the bottom of the camera frame:
- **Horizontal Bounds:** $x_1 \approx 384\text{ to }491\text{ px}$, $x_2 \approx 1433\text{ to }1536\text{ px}$ (Width $\approx 1050\text{ px}$, centered horizontally).
- **Vertical Bounds:** $y_1 \approx 872\text{ to }896\text{ px}$, $y_2 \approx 1063\text{ to }1080\text{ px}$ (Height $\approx 190\text{ px}$, flush against the bottom edge).
- **Physical Interpretation:** This represents the **front hood and dashboard of the CARLA ego vehicle** upon which the forward-facing camera is mounted. Because Faster R-CNN (COCO) detects car hoods as `car`, the model proposed a bounding box on the ego vehicle hood in every single frame, which was accepted as a `vehicle`.

### 3.2 Ego Vehicle Occurrences vs Track ID 1
The ego vehicle appears in **all 40 frames**. However, **Track ID 1 is not exclusively the ego vehicle**:
1. **Ego Vehicle with Track ID 1 (23 frames):**
   - `test/anomaly/Town01/running-pedestrian/scenario-1`: Frames 6, 7, 8, 9 (Ticks 30, 35, 40, 45).
   - `test/anomaly/Town01/spawn-props/scenario-1`: Frames 10, 11, 12, 13, 14, 15, 17, 18, 19.
   - `test/anomaly/Town01/change-weather/scenario-10`: Frames 30, 31, 32, 33, 34, 35, 36, 37, 38, 39.
2. **Ego Vehicle with Track ID 2 (11 frames):**
   - `test/anomaly/Town01/spawn-props/scenario-1`: Frame 16 (Tick 30).
   - `test/anomaly/Town01/change-weather/scenario-1`: Frames 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 (Ticks 0–45).
3. **Ego Vehicle with Other Track IDs (6 frames):**
   - `test/anomaly/Town01/running-pedestrian/scenario-1`:
     - Frame 0 (Tick 0): Track ID 6
     - Frame 1 (Tick 5): Track ID 7
     - Frame 2 (Tick 10): Track ID 9
     - Frame 3 (Tick 15): Track ID 8
     - Frame 4 (Tick 20): Track ID 6
     - Frame 5 (Tick 25): Track ID 3
4. **Track ID 1 Belonging to Non-Ego Objects (8 frames):**
   - In Frames 0, 1, 2, 3, 4, 5, 16, and 20, Track ID 1 belongs to **genuine external hazards**:
     - Frame 0 (Tick 0): External Vehicle parked/moving on street (`[494, 559, 778, 721]`).
     - Frames 1, 2, 4, 20: Pedestrians walking in roadway (`[510, 518, 554, 628]`, etc.).
     - Frames 3, 5, 16: Road debris / spawned crates (`[1546, 598, 1640, 744]`, etc.).

### 3.3 Separation of Genuine External Vehicles (19 Objects)
A total of **19 genuine external vehicles** exist in the dataset and must be strictly preserved:
- Frame 0 (Tick 0): Track ID 1 (`[494, 559, 778, 721]`)
- Frames 20–26 (Ticks 0–30): Track IDs 4, 5, 7, 14 (Parked/traffic vehicles in Town01, $y \in [527, 972]$)
- Frames 30–39 (Ticks 0–45): Track IDs 2, 7, 8, 10, 11, 15, 16, 17 (Traffic vehicles driving in Town01, $y \in [521, 785]$)

---

## 4. Scientific Impact Analysis & Recovery Recommendation

### 4.1 Evaluation Contamination Risk
If the 40 ego-vehicle bounding boxes remain categorized as `vehicle`:
1. **Target Hazard Contamination:** The evaluation taxonomy treats `vehicle` as an external collision hazard. Including the camera vehicle artificially inflates ground-truth vehicle count by $+40$ (from 19 genuine vehicles to 59).
2. **False Precision/Recall Inflation:** Because Faster R-CNN consistently detects its own vehicle's hood, the evaluator records $+40$ True Positives, producing misleadingly high recall and F1 scores on collision hazards.
3. **Severe MOTA Distortion:** Multiple track ID reassignments on the ego vehicle (IDs 6, 7, 9, 8, 6, 3, 1, 2) would artificially inflate tracker identity switches.

### 4.2 Deletion vs `ignore_ambiguous` Trade-off
- **Risk of Simple Deletion:** If the ego vehicle boxes are simply deleted from the annotation file:
  - The ground-truth count drops to 19 genuine vehicles.
  - However, Faster R-CNN will still predict the car hood in all 40 frames.
  - Under [`TrackingEvaluator.evaluate_detections`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/tracking_evaluator.py#L328-L350), any predicted box that has no matching ground-truth and no overlapping ignore region is penalized as a **Target-Class False Alarm (Ghost Hazard)**.
  - The detector would receive **40 false-positive penalties**, artificially degrading precision.
- **The Mathematically & Scientifically Sound Solution: `ignore_ambiguous`**
  - Reclassify the 40 ego-vehicle boxes to `ObjectCategory.IGNORE_AMBIGUOUS` (`"ignore_ambiguous"`).
  - Add explicit review notes: `"CARLA ego vehicle front hood/dashboard; excluded from external collision hazards"`.
  - **Behavior under Cognix Evaluation Protocol:**
    1. In [`TrackingEvaluator`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/tracking_evaluator.py#L278): `active_gts` excludes `ignore_ambiguous`, so ground-truth vehicle count is exactly 19.
    2. In detection matching: When Faster R-CNN detects the ego hood, $\text{IoU} \ge 0.50$ against the ignore box triggers `overlaps_ignore = True`, **absorbing the prediction without a false-positive penalty**.
    3. In tracking evaluation ([`TrackingEvaluator.evaluate_tracking`](file:///C:/Users/Aditya/Cognix2_recovered/cognix/evaluation/tracking_evaluator.py#L420)): `ignore_ambiguous` objects are excluded from trajectory tracking, preventing tracking metric contamination.

---

## 5. Complete Frame-by-Frame Completion Status Table

| Frame | Scenario | Tick | Objects | Class Breakdown | Reviewer | Confirmed Timestamp (UTC) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 00 | `running-pedestrian/scenario-1` | 00 | 7 | veh:2 (1 ego, 1 ext), per:4, deb:1 | Aditya | `2026-10-20T11:10:43.803Z` |
| 01 | `running-pedestrian/scenario-1` | 05 | 8 | veh:1 (ego), per:4, deb:3 | Aditya | `2026-10-20T11:11:00.438Z` |
| 02 | `running-pedestrian/scenario-1` | 10 | 10 | veh:1 (ego), per:6, deb:3 | Aditya | `2026-10-20T11:11:15.189Z` |
| 03 | `running-pedestrian/scenario-1` | 15 | 9 | veh:1 (ego), per:5, deb:3 | Aditya | `2026-10-20T11:11:31.214Z` |
| 04 | `running-pedestrian/scenario-1` | 20 | 6 | veh:1 (ego), per:2, deb:3 | Aditya | `2026-10-20T11:11:44.108Z` |
| 05 | `running-pedestrian/scenario-1` | 25 | 5 | veh:1 (ego), per:2, deb:2 | Aditya | `2026-10-20T11:11:52.590Z` |
| 06 | `running-pedestrian/scenario-1` | 30 | 3 | veh:1 (ego), per:2, deb:0 | **UNCONFIRMED** | `N/A (Pending Review)` |
| 07 | `running-pedestrian/scenario-1` | 35 | 3 | veh:1 (ego), per:2, deb:0 | Aditya | `2026-10-20T11:13:16.997Z` |
| 08 | `running-pedestrian/scenario-1` | 40 | 6 | veh:1 (ego), per:5, deb:0 | Aditya | `2026-10-20T11:14:37.696Z` |
| 09 | `running-pedestrian/scenario-1` | 45 | 5 | veh:1 (ego), per:4, deb:0 | Aditya | `2026-10-20T11:15:33.295Z` |
| 10 | `spawn-props/scenario-1` | 00 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:16:23.446Z` |
| 11 | `spawn-props/scenario-1` | 05 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:16:37.241Z` |
| 12 | `spawn-props/scenario-1` | 10 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:16:49.360Z` |
| 13 | `spawn-props/scenario-1` | 15 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:17:06.560Z` |
| 14 | `spawn-props/scenario-1` | 20 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:17:22.295Z` |
| 15 | `spawn-props/scenario-1` | 25 | 2 | veh:1 (ego), per:0, deb:1 | Aditya | `2026-10-20T11:17:53.495Z` |
| 16 | `spawn-props/scenario-1` | 30 | 2 | veh:1 (ego), per:0, deb:1 | Aditya | `2026-10-20T11:18:53.357Z` |
| 17 | `spawn-props/scenario-1` | 35 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:19:13.174Z` |
| 18 | `spawn-props/scenario-1` | 40 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:19:24.316Z` |
| 19 | `spawn-props/scenario-1` | 45 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:19:36.707Z` |
| 20 | `change-weather/scenario-1` | 00 | 4 | veh:2 (1 ego, 1 ext), per:2, deb:0 | Aditya | `2026-10-20T11:20:56.656Z` |
| 21 | `change-weather/scenario-1` | 05 | 3 | veh:2 (1 ego, 1 ext), per:1, deb:0 | Aditya | `2026-10-20T11:21:24.486Z` |
| 22 | `change-weather/scenario-1` | 10 | 3 | veh:2 (1 ego, 1 ext), per:1, deb:0 | Aditya | `2026-10-20T11:21:52.342Z` |
| 23 | `change-weather/scenario-1` | 15 | 3 | veh:2 (1 ego, 1 ext), per:1, deb:0 | Aditya | `2026-10-20T11:22:26.334Z` |
| 24 | `change-weather/scenario-1` | 20 | 3 | veh:2 (1 ego, 1 ext), per:1, deb:0 | Aditya | `2026-10-20T11:22:55.226Z` |
| 25 | `change-weather/scenario-1` | 25 | 3 | veh:2 (1 ego, 1 ext), per:1, deb:0 | Aditya | `2026-10-20T11:23:20.223Z` |
| 26 | `change-weather/scenario-1` | 30 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:23:51.133Z` |
| 27 | `change-weather/scenario-1` | 35 | 2 | veh:1 (ego), per:1, deb:0 | Aditya | `2026-10-20T11:24:26.597Z` |
| 28 | `change-weather/scenario-1` | 40 | 2 | veh:1 (ego), per:1, deb:0 | Aditya | `2026-10-20T11:24:52.451Z` |
| 29 | `change-weather/scenario-1` | 45 | 1 | veh:1 (ego), per:0, deb:0 | Aditya | `2026-10-20T11:25:15.638Z` |
| 30 | `change-weather/scenario-10` | 00 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:25:37.386Z` |
| 31 | `change-weather/scenario-10` | 05 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:25:49.000Z` |
| 32 | `change-weather/scenario-10` | 10 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:26:01.066Z` |
| 33 | `change-weather/scenario-10` | 15 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:26:12.305Z` |
| 34 | `change-weather/scenario-10` | 20 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:26:24.964Z` |
| 35 | `change-weather/scenario-10` | 25 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:26:47.860Z` |
| 36 | `change-weather/scenario-10` | 30 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:27:05.437Z` |
| 37 | `change-weather/scenario-10` | 35 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:27:24.048Z` |
| 38 | `change-weather/scenario-10` | 40 | 3 | veh:3 (1 ego, 2 ext), per:0, deb:0 | Aditya | `2026-10-20T11:27:53.604Z` |
| 39 | `change-weather/scenario-10` | 45 | 2 | veh:2 (1 ego, 1 ext), per:0, deb:0 | Aditya | `2026-10-20T11:28:08.948Z` |

---

## 6. Action Items & Next Steps

1. **Confirmation of Frame 6:**
   - Frame 6 (Tick 30 of `running-pedestrian/scenario-1`) contains 3 saved annotations but was not confirmed (`reviewer_id: None`). Aditya can confirm this frame in the studio UI or confirm authorization to sign it with the session metadata.
2. **Approval of Ego Vehicle Recovery Procedure:**
   - Awaiting user authorization to apply the recommended narrowest correction:
     - Reclassify the **40 ego vehicle hood bounding boxes** ($y_1 \ge 800, y_2 \ge 1050, \Delta x \ge 800$) to `category: "ignore_ambiguous"`.
     - Retain all 19 genuine external vehicles unchanged.
     - Add review notes documenting the ego vehicle hood exclusion.
3. **Preservation Status:**
   - Working file is preserved at hash `cde2040b137f6915c57e7b144d210a104673d655efacbd1b8dab1360fbe58e09`.
   - Backup is verified at `reports/carla_runtime_v1/evaluation_runs/annotation_pass_001/forensic_backup/working_human_review_pilot_annotations_pass001_20261020_170200.json`.
   - No Git writes, resets, or destructive modifications have taken place. Tracked working tree remains 100% clean.

# Cognix — Track Identity Reconciliation Audit and Proposed Correction Report

**Report Identifier:** `track_identity_reconciliation_001`
**Execution Timestamp:** 2026-10-20T17:35:00Z
**Branch:** `gat-testing-complete-checkpoint`
**HEAD Commit:** `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Perception Environment:** `C:\Cognix_Envs\m6_perception`
**Status:** **AUDIT & PROPOSED CORRECTIONS COMPLETE — NO ANNOTATION WRITES PERFORMED — IDENTITY SCORING FROZEN**

---

## 1. Executive Summary & Forensic Checkpoint Ledger

Following the successful exclusion of 40 CARLA ego-vehicle hood boxes in milestone `ego_hood_correction_001`, this audit investigates and resolves the **nine reported track-identity discontinuities** across the 40-frame anomaly-pilot dataset.

### 1.1 Strict Preservation and Operational Boundary Adherence
- **Zero Git Writes:** No `git add`, `commit`, `push`, `pull`, `fetch`, `merge`, `rebase`, `reset`, `clean`, or branch switches were executed (`git diff --stat` remains 100% empty).
- **Working File Preservation:** The working annotation file was **not modified** during this audit. This pass is strictly a forensic analysis, visual review generation, and proposed mapping deliverable.
- **Pre-Audit Forensic Backup:** Created and byte-for-byte verified prior to inspection.
- **Reference Integrity:** The 40 excluded ego-hood boxes remain permanently excluded and recorded in `ego_hood_exclusion_ledger.json`. All immutable reference baselines remain untouched.
- **Scoring Status:** Official tracking metrics (CLEAR MOTA, ID Switches, Fragmentations) remain **strictly frozen** and marked **NOT READY FOR SCORING** until human review adjudicates the proposed mappings.

### 1.2 Hash and Checkpoint Ledger

| Asset / Queue | Role | Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M8 Canonical Queue** | Immutable Reference | `reports/carla_runtime_v1/evaluation_runs/m8_ground_truth_tracking_001/pilot_annotations.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **M9 Immutable Baseline** | Benchmark Baseline | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/baseline_pilot_annotations_immutable.json` | `bb20dd466a2e518d19e44e8ec0d54eb5b5632a7dcb6a42c32db3f833b6989f15` | **100% IMMUTABLE / UNTOUCHED** |
| **Normal Control Reference**| Benchmark Reference | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/normal_control_annotations.json` | `3d8733b4fd3b187daf7ce8a1aeebbf003ee67cc82cd28fe7d03f248fc2e451ad` | **100% IMMUTABLE / UNTOUCHED** |
| **Ego-Hood Exclusion Ledger**| Forensic Ledger | `reports/carla_runtime_v1/evaluation_runs/ego_hood_correction_001/ego_hood_exclusion_ledger.json` | `1335f2c4a1dd5ebc239ab32351a27141ee773c6a6e2ae4dfb9a0bc76085b8a86` | **40 EXCLUSIONS PRESERVED** |
| **Pre-Reconciliation Backup**| Audit Backup | `reports/.../track_identity_reconciliation_001/forensic_backup/working_human_review_pilot_annotations_pre_reconciliation_20261020_172500.json` | `d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06` | **BYTE-FOR-BYTE VERIFIED** |
| **Working Pilot Annotations**| Current Working Set | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations.json` | `d745e1729920da98aa3d9f009ce5706f307dad68ffc9d51dea087071c819eb06` | **UNMODIFIED / 81 TARGET OBJS** |

---

## 2. Root Cause Analysis: Why Did Track Discontinuities Occur?

Analysis of the 81 retained target objects across all 40 frames revealed the exact root mechanism behind the track ID discontinuities:

1. **Per-Frame Counter Resets in Early Manual Drafting (Frames 00–04):**
   In `test/anomaly/Town01/running-pedestrian/scenario-1`, Frames 00 through 04 were originally manually drafted by the reviewer before AI assistance was enabled. The annotation interface automatically numbered bounding boxes starting from `1, 2, 3, 4, 5...` on **each frame independently**:
   - The first object drawn on Frame 00 became `track_id=1` (the parked car).
   - The first object drawn on Frame 01 became `track_id=1` (a running pedestrian).
   - The first object drawn on Frame 03 became `track_id=1` (a road crate on the far right).
   - The first object drawn on Frame 04 became `track_id=1` (a walking pedestrian on the sidewalk).
   Because Track ID 1 was reassigned per-frame to different objects, Track ID 1 hopped across 3 different classes (`vehicle` $\rightarrow$ `person` $\rightarrow$ `road_debris_or_prop`). The exact same per-frame auto-incrementing occurred for Track IDs 2, 3, 4, 5, 7, and 8.

2. **Duplicate Co-located Bounding Boxes from AI-Assisted Proposal Ingestion:**
   When AI proposals were ingested in milestone `m9_annotation_readiness_001` (Step 8), Faster R-CNN proposals were accepted while the human reviewer's existing manual boxes were also retained. In Frames 00, 01, 02, and 03, the schema prevented assigning duplicate `track_id` values to two boxes in the same frame. Consequently, AI proposals on pedestrians were assigned distinct track IDs (e.g., Track 7, Track 8, Track 10, Track 13), creating co-located bounding box pairs on the exact same physical individual.

3. **Detector-Tracker Frame Resets in `change-weather/scenario-1`:**
   In `change-weather/scenario-1`, Track ID 4 was initially assigned to a distant oncoming vehicle in Frame 20 (Tick 00), but was assigned to a pedestrian on the right sidewalk in Frame 21 (Tick 05). In Frame 27 (Tick 35), an entity on the left sidewalk was manually added with Track ID 3, creating a 35-tick jump across the roadway from the pedestrian labeled Track ID 3 in Frame 20.

---

## 3. Detailed Forensic Audit of All Nine Reported Discontinuities

Each case was evaluated using:
- Physical RGB image files and their verified SHA-256 hashes.
- Exact bounding box coordinates $[x_1, y_1, x_2, y_2]$, geometric center $(c_x, c_y)$, and bounding dimensions $(w, h)$.
- Inter-frame spatial displacement between consecutive 5-tick samples.
- Class semantics and visual appearance crops in `visual_review/crops/`.

```
========================================================================================================================
Summary of the Nine Discontinuities
Case 1: RP Sc-1 | Track 1 | Ticks 00, 05, 10, 15, 20, 25 | Classes: vehicle, person, road_debris (6 distinct objects)
Case 2: RP Sc-1 | Track 2 | Ticks 00, 05, 10, 15, 20, 25 | Classes: person, road_debris (5 distinct objects)
Case 3: RP Sc-1 | Track 3 | Ticks 00, 05, 10, 15, 20     | Classes: person, road_debris (4 distinct objects)
Case 4: RP Sc-1 | Track 4 | Ticks 00, 05, 10, 15, 20     | Classes: person, road_debris (5 distinct objects)
Case 5: RP Sc-1 | Track 5 | Ticks 00, 05, 10, 15, 20     | Class: road_debris (Jumping across 4 different road crates)
Case 6: RP Sc-1 | Track 7 | Ticks 00, 10, 15             | Classes: person, road_debris (3 distinct objects)
Case 7: RP Sc-1 | Track 8 | Ticks 05, 10                 | Classes: person, road_debris (2 distinct objects)
Case 8: CW Sc-1 | Track 3 | Ticks 00, 35                 | Class: person (35-tick gap, 516px jump across street)
Case 9: CW Sc-1 | Track 4 | Ticks 00, 05                 | Classes: vehicle, person (Oncoming car vs sidewalk person)
========================================================================================================================
```

---

### Case 1: `running-pedestrian/scenario-1` — Current Track ID 1
* **Discontinuity Type:** Severe Multi-Class & Large Spatial Jumps ($>1000\text{ px}$)
* **Affected Objects:** 6 occurrences across Ticks 00, 05, 10, 15, 20, 25.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 00 | 00 | `478af98d9da2` | `obj_1792487133690` | `vehicle` | `[494, 559, 778, 721]` | `(636.0, 640.0)` | $284 \times 162$ | `unspecified` | Stationary parked red hatchback on left curb |
| 01 | 05 | `8676f4d994bc` | `obj_1792490117128` | `person` | `[510, 518, 554, 628]` | `(532.0, 573.0)` | $44 \times 110$ | `unspecified` | Lead pedestrian sprinting left across roadway |
| 02 | 10 | `e7be2fc3afc6` | `obj_1792490322347` | `person` | `[288, 522, 357, 653]` | `(322.5, 587.5)` | $69 \times 131$ | `unspecified` | Lead pedestrian continuing leftward sprint |
| 03 | 15 | `b72ae0860491` | `obj_1792490469863` | `road_debris_or_prop`| `[1546, 598, 1640, 744]`| `(1593.0, 671.0)`| $94 \times 146$ | `unspecified` | Wooden roadway crate passing on right |
| 04 | 20 | `7bf4217dcc54` | `obj_1792490602453` | `person` | `[1157, 525, 1210, 637]`| `(1183.5, 581.0)`| $53 \times 112$ | `unspecified` | Pedestrian 1 walking forward on right sidewalk |
| 05 | 25 | `17738f7c8e29` | `obj_1792494562696` | `road_debris_or_prop`| `[717, 494, 783, 602]` | `(750.0, 548.0)` | $66 \times 108$ | `human_added`| Crate D further down the roadway corridor |

* **Analysis & Physical Truth:**
  Track ID 1 does **not** represent a single physical entity. It connects five distinct real-world objects:
  1. Parked vehicle at Tick 00.
  2. Lead running pedestrian at Ticks 05 and 10 (velocity $\approx 42\text{ px/tick}$ leftward).
  3. Road crate at Tick 15 ($x=1593$).
  4. Sidewalk pedestrian at Tick 20 ($x=1183.5$).
  5. Distant road crate at Tick 25 ($x=750$).
* **Proposed Disentanglement:**
  - `obj_1792487133690` (Tick 00) $\rightarrow$ Assign to dedicated Track `101` (`vehicle`).
  - `obj_1792490117128` (Tick 05) & `obj_1792490322347` (Tick 10) $\rightarrow$ Assign to dedicated Track `102` (Lead Runner).
  - `obj_1792490469863` (Tick 15) $\rightarrow$ Assign to dedicated Track `106` (Crate C).
  - `obj_1792490602453` (Tick 20) $\rightarrow$ Assign to Track `12` (Pedestrian 1 on sidewalk).
  - `obj_1792494562696` (Tick 25) $\rightarrow$ Assign to dedicated Track `107` (Crate D).
* **Confidence:** **High**. Every object is visually and geometrically separable with zero ambiguity.
* **Risk of Merging Unrelated Objects:** None; this operation splits an erroneous multi-object amalgamation into clean, single-identity tracks.

---

### Case 2: `running-pedestrian/scenario-1` — Current Track ID 2
* **Discontinuity Type:** Large Spatial Jumps ($>450\text{ px}$) and Class Jump to `road_debris_or_prop` at Tick 25.
* **Affected Objects:** 6 occurrences across Ticks 00, 05, 10, 15, 20, 25.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 00 | 00 | `478af98d9da2` | `obj_1792489707036` | `person` | `[1055, 527, 1080, 591]`| `(1067.5, 559.0)`| $25 \times 64$ | `unspecified` | Pedestrian 1 on right sidewalk |
| 01 | 05 | `8676f4d994bc` | `obj_1792490198709` | `person` | `[579, 527, 624, 614]` | `(601.5, 570.5)` | $45 \times 87$ | `unspecified` | Second runner sprinting left across road |
| 02 | 10 | `e7be2fc3afc6` | `obj_1792490347072` | `person` | `[439, 522, 494, 651]` | `(466.5, 586.5)` | $55 \times 129$ | `unspecified` | Second runner continuing leftward sprint |
| 03 | 15 | `b72ae0860491` | `obj_1792490475680` | `person` | `[1135, 527, 1187, 623]`| `(1161.0, 575.0)`| $52 \times 96$ | `unspecified` | Pedestrian 1 on right sidewalk |
| 04 | 20 | `7bf4217dcc54` | `obj_1792490605123` | `person` | `[1059, 559, 1089, 623]`| `(1074.0, 591.0)`| $30 \times 64$ | `unspecified` | Pedestrian 2 on right sidewalk |
| 05 | 25 | `17738f7c8e29` | `obj_1792494577069` | `road_debris_or_prop`| `[890, 250, 1069, 623]` | `(979.5, 436.5)` | $179 \times 373$| `human_added`| Crate A on roadway |

* **Analysis & Physical Truth:**
  - Tick 00 is Pedestrian 1 on sidewalk ($x=1067.5$).
  - Ticks 05 and 10 represent the Second Runner ($x=601.5 \rightarrow 466.5$, velocity $\approx 27\text{ px/tick}$ leftward).
  - Tick 15 jumps 694px back to Pedestrian 1 on sidewalk ($x=1161.0$).
  - Tick 20 is Pedestrian 2 on sidewalk ($x=1074.0$).
  - Tick 25 is Crate A on roadway ($x=979.5$).
* **Proposed Disentanglement:**
  - `obj_1792489707036` (Tick 00) & `obj_1792490475680` (Tick 15) $\rightarrow$ Assign to Track `12` (Pedestrian 1).
  - `obj_1792490198709` (Tick 05) & `obj_1792490347072` (Tick 10) $\rightarrow$ Assign to dedicated Track `103` (Second Runner).
  - `obj_1792490605123` (Tick 20) $\rightarrow$ Assign to Track `16` (Pedestrian 2).
  - `obj_1792494577069` (Tick 25) $\rightarrow$ Assign to dedicated Track `104` (Crate A).
* **Confidence:** **High**.

---

### Case 3: `running-pedestrian/scenario-1` — Current Track ID 3
* **Discontinuity Type:** Spatial Jump at Tick 05 and Class Jump to `road_debris_or_prop` at Tick 20.
* **Affected Objects:** 5 occurrences across Ticks 00, 05, 10, 15, 20.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 00 | 00 | `478af98d9da2` | `obj_1792489719606` | `person` | `[629, 529, 670, 561]` | `(649.5, 545.0)` | $41 \times 32$ | `unspecified` | Lead runner beginning sprint |
| 01 | 05 | `8676f4d994bc` | `obj_1792490217724` | `person` | `[1073, 527, 1121, 598]`| `(1097.0, 562.5)`| $48 \times 71$ | `unspecified` | Pedestrian 1 on right sidewalk |
| 02 | 10 | `e7be2fc3afc6` | `obj_1792490356024` | `person` | `[1103, 522, 1141, 619]`| `(1122.0, 570.5)`| $38 \times 97$ | `unspecified` | Pedestrian 1 on right sidewalk |
| 03 | 15 | `b72ae0860491` | `obj_1792490478756` | `person` | `[1043, 550, 1064, 609]`| `(1053.5, 579.5)`| $21 \times 59$ | `unspecified` | Pedestrian 2 on right sidewalk |
| 04 | 20 | `7bf4217dcc54` | `obj_1792490610042` | `road_debris_or_prop`| `[904, 303, 1059, 614]` | `(981.5, 458.5)` | $155 \times 311$| `unspecified` | Crate A on roadway |

* **Analysis & Physical Truth:**
  - Tick 00 is Lead Runner starting across the road ($x=649.5$).
  - Ticks 05 & 10 are Pedestrian 1 on right sidewalk ($x=1097.0 \rightarrow 1122.0$).
  - Tick 15 is Pedestrian 2 on right sidewalk ($x=1053.5$).
  - Tick 20 is Crate A on the roadway ($x=981.5$).
* **Proposed Disentanglement:**
  - `obj_1792489719606` (Tick 00) $\rightarrow$ Assign to dedicated Track `102` (Lead Runner).
  - `obj_1792490217724` (Tick 05) & `obj_1792490356024` (Tick 10) $\rightarrow$ Assign to Track `12` (Pedestrian 1).
  - `obj_1792490478756` (Tick 15) $\rightarrow$ Assign to Track `16` (Pedestrian 2).
  - `obj_1792490610042` (Tick 20) $\rightarrow$ Assign to dedicated Track `104` (Crate A).
* **Confidence:** **High**.

---

### Case 4: `running-pedestrian/scenario-1` — Current Track ID 4
* **Discontinuity Type:** Alternating Class Flips (`person` $\rightarrow$ `debris` $\rightarrow$ `debris` $\rightarrow$ `person` $\rightarrow$ `debris`) and Spatial Jumps.
* **Affected Objects:** 5 occurrences across Ticks 00, 05, 10, 15, 20.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 00 | 00 | `478af98d9da2` | `obj_1792489745555` | `person` | `[673, 520, 707, 554]` | `(690.0, 537.0)` | $34 \times 34$ | `unspecified` | Second runner beginning sprint |
| 01 | 05 | `8676f4d994bc` | `obj_1792490233877` | `road_debris_or_prop`| `[702, 287, 858, 635]` | `(780.0, 461.0)` | $156 \times 348$| `unspecified` | Crate B on left side of roadway |
| 02 | 10 | `e7be2fc3afc6` | `obj_1792490366164` | `road_debris_or_prop`| `[1329, 575, 1384, 660]`| `(1356.5, 617.5)`| $55 \times 85$ | `unspecified` | Crate C passing on right |
| 03 | 15 | `b72ae0860491` | `obj_1792490481344` | `person` | `[137, 506, 242, 692]` | `(189.5, 599.0)` | $105 \times 186$| `unspecified` | Lead runner finishing sprint on far left |
| 04 | 20 | `7bf4217dcc54` | `obj_1792490613825` | `road_debris_or_prop`| `[817, 413, 908, 575]` | `(862.5, 494.0)` | $91 \times 162$ | `unspecified` | Crate E in middle roadway |

* **Analysis & Physical Truth:**
  - Tick 00 is Second Runner starting sprint ($x=690.0$).
  - Tick 05 is Crate B ($x=780.0$).
  - Tick 10 is Crate C ($x=1356.5$).
  - Tick 15 is Lead Runner reaching the far left side ($x=189.5$).
  - Tick 20 is Crate E ($x=862.5$).
* **Proposed Disentanglement:**
  - `obj_1792489745555` (Tick 00) $\rightarrow$ Assign to dedicated Track `103` (Second Runner).
  - `obj_1792490233877` (Tick 05) $\rightarrow$ Assign to dedicated Track `105` (Crate B).
  - `obj_1792490366164` (Tick 10) $\rightarrow$ Assign to dedicated Track `106` (Crate C).
  - `obj_1792490481344` (Tick 15) $\rightarrow$ Assign to dedicated Track `102` (Lead Runner).
  - `obj_1792490613825` (Tick 20) $\rightarrow$ Assign to dedicated Track `108` (Crate E).
* **Confidence:** **High**.

---

### Case 5: `running-pedestrian/scenario-1` — Current Track ID 5
* **Discontinuity Type:** Large Spatial Jumps across multiple stationary roadway obstacles.
* **Affected Objects:** 5 occurrences across Ticks 00, 05, 10, 15, 20.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 00 | 00 | `478af98d9da2` | `obj_1792489803625` | `road_debris_or_prop`| `[759, 344, 883, 614]` | `(821.0, 479.0)` | $124 \times 270$| `unspecified` | Crate B on left side of roadway |
| 01 | 05 | `8676f4d994bc` | `obj_1792490251582` | `road_debris_or_prop`| `[913, 394, 1020, 609]`| `(966.5, 501.5)` | $107 \times 215$| `unspecified` | Crate A on roadway |
| 02 | 10 | `e7be2fc3afc6` | `obj_1792490381878` | `road_debris_or_prop`| `[620, 198, 833, 637]` | `(726.5, 417.5)` | $213 \times 439$| `unspecified` | Crate B on left side of roadway |
| 03 | 15 | `b72ae0860491` | `obj_1792490485457` | `road_debris_or_prop`| `[480, 72, 782, 685]`  | `(631.0, 378.5)` | $302 \times 613$| `unspecified` | Crate B on left side of roadway |
| 04 | 20 | `7bf4217dcc54` | `obj_1792490616740` | `road_debris_or_prop`| `[734, 502, 814, 596]` | `(774.0, 549.0)` | $80 \times 94$  | `unspecified` | Crate D down the road |

* **Analysis & Physical Truth:**
  - All objects belong to the `road_debris_or_prop` class, but Track 5 ping-pongs between two different crates:
    - Ticks 00, 10, and 15 track Crate B as the ego vehicle approaches and passes it on the left ($x=821.0 \rightarrow 726.5 \rightarrow 631.0$, size expanding from $124 \times 270$ to $302 \times 613$).
    - Tick 05 hops over to Crate A on the right ($x=966.5$).
    - Tick 20 hops to Crate D ($x=774.0$).
* **Proposed Disentanglement:**
  - `obj_1792489803625` (Tick 00), `obj_1792490381878` (Tick 10), `obj_1792490485457` (Tick 15) $\rightarrow$ Assign to dedicated Track `105` (Crate B).
  - `obj_1792490251582` (Tick 05) $\rightarrow$ Assign to dedicated Track `104` (Crate A).
  - `obj_1792490616740` (Tick 20) $\rightarrow$ Assign to dedicated Track `107` (Crate D).
* **Confidence:** **High**.

---

### Case 6: `running-pedestrian/scenario-1` — Current Track ID 7
* **Discontinuity Type:** Class Discontinuity (`person` $\rightarrow$ `road_debris_or_prop`) and Spatial Jump of 378px.
* **Affected Objects:** 3 occurrences across Ticks 00, 10, 15.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 00 | 00 | `478af98d9da2` | `obj_1792494631470_7_1` | `person` | `[650, 550, 669, 573]` | `(659.5, 561.5)` | $19 \times 23$ | `ai_accepted` | AI proposal on Lead Runner |
| 02 | 10 | `e7be2fc3afc6` | `obj_1792490419545` | `person` | `[1023, 550, 1052, 605]`| `(1037.5, 577.5)`| $29 \times 55$ | `unspecified` | Pedestrian 2 on right sidewalk |
| 03 | 15 | `b72ae0860491` | `obj_1792490493262` | `road_debris_or_prop`| `[908, 344, 1034, 603]` | `(971.0, 473.5)` | $126 \times 259$| `unspecified` | Crate A on roadway |

* **Analysis & Physical Truth:**
  - Tick 00 is an AI proposal on Lead Runner (`[650..669]` vs manual draft `[629..670]`).
  - Tick 10 is Pedestrian 2 walking along the right sidewalk ($x=1037.5$).
  - Tick 15 is Crate A on the roadway ($x=971.0$).
* **Proposed Disentanglement:**
  - `obj_1792494631470_7_1` (Tick 00) $\rightarrow$ Overlaps Lead Runner `obj_1792489719606`. **Flagged for Human Adjudication** (retire duplicate AI box or reassign to Track `102`).
  - `obj_1792490419545` (Tick 10) $\rightarrow$ Assign to Track `16` (Pedestrian 2).
  - `obj_1792490493262` (Tick 15) $\rightarrow$ Assign to dedicated Track `104` (Crate A).
* **Confidence:** **High** for Ticks 10 and 15; **Adjudication Required** for Tick 00 duplicate.

---

### Case 7: `running-pedestrian/scenario-1` — Current Track ID 8
* **Discontinuity Type:** Class Discontinuity (`person` $\rightarrow$ `road_debris_or_prop`).
* **Affected Objects:** 2 occurrences across Ticks 05, 10.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 01 | 05 | `8676f4d994bc` | `obj_1792494651746_8_1` | `person` | `[1087, 528, 1109, 596]`| `(1098.0, 562.0)`| $22 \times 68$ | `ai_accepted` | AI proposal on Pedestrian 1 |
| 02 | 10 | `e7be2fc3afc6` | `obj_1792490442869` | `road_debris_or_prop`| `[915, 360, 1025, 587]` | `(970.0, 473.5)` | $110 \times 227$| `unspecified` | Crate A on roadway |

* **Analysis & Physical Truth:**
  - Tick 05 is an AI proposal on Pedestrian 1 on the sidewalk (overlaps manual draft `obj_1792490217724` at `[1073..1121]`).
  - Tick 10 is Crate A on the roadway.
* **Proposed Disentanglement:**
  - `obj_1792494651746_8_1` (Tick 05) $\rightarrow$ Duplicate box on Pedestrian 1. **Flagged for Human Adjudication** (retire duplicate AI box or reassign to Track `12`).
  - `obj_1792490442869` (Tick 10) $\rightarrow$ Assign to dedicated Track `104` (Crate A).
* **Confidence:** **High**.

---

### Case 8: `change-weather/scenario-1` — Current Track ID 3
* **Discontinuity Type:** Large Temporal Gap (35 ticks / 7 frames) and Spatial Jump across roadway (516px).
* **Affected Objects:** 2 occurrences at Ticks 00 and 35.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 20 | 00 | `f10e6f206bcd` | `obj_1792495220726_3` | `person` | `[1039, 536, 1062, 597]`| `(1050.5, 566.5)`| $23 \times 61$ | `human_corrected`| Sidewalk pedestrian walking forward |
| 27 | 35 | `c4bce1275eeb` | `obj_1792495448698` | `person` | `[516, 529, 553, 592]` | `(534.5, 560.5)` | $37 \times 63$ | `human_added`| Distant object on opposite (left) curb |

* **Analysis & Physical Truth:**
  - At Tick 00, `obj_1792495220726_3` is the beginning of the clear pedestrian trajectory on the right sidewalk. Across Ticks 05, 10, 15, 20, 25, that person walked directly towards the ego vehicle, growing from $23 \times 61$ to $106 \times 338$ ($x \approx 1050 \rightarrow 1534$) before exiting off-screen right at Tick 30.
  - At Tick 35, `obj_1792495448698` is a distant entity on the opposite (left) sidewalk. In M7 model predictions, this was classified with score 0.44 as a `fire hydrant`.
  - There is zero continuity between these two observations: a 35-tick gap and opposite sides of the street.
* **Proposed Disentanglement:**
  - `obj_1792495220726_3` (Tick 00) $\rightarrow$ Reconcile into the right sidewalk pedestrian trajectory (or keep Track `3`).
  - `obj_1792495448698` (Tick 35) $\rightarrow$ Separate into dedicated Track `35` or **Flag for Human Adjudication** regarding whether it is a genuine pedestrian vs distant prop.
* **Confidence:** **High** that they are distinct entities.

---

### Case 9: `change-weather/scenario-1` — Current Track ID 4
* **Discontinuity Type:** Immediate Class Discontinuity (`vehicle` $\rightarrow$ `person`) between adjacent frames (Ticks 00 and 05).
* **Affected Objects:** 2 occurrences at Ticks 00 and 05.

| Frame | Tick | Frame SHA-256 | Object ID | Category | BBox $[x_1, y_1, x_2, y_2]$ | Center $(c_x, c_y)$ | Dimensions | Provenance | Visual Evidence |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 20 | 00 | `f10e6f206bcd` | `obj_1792495226200` | `vehicle` | `[876, 527, 926, 583]` | `(901.0, 555.0)` | $50 \times 56$ | `human_added`| Distant oncoming vehicle in lane |
| 21 | 05 | `d4fa5804b55f` | `obj_1792495271654_4` | `person` | `[1076, 535, 1105, 618]`| `(1090.5, 576.5)`| $29 \times 83$ | `human_corrected`| Sidewalk pedestrian walking forward |

* **Analysis & Physical Truth:**
  - At Tick 00, `obj_1792495226200` is the distant oncoming car in the opposing lane ($x=901.0$). At Tick 05, the exact same oncoming car is tracked as Track ID 5 (`obj_1792495264525_5`, `box=[862, 544, 915, 579]`, center `888.5, 561.5`), which continues forward across Ticks 10, 15, and 20.
  - At Tick 05, Track ID 4 was accidentally assigned to the pedestrian on the right sidewalk (`obj_1792495271654_4`), creating a class flip.
* **Proposed Disentanglement:**
  - `obj_1792495226200` (Tick 00) $\rightarrow$ Merge into Track `5` (the oncoming vehicle trajectory).
  - `obj_1792495271654_4` (Tick 05) $\rightarrow$ Reconcile into the right sidewalk pedestrian trajectory (e.g. Track `3`).
* **Confidence:** **High**.

---

## 4. Visual Review Package Reference

A complete visual review package has been rendered and stored in:
`reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/visual_review/`

### 4.1 Overview Composites
1. **`running_pedestrian_frames_00_to_04_composite.jpg`** ($1920 \times 1620$):
   Multi-panel grid displaying Frames 00 through 04 with bounding boxes, category colors (Orange = Person, Green = Road Debris, Blue = Vehicle), and current Track IDs (`T01`, `T02`, etc.) clearly visible.
2. **`change_weather_suspect_frames_composite.jpg`** ($1920 \times 1080$):
   Side-by-side comparison of Frames 20, 21, and 27 showing the oncoming car, sidewalk pedestrian, and distant entity.

### 4.2 High-Resolution Individual Frame Renders
- `running_pedestrian_frame_00_annotated.jpg` (Frame 00, Tick 00)
- `running_pedestrian_frame_01_annotated.jpg` (Frame 01, Tick 05)
- `running_pedestrian_frame_02_annotated.jpg` (Frame 02, Tick 10)
- `running_pedestrian_frame_03_annotated.jpg` (Frame 03, Tick 15)
- `running_pedestrian_frame_04_annotated.jpg` (Frame 04, Tick 20)
- `change_weather_frame_20_annotated.jpg` (Frame 20, Tick 00)
- `change_weather_frame_21_annotated.jpg` (Frame 21, Tick 05)
- `change_weather_frame_27_annotated.jpg` (Frame 27, Tick 35)

### 4.3 Object Crops Directory
`reports/carla_runtime_v1/evaluation_runs/track_identity_reconciliation_001/visual_review/crops/` contains 36 high-resolution image crops of every individual object involved in the nine discontinuities, allowing pixel-level verification of clothing, vehicle models, and crate geometry.

---

## 5. Proposed Complete Identity Mapping & Reconciliation Plan

To achieve mathematically valid ground truth for tracking evaluation without introducing new ID collisions or violating schema rules, the following mapping is proposed:

### 5.1 Physical Trajectory Allocation Table (`running-pedestrian/scenario-1`)

| Physical Trajectory | Entity Description | Class | Ticks Present | Object IDs Involved | Current Track IDs | Proposed Reconciled Track ID | Confidence |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **Parked Vehicle** | Red hatchback on left curb | `vehicle` | `00` | `obj_1792487133690` | 1 | **101** | High |
| **Lead Runner** | Sprinter crossing road to left | `person` | `00, 05, 10, 15` | `obj_1792489719606`, `obj_1792490117128`, `obj_1792490322347`, `obj_1792490481344` | 3, 1, 1, 4 | **102** | High |
| **Second Runner** | Sprinter following lead runner | `person` | `00, 05, 10` | `obj_1792489745555`, `obj_1792490198709`, `obj_1792490347072` | 4, 2, 2 | **103** | High |
| **Pedestrian 1** | Walking along right sidewalk | `person` | `00` to `45` | `obj_1792489707036` (00), `obj_1792490217724` (05), `obj_1792490356024` (10), `obj_1792490475680` (15), `obj_1792490602453` (20), `obj_..._12` (25, 30, 35, 40), `obj_..._23` (45) | 2, 3, 3, 2, 1, 12, 12, 12, 12, 23 | **12** | High |
| **Pedestrian 2** | Walking diagonally rightwards | `person` | `10` to `45` | `obj_1792490419545` (10), `obj_1792490478756` (15), `obj_1792490605123` (20), `obj_..._15` (25), `obj_..._16` (30, 35), `obj_..._18` (40), `obj_..._22` (45) | 7, 3, 2, 15, 16, 16, 18, 22 | **16** | High |
| **Crate A** | Stationary crate on roadway right | `debris` | `05, 10, 15, 20, 25` | `obj_1792490251582`, `obj_1792490442869`, `obj_1792490493262`, `obj_1792490610042`, `obj_1792494577069` | 5, 8, 7, 3, 2 | **104** | High |
| **Crate B** | Stationary crate on roadway left | `debris` | `00, 05, 10, 15` | `obj_1792489803625`, `obj_1792490233877`, `obj_1792490381878`, `obj_1792490485457` | 5, 4, 5, 5 | **105** | High |
| **Crate C** | Passing crate on far right sidewalk | `debris` | `05, 10, 15` | `obj_1792490271605`, `obj_1792490366164`, `obj_1792490469863` | 6, 4, 1 | **106** | High |
| **Crate D** | Stationary crate down the roadway | `debris` | `20, 25` | `obj_1792490616740`, `obj_1792494562696` | 5, 1 | **107** | High |
| **Crate E** | Small obstacle in center road | `debris` | `20` | `obj_1792490613825` | 4 | **108** | High |
| **Distant Pedestrian**| Small figure in background | `person` | `10, 15` | `obj_1792490406069`, `obj_1792490489252` | 6, 6 | **6** | High |

---

### 5.2 Physical Trajectory Allocation Table (`change-weather/scenario-1`)

| Physical Trajectory | Entity Description | Class | Ticks Present | Object IDs Involved | Current Track IDs | Proposed Reconciled Track ID | Confidence |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **Oncoming Car** | Sedan driving towards ego in opposing lane | `vehicle` | `00` to `20` | `obj_1792495226200` (00), `obj_..._5` (05, 10, 15, 20) | 4, 5, 5, 5, 5 | **5** | High |
| **Sidewalk Pedestrian** | Pedestrian walking towards camera on right | `person` | `00` to `25` | `obj_..._3` (00), `obj_..._4` (05), `obj_..._6` (10), `obj_..._8` (15), `obj_..._9` (20), `obj_..._10` (25) | 3, 4, 6, 8, 9, 10 | **3** | High |
| **Left Sidewalk Entity** | Distant prop / figure on far left curb | `person` | `35` | `obj_1792495448698` | 3 | **35** (or exclude) | **Adjudication Required** |

---

## 6. Cases Requiring Human Adjudication Prior to Modification

While the nine trajectory discontinuities have clear physical resolutions, the following **five specific items require explicit human reviewer sign-off**:

1. **Duplicate AI Proposal Boxes in `running-pedestrian/scenario-1`:**
   In Frames 00, 01, 02, and 03, both the manual draft box and an AI proposal box exist on the exact same person:
   - **Frame 00 (Tick 00):** Manual box `obj_1792489719606` (`[629, 529, 670, 561]`) vs AI proposal `obj_1792494631470_7_1` (`[650, 550, 669, 573]`).
   - **Frame 01 (Tick 05):** Manual box `obj_1792490217724` (`[1073, 527, 1121, 598]`) vs AI proposal `obj_1792494651746_8_1` (`[1087, 528, 1109, 596]`).
   - **Frame 02 (Tick 10):** Manual box `obj_1792490356024` (`[1103, 522, 1141, 619]`) vs AI proposal `obj_1792494668597_10_1` (`[1109, 530, 1129, 605]`).
   - **Frame 03 (Tick 15):** Manual box `obj_1792490478756` (`[1043, 550, 1064, 609]`) vs AI proposal `obj_1792494683431_13` (`[1051, 562, 1070, 600]`).
   *Adjudication Question:* Should the duplicate AI proposal box be removed (recommended, preserving the human draft), or should the two boxes be merged/averaged? Retaining both boxes violates tracking evaluation rules because the schema cannot assign the same track ID twice in one frame.

2. **Distant Figure vs Prop in Frame 27 (Tick 35) of `change-weather/scenario-1`:**
   Box `obj_1792495448698` at `[516, 529, 553, 592]` was added manually as `person`. Faster R-CNN classified it as a fire hydrant.
   *Adjudication Question:* Confirm whether this is a distant pedestrian or roadside prop. If pedestrian, assign dedicated Track ID `35`. If prop, exclude from target evaluation.

3. **Frame 38 (Tick 40) Overlapping Boxes in `running-pedestrian/scenario-1`:**
   Two pedestrian boxes overlap at `[1121, 534, 1150, 592]` (Track 19) and `[1121, 518, 1165, 604]` (Track 20).
   *Adjudication Question:* Confirm whether this represents two closely walking people or a duplicate annotation on one person.

---

## 7. Recommendation for the Smallest Valid Correction

To minimize disruption while restoring tracking validity:
1. **Approval Phase:** The human reviewer reviews the visual review composites in `visual_review/` and approves:
   - The proposed track ID mappings in Section 5.
   - The adjudication decisions in Section 6 (specifically, retiring the 4 duplicate AI proposal boxes).
2. **Execution Phase (Next Turn):**
   - Apply the approved track ID mappings in a single atomic write to `working_human_review_pilot_annotations.json`.
   - Recompute and document the new file hash.
   - Run verification tests to guarantee zero schema violations, zero duplicate track IDs per frame, and 100% trajectory class consistency.
3. **Tracking Evaluation:** Keep tracking metrics frozen until this approval is received.

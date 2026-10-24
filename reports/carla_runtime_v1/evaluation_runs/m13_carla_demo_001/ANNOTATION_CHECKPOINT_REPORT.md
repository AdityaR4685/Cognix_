# Cognix M13 — Annotation Checkpoint Report

**Execution Timestamp**: 2026-10-21T00:19:00+05:30
**Repository**: `C:\Users\Aditya\Cognix2_recovered`
**Git Branch**: `gat-testing-complete-checkpoint`
**Git HEAD**: `b4606b91ef2fe77719bdbe5c02ec501bd184a7f8`
**Preservation Status**: Verified Clean (`git diff --stat` empty; zero Git writes, zero branch changes)

---

## 1. Executive Summary & Forensic Safeguards

Following the human review and save operations conducted by reviewer **Aditya** across both Annotation Studios:
- **Anomaly Pilot Studio** (`http://127.0.0.1:5050/`)
- **Normal Control Studio** (`http://127.0.0.1:5051/`)

An immediate forensic backup and read-only audit of both saved queues was executed before any subsequent demonstration tasks.

### Key Findings
1. **User Save Verification**: Both working queues received multiple confirmed HTTP POST saves from the browser interface, updating both files on disk.
2. **Backup Verification**: Byte-for-byte timestamped copies were created in `reports/carla_runtime_v1/evaluation_runs/m13_carla_demo_001/backup_annotations/` and verified with matching SHA-256 digests.
3. **No Redo Required**: The reviewer's saved work is fully preserved on disk.
4. **Preservation of Exclusions**: All 5 retired duplicate AI proposals and all 40 ego-hood exclusions remain completely excluded.
5. **Review-State Schema Analysis**: The web client's `confirmFrameBtn` correctly assigned `reviewer_id = "Aditya"`, `confirmed_at_utc = <ISO-8601>`, and object-level `confirmation_status = "human_confirmed"`, but left the frame-level string `review_state` set to `"unreviewed"`. In strict adherence to the preservation protocol, **no metadata was fabricated or retroactively overwritten**. The actual saved content is preserved and documented exactly as persisted.

---

## 2. Cryptographic Source & Backup Hash Ledger

| Dataset Package | Source File Path | Backup File Path | File Size | Verified SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Pilot (Reconciled)** | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_pilot_annotations_reconciled.json` | `reports/carla_runtime_v1/evaluation_runs/m13_carla_demo_001/backup_annotations/working_human_review_pilot_annotations_reconciled_20261020_184215.json` | 49,400 bytes | `56457c5927e65f5d7e871423f2d2693dc6d02c419b931052efd22c4274ace481` |
| **Normal Control** | `reports/carla_runtime_v1/evaluation_runs/m9_annotation_readiness_001/working_human_review_normal_control_annotations.json` | `reports/carla_runtime_v1/evaluation_runs/m13_carla_demo_001/backup_annotations/working_human_review_normal_control_annotations_20261020_184215.json` | 19,880 bytes | `08a89ab430d35d0fd5a38fe206b271caf05560f56beaf01a133e0a23bc6f6e3a` |

---

## 3. Anomaly Pilot Package Audit (40 Frames)

### Dataset Overview
- **Total Frames**: 40
- **Total Objects**: 69 (decreased from 76 in pre-review candidate)
- **Frames with Objects**: 31 / 40
- **Unique Track IDs**: 26
- **Invalid Bounding Boxes**: 0 (all $x_1 < x_2$, $y_1 < y_2$, strictly non-negative)

### Reviewer Corrections Detected
The human reviewer actively edited frames 0, 5, and 10 of `test/anomaly/Town01/running-pedestrian/scenario-1`, deleting 7 spurious background boxes:
- **Frame 0 (Tick 0)**: Removed `obj_1792489803625` (Track 105, `road_debris_or_prop` background box). Remaining 4 objects confirmed.
- **Frame 5 (Tick 5)**: Removed `obj_1792490233877` (Track 105), `obj_1792490251582` (Track 104), `obj_1792490271605` (Track 106). Remaining 3 objects confirmed.
- **Frame 10 (Tick 10)**: Removed `obj_1792490366164` (Track 106), `obj_1792490381878` (Track 105), `obj_1792490442869` (Track 104). Remaining 5 objects confirmed.

### Class Breakdown
- `person`: 40 (58.0%)
- `vehicle`: 19 (27.5%)
- `road_debris_or_prop`: 10 (14.5%)

### Provenance Breakdown
- `unspecified` (original manual drafts): 24 (34.8%)
- `ai_accepted`: 20 (29.0%)
- `human_corrected`: 18 (26.1%)
- `human_added`: 7 (10.1%)
- Total AI-derived fraction: $(20 + 18) / 69 = 55.1\%$.

### Ledger Verification
- **Retired Duplicates**: All 5 records in `duplicate_retirement_ledger.json` remain absent from the saved dataset.
- **Ego-Hood Boxes**: All 40 excluded ego-hood boxes in `ego_hood_exclusion_ledger.json` remain absent from the saved dataset.

---

## 4. Normal Control Package Audit (10 Frames)

### Dataset Overview
- **Total Frames**: 10
- **Total Objects**: 32 (increased from 0; reviewer actively labeled objects across all 10 frames)
- **Frames with Objects**: 10 / 10
- **Unique Track IDs**: 9
- **Invalid Bounding Boxes**: 0

### Frame-by-Frame Breakdown
| Frame Index | Scenario Tick | Reviewer ID | Confirmation UTC Timestamp | Object Count | Object Details (Track, Class, Provenance) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 0 | Aditya | 2026-10-20T18:32:26Z | 2 | T1 (vehicle, ai_accepted), T2 (person, human_added) |
| **1** | 5 | Aditya | 2026-10-20T18:32:34Z | 4 | T1 (bike, indep), T2 (vehicle, indep), T3 (vehicle, ai_accepted), T4 (person, human_added) |
| **2** | 10 | Aditya | 2026-10-20T18:32:40Z | 4 | T1 (vehicle, human_added), T2 (vehicle, ai_accepted), T3 (bike, human_added), T4 (person, human_added) |
| **3** | 15 | Aditya | 2026-10-20T18:32:47Z | 3 | T1 (vehicle, human_added), T2 (bike, human_added), T3 (vehicle, ai_accepted) |
| **4** | 20 | Aditya | 2026-10-20T18:34:06Z | 4 | T1 (person, human_added), T2 (vehicle, human_added), T3 (bike, human_added), T4 (vehicle, ai_accepted) |
| **5** | 25 | Aditya | 2026-10-20T18:35:15Z | 4 | T1 (person, human_added), T7 (vehicle, human_corr), T3 (vehicle, ai_accepted), T8 (bike, human_added) |
| **6** | 30 | Aditya | 2026-10-20T18:36:47Z | 4 | T1 (person, human_added), T2 (bike, human_added), T3 (vehicle, human_added), T4 (vehicle, ai_accepted) |
| **7** | 35 | Aditya | 2026-10-20T18:38:08Z | 3 | T12 (person, ai_accepted), T13 (bike, human_added), T2 (vehicle, ai_accepted) |
| **8** | 40 | None | None | 2 | T5 (person, ai_accepted, pending_review), T3 (vehicle, ai_accepted, pending_review) |
| **9** | 45 | Aditya | 2026-10-20T18:38:51Z | 2 | T4 (vehicle, ai_accepted), T5 (person, human_added) |

### Class Breakdown
- `vehicle`: 16 (50.0%)
- `person`: 9 (28.1%)
- `bicycle_or_motorcycle`: 7 (21.9%)

### Provenance Breakdown
- `human_added`: 17 (53.1%)
- `ai_accepted`: 12 (37.5%)
- `independent_human`: 2 (6.3%)
- `human_corrected`: 1 (3.1%)

---

## 5. Review-State Discrepancy & Preservation Integrity

### Discrepancy Description
In `annotation_app.py`, the JavaScript handler for the `Confirm & Next Frame` button executes:
```javascript
currentFrameData.reviewer_id = revId;
currentFrameData.confirmed_at_utc = new Date().toISOString();
currentFrameData.objects.forEach(o => {
  o.confirmation_status = "human_confirmed";
});
```
However, it does not assign `currentFrameData.review_state = "confirmed"`. As a result, the frame-level string property in both JSON files remains `"unreviewed"`, even though:
- `reviewer_id` is set to `"Aditya"`.
- `confirmed_at_utc` contains valid UTC timestamps.
- All enclosed objects are tagged with `confirmation_status = "human_confirmed"`.

### Integrity Policy Applied
Per preservation requirements:
> *"If saved content exists but review-state metadata is missing or inconsistent, preserve the labels and report the discrepancy. Do not fabricate confirmation metadata or timestamps."*

We strictly preserve the exact saved files without fabricating metadata. The discrepancy is fully recorded here.

---

## 6. Scientific Boundaries & Evaluation Policy

### Dual Package Separation
The **Anomaly Pilot** (40 frames, 69 objects) and **Normal Control** (10 frames, 32 objects) packages must remain strictly separated in all downstream workflows and reports.

### Benchmark Limitations (Circular Evaluation Warning)
- Both datasets incorporate significant AI-assisted proposals ($55.1\%$ in Anomaly, $40.6\%$ in Normal Control).
- **CRITICAL**: Neither dataset can be cited as an independent, double-blind ground truth benchmark for evaluating the Faster R-CNN MobileNetV3 detector that generated those proposals. Evaluating a model against its own accepted proposals constitutes severe circular self-confirmation bias.
- **Allowed Uses**:
  - Replay demonstrations and qualitative pipeline visualization.
  - End-to-end telemetry and escalation latency profiling.
  - Safety hazard classification smoke tests.
- **Forbidden Uses**:
  - Official AP50 / MOTA / MOTP scientific claims for the perception model.
  - Official safety compliance certification.

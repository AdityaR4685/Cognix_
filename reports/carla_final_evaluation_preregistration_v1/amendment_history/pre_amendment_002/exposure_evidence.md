# Historical TEST exposure evidence — Amendment 001

This is a documented pre-TEST-access protocol correction based on recovered historical evidence supplied by the user. No official TEST payload, archive, directory, metadata inventory, image, feather table, mask or sensor sample was accessed in this amendment. Temporary probe files had been deleted afterward; they were not recreated or reopened.

The recovered evidence establishes exactly these known official TEST exposures:

| Exact official scenario ID | Historical exposure extent | Classification | Final role exclusion |
|---|---|---|---|
| `test/anomaly/Town01/change-weather/scenario-1` | Complete: the main sequential probe captured the scenario completely; earlier 64 KB / 8 MB probes also inspected its beginning | SCHEMA_EXPOSED_DIAGNOSTIC_ONLY | Both FINAL_CONFORMAL_CAL and FINAL_EVALUATION, entire scenario |
| `test/anomaly/Town01/change-weather/scenario-10` | Partial: the same main probe continued into this scenario and reached 93 RGB frames | SCHEMA_EXPOSED_DIAGNOSTIC_ONLY | Both FINAL_CONFORMAL_CAL and FINAL_EVALUATION, entire scenario |

Main probe: official `carlanomaly-base-test.tar.gz`; inclusive byte range **0–125,829,119** (125,829,120 bytes); **HTTP 206**; sequential gzip/tar parsing. Scenario-1 was complete; scenario-10 was partial. Do not infer specific frame filenames, additional modalities for the partial capture, earlier probe byte ranges, or other scenario IDs from those counts.

`recovered_historical_evidence.json` records the recovered facts and their provenance as human-provided historical evidence. Existing source documentation independently records TEST schema inspection for GNSS/IMU/labels/segmentation and the all-False change-weather/scenario-1 observation-label pattern. The earlier report relied on those incomplete source summaries and incorrectly left the town unresolved and omitted scenario-10. The recovered evidence supersedes that conclusion. Synthetic unit fixtures remain synthetic and establish no additional official exposures.

The historical **file-level ledger remains incomplete**. No evidence presently establishes any additional official TEST scenario IDs; none are claimed or inferred. Exactly the two known IDs above replace the provisional cross-town quarantine. Both complete and partial exposure disqualify the whole scenario from both final roles; touched ticks are never split off to rehabilitate the remaining ticks as untouched.

The full original preregistration, including its manifest and detached hash, is preserved in `amendment_history/pre_amendment_001/`. Its manifest SHA-256 is `2e7c20eb2d1a40367c6568027ddaea9e6650b80bee7d57770d94a644772fd6ab`. `amendment_record.json` documents this correction and `amendment_integrity.json` records exact byte changes. Scientific methods, seeds, metrics, thresholds, conformal construction, partition seed/ratio/stratification, model/checkpoint bindings and generic COGNIX remain unchanged. No metadata inventory, partition, prediction, training, fitting or tuning occurred. This amendment stops after resealing.

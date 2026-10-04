# COGNIX CARLA fresh-split final conformal calibration — protocol v4

READY_FOR_SEPARATELY_AUTHORIZED_FRESH_SPLIT_FINAL_CONFORMAL_CAL_ATTEMPT_003

Offline preparation only. Readiness does not authorize TEST access. Attempt003 has not executed. ATTEMPT_NUMBER=003, EXECUTION_PROTOCOL_VERSION=v4, OUTPUT_GENERATION=attempt003_v2. No Attempt004 exists here.

SCORER_AND_FEATURE_RULE_IDENTICAL_TO_SEALED_V3; FORMAL_CALIBRATION_PARTITION_AND_DERIVED_CONFORMAL_RANK_AMENDED.

V3 is the sealed, unexecuted precursor fixing the scorer and segmentation OOV rule before the fresh partition. It is superseded only as the execution/calibration protocol. Its historical OOV amendment record is preserved byte-for-byte for scorer provenance; that record's old125-recompute proposal is superseded by fresh_split_amendment_record.json. V1/v2/v3 and both failed-attempt records remain unchanged.

Original CAL125 were exposed during failed attempts and a post-access OOV amendment followed. None of the92 partial score values was used or reused. The original125 are permanently PRIOR_EXPOSED_CAL, opaque and retired from formal calibration/evaluation/tuning/model comparison. Recomputing them does not restore freshness. Class32 semantics UNKNOWN.

The original500-EVAL pool has SCIENTIFIC_PAYLOAD_DECODE=0 according to preserved ledgers/code gates and current human attestation. Opaque EVAL bytes previously transited/decompressed; this is not a no-byte-transport claim or a global access proof. A preregistration bound the frozen v3 scorer before exactly one secrets.randbits(128) seed and one PCG64 permutation of opaque sorted scenario IDs. First100 -> fresh CAL; remaining400 -> final EVAL. No balancing, stratification, semantic ID parsing, seed search/redraw, composition inspection or outcomes. Seed permanently spent. Verification checks the stored bijection without RNG replay.

Four immutable stream roles total627: FRESH_CAL100; FINAL_EVAL_OPAQUE_DISCARD400; PRIOR_EXPOSED_CAL_OPAQUE_DISCARD125; HISTORICAL_EXCLUDED_OPAQUE_DISCARD2. Only fresh CAL enters Image/Feather/labels/features/UQ/models. Both scanner and processor gate it. Prior CAL and exclusions are separate provenance classes. Announced body sizes and completed opaque discard chunks have separate counters.

Frozen segmentation: uint8 2-D/3-D; channel0 first; IDs0..28 unchanged; every ID29..255 -> existing Other22; unchanged29-bin extractor. Camera/IMU, all15 states/checkpoints, all fitted parameters, graph/UQ/normality/labels are identical to v3. Runtime pins/settings are identical: Python3.12.13, NumPy2.0.2, Pillow12.3.0, PyArrow25.0.1, pandas2.2.3, PyTorch2.10.0+cu128, CUDA12.8, cuDNN91002, exactly one Tesla T4. Local CPU tests do not certify GPU numerics.

Score=1-P_t(Y_t), scenario max over synchronized t>=1; alpha=.05; augmented100 scenario scores plus infinity; k=96; inclusive <=; one independent pooled threshold per fixed scorer. Full archive size/hash and exact completion of all100 fresh CAL are required before thresholds. No old scores/thresholds, model selection, subgroup tuning, EVAL metrics or calibration method ranking.

Validity is conditional on scorer freeze, scientifically undecoded original500-EVAL pool, one uniform scenario assignment and no outcome adaptations. It is finite-catalogue marginal over assignment and a uniformly chosen held-out scenario, not conditional on a realized seed/split/calibration sample. No subgroup/town/type, future-distribution/iid-whole-dataset, timestep independence, joint15-scorer or all400-EVAL simultaneous coverage claim. The specified seeded PCG64 implements the intended uniform-assignment premise; no independent randomness theorem for the PRNG is asserted.

Verify the detached ZIP hash before extraction into /kaggle/working/cognix_cal_bundle_v4. Notebook defaults False, runs preflight separately and never installs/downloads automatically. Offline preflight:

    python -B offline_checks.py runner.py --preflight

Only later explicit human authorization may enable --execute-authorized-final-cal. Output is fixed to /kaggle/working/cognix_final_conformal_cal_attempt003_v2 and existing output refuses. Marker before network; exactly one full sequential GET, Accept-Encoding identity, no redirects/Range/retry/resume, no raw archive retained. Expected compressed size91538225599 and full SHA267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a. Failures emit sealed access/failure/result evidence and forbid partial thresholds.

FRESH_PARTITION_SHA256SUMS binds the irrevocable partition and preregistration. BUNDLE_SHA256SUMS covers all bundle files except itself, detached seal, ZIP and detached ZIP hash. ZIP includes both bundle seals and excludes itself; no circular dependency. Historical evidence scripts are not verification targets. Offline guards are process-level only; no OS-wide capture is claimed. Review and exact human staging commands are in the unsealed reports/carla_final_conformal_cal_v4_review.json. Nothing was staged/committed/pushed.

# COGNIX CARLA final conformal CAL v3 — preparation only

READY_FOR_SEPARATELY_AUTHORIZED_FINAL_CONFORMAL_CAL_ATTEMPT_003

This readiness means offline preparation passed. It does not authorize TEST access and does not mean Attempt 003 occurred. No scientific runner execution, network request, TEST payload access, fitting, tuning, staging, commit or push occurred during preparation.

Attempt 002 is INCOMPLETE_NO_RETRY: 92/125 CAL completed, 0 EVAL decoded, one GET, 62,739,644,426 compressed bytes, 455 roots, no final thresholds. Its immutable historical evidence is sealed separately. Never retry/resume Attempt 002 or reuse its partial CAL scores. Class 32 semantic identity, failing scenario/tick and unrecorded properties remain UNKNOWN.

V3 explicitly amends segmentation preprocessing after the failure: accept uint8 2-D/3-D, select channel 0 for 3-D, preserve IDs 0..28, collapse every ID >28 to existing CARLA Other=22, then call the byte-identical 29-bin extractor. No special cases, frequencies, outcomes, class meaning, labels or membership inform mapping. The 92 partial scores were not parsed or used for repair selection. This is a post-failure scientific compatibility amendment, not a v2/v3 science-identity claim. It can change OOV scores.

CAL/EVAL membership (125/500), exclusions, all 15 checkpoints/state hashes, fitted parameters, Camera/IMU features, graph/weights/UQ, label semantics, conformal definition and runtime pins are unchanged. Scores are 1-P_t(Y_t), scenario max over t>=1, alpha=.05, augmented 125 scores plus infinity, k=120, inclusive <=, one pooled threshold per scorer. No method ranking or EVAL metrics. See unchanged_scientific_bindings.json for every hash and segmentation_oov_amendment.json for the declared change.

Runtime lock: Python 3.12.13, NumPy 2.0.2, Pillow 12.3.0, PyArrow 25.0.1, pandas 2.2.3, PyTorch 2.10.0+cu128, CUDA 12.8, cuDNN 91002, exactly one visible Tesla T4. Lock and requirements files are byte-identical to v2; v3 now explicitly enforces the existing pandas pin and exact Tesla T4 name. Local CPU tests do not certify those GPU numerics.

Attach the upload ZIP as a Kaggle input and check the externally recorded ZIP SHA256 before extraction. Extract into /kaggle/working/cognix_cal_bundle_v3. Notebook execution defaults to False. Preflight is offline:

    python -B offline_checks.py runner.py --preflight

Only a later separately authorized human action may use the existing --execute-authorized-final-cal switch. No automatic execution is provided by preparation. Output is fixed to /kaggle/working/cognix_final_conformal_cal_attempt003_v1; existence refuses execution. A permanent attempt marker precedes access; exactly one full byte-zero sequential GET, no redirects, Range, retry, resume or raw-archive retention. Full expected size 91538225599 and SHA256 267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a and complete 125-CAL membership are required before thresholds. Failures seal access/failure/result evidence and forbid partial thresholds. Every CAL scenario is recomputed; no Attempt-002 partial scores are read.

EVAL/excluded bodies physically transit and decompress but remain opaque, never reaching image/Feather/label/feature/model decoders. Historical opaque_body_bytes_discarded_by_role counts CAL bytes before decode/discard decisions; its CAL value is not entirely opaque/discarded. EVAL/EXCLUDED values are opaque discard. Original streaming code/ledger is unchanged.

Verification uses only synthetic fixtures and frozen TRAIN artifacts, with process-local Python audit hooks. No OS-wide capture is claimed. Historical evidence/*.txt and evidence/v2_scientific_bindings.json retain explicitly historical provenance and are never execution targets. BUNDLE_SHA256SUMS covers all bundle files except itself, detached seal, upload ZIP and detached ZIP digest; ZIP includes manifest and its detached seal, and excludes itself. v1/v2 remain byte-identical.

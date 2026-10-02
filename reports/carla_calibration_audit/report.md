# Initial Repository / Artifact State

Initial Git status contained 13 intentional untracked real-adapter files/tests and no tracked changes. `initial_state.json` inventories and hashes 132 repository files and inventories 12,054 probe artifacts by size and modification time. No AGENTS.md was found in the workspace/parent search. Existing probe scripts and JSON are directly in the probe root; an `analysis/` directory is absent.

Preserved: 3,001,024,512-byte TRAIN prefix; complete scenario-1 and scenario-10 under `dataset/train/Town01/`; `cache_a`, `cache_b`, `cache_v2_a`, `cache_v2_b`; every old analysis/protocol review artifact. v2 NPZ size is 6,422,686 bytes and its source content hash is `219f23a2fd442b9a9fd1f66e4c5282af40cfcfb4370ead627ffe21b4e5e71e0f`. No cache rebuild was needed.

# Separation Diagnosis

Production fits were reproduced from `cache_v2_a` with bootstrap seed 42, five Mahalanobis members, TRAIN_NORMAL scenario-1 and CAL_NORMAL scenario-10. Each modality uses all 2,999 CAL normal observations plus its own modality-specific pseudo rows. The saved member matrices permit reproduction without raw image access.

| Agent | Normal mean-score range | Pseudo mean-score range | Overlap | min(normal) − max(pseudo) | Best threshold | Violations | Geometry |
|---|---|---|---|---|---|---|---|
| Camera | [1.63723118e-11, 0.620830214] | [6.18703463e-36, 3.59674257e-06] | [1.63723118e-11, 3.59674257e-06] | -3.5967262e-06 | 8.070621e-07 | 22 | overlapping |
| Seg | [4.78632631e-25, 0.935894069] | [0, 0.72283724] | [4.78632631e-25, 0.72283724] | -0.72283724 | 0.0326894742 | 159 | overlapping |
| GNSS | [3.60719997e-05, 0.944117627] | [0, 0] | none | 3.60719997e-05 | 1.80359999e-05 | 0 | completely_separated |
| IMU | [7.65347924e-05, 0.986814204] | [0, 0.671071552] | [7.65347924e-05, 0.671071552] | -0.670995017 | 5.94622091e-05 | 139 | overlapping |

Threshold rule is normal iff score > threshold; ties count as pseudo. The minimum-violation threshold is selected only for reporting, never deployment. Quasi separation means exact shared class boundary with a nonconstant score support. None of the four real mean-score distributions is quasi separated.

| Agent | Pooled member normal range | Pooled member pseudo range | Gap | Best threshold | Violating member entries |
|---|---|---|---|---|---|
| Camera | [1.43554067e-12, 0.636148786] | [2.20655201e-37, 5.50046745e-06] | -5.50046602e-06 | 1.26256614e-06 | 113 |
| Seg | [1.7668546e-30, 0.942742555] | [0, 0.736562491] | -0.736562491 | 0.0275783093 | 810 |
| GNSS | [2.61999446e-06, 0.946391205] | [0, 0] | 2.61999446e-06 | 1.30999723e-06 | 0 |
| IMU | [6.94369331e-06, 0.987108783] | [0, 0.684721711] | -0.684714767 | 6.03481805e-05 | 705 |

Pooled-member geometry is separate from row weighting. The likelihood always weights observations equally; member entries here are a diagnostic only. Full overlap intervals, minima, maxima and raw matrices are retained in `separation_results.json` and the four NPZ files.

| Fixed raw slope | Camera NLL | Seg NLL | GNSS NLL | IMU NLL |
|---|---|---|---|---|
| 1 | 0.513632893 | 0.322035046 | 0.420135978 | 0.416287018 |
| 2 | 0.499833512 | 0.276180900 | 0.335400827 | 0.332251516 |
| 5 | 0.468565087 | 0.192116332 | 0.194863201 | 0.206608419 |
| 10 | 0.435337206 | 0.159533338 | 0.122768772 | 0.151439510 |
| 20 | 0.395184925 | 0.187478825 | 0.085626781 | 0.124694455 |
| 50 | 0.329076772 | 0.336686495 | 0.056913117 | 0.128636248 |
| 100 | 0.270648536 | 0.460946666 | 0.045144531 | 0.190409628 |
| 200 | 0.206331667 | 0.560331160 | 0.037446325 | 0.332106526 |
| 500 | 0.125745583 | 0.706180335 | 0.029932726 | 0.672401526 |
| 1000 | 0.081085637 | 0.843311775 | 0.024785745 | 0.812327802 |
| 2000 | 0.054859044 | 0.988807496 | 0.020025903 | 0.864184518 |
| 5000 | 0.041498623 | 1.117931089 | 0.014027395 | 0.886100782 |

For each fixed slope, only the intercept is optimized using a deterministic grid/multiple-basin bounded refinement. This is an auditable numerical search, not a proof of global intercept optimality. No deployed parameter is chosen from this curve.

GNSS is strictly separated even at the individual-member level: choose any threshold between maximum pseudo member score 0 and minimum normal member score 2.619994457e-6. Along b = −a·threshold, every normal member probability tends to 1 and every pseudo member probability tends to 0. The mathematical unregularized NLL has infimum 0, unattainable at finite a,b: no finite MLE. Numerical q clipping introduces an artificial floor, not a scientific optimum.

Camera is overlapping, not completely or quasi separated. Its curve is still descending at 5,000. This is evidence of an effectively separable regime and a potentially unbounded direction, but it does not establish that a finite MLE cannot exist beyond the explored slopes. The current production solution is demonstrably nonstationary. Camera therefore remains unvalidated, with suspected separation explicitly distinguished from proof. No larger initialization or further scenario-based tuning was performed.

# Calibration Validity Rule

`optimizer_converged` retains the original numerical termination result. `finite_optimum` is a conservative acceptance flag for a finite stationary candidate, not a theorem of global MLE existence. Acceptance requires numerical convergence, finite parameters/NLL, max absolute gradient in (log standardized slope, intercept) coordinates < 1e-8, strictly positive symmetric local Hessian eigenvalues, no pooled strict/quasi separation, and no materially better fixed/relative-slope diagnostic basin. NLL differences use 1e-10 numerical resolution; there is no slope threshold. Tail decreases at 1,000→2,000→5,000 plus an observed lower profile direction flag suspected separation for overlapping scores.

Invalid mappings retain parameters, NLL, all curves, geometry and Jensen diagnostics. Default calibrator probability methods and real-agent prediction/UQ methods raise. Explicit `diagnostic=True` permits audit-only values; predictions carry validity/status and diagnostic metadata. Scientific handoffs must store null probability/UQ for invalid agents.

| Agent | Optimizer converged | Gradient max abs | Minimum curvature | Raw slope | Finite candidate accepted |
|---|---|---|---|---|---|
| Camera | True | 0.202183395 | -0.0100398089 | 54.1207331 | False |
| Seg | True | 2.20416285e-16 | 0.00579886671 | 10.2775374 | True |
| GNSS | True | 0.127166433 | -0.000139521431 | 46.934635 | False |
| IMU | True | 7.35356252e-12 | 0.000257112842 | 30.9015051 | True |

# Calibration Implementation Changes

The objective is unchanged: p_it = sigmoid(a·s_it+b), q_i = mean_t p_it, minimize mean_i BCE(y_i,q_i). All-member mean/std standardization converts A=exp(u) to raw a=A/std and b=B−A·mean/std. One-class scores remain bounded [0,1]. Sigmoid arguments remain clipped to [−500,500], objective q to [1e-12,1−1e-12]; neither is a slope cap.

The original optimizer remains fixed-start damped Newton with analytic gradient, finite-difference Hessian, damping 1e-6, backtracking up to 40 halvings, at most 300 iterations/start and starts A=1,10,2. Termination requires NLL change <1e-12 and maximum accepted step <1e-10. It never checked stationarity: Camera/GNSS terminate with gradients 0.202/0.127 and negative local curvature. The new audit blocks that result. Tolerances, starts and objective were not loosened or retuned. Failed refits clear old readiness; fitting a new normality model invalidates the previous calibrator.

New cache schema v2 writes `cal_target_normal`; legacy caches are migrated in memory only, preserve source identity/hash and expose explicit target-name migration metadata. Existing cache bytes are not rewritten. Jensen max/median/p90 remains descriptive.

# CarlAnomaly Probability Semantics

`prob_normal`, `target_normal`, and `cal_target_normal` replace real-adapter safety names. Prediction metadata states exactly: `P(target=normal) under TRAIN-derived clean-vs-pseudo calibration`. The generic confidence field remains unchanged and equals the normality prediction with explicit metadata. This depends on pseudo recipe sampling and class balance. It is not physical safety, safe-driving probability, a guaranteed official-anomaly posterior, or calibration on official anomaly types. The scalar calibrator is a historical mathematical reference, not the production path.

# UQ Status

The entropy decomposition is unchanged: total=H(mean p_t); aleatoric=mean H(p_t); epistemic=max(0,total−aleatoric). Near-zero probability-space epistemic remains an observed negative result of this CARLA feature/bootstrap setup, not a universal COGNIX result. Score standard deviation is a separate diagnostic. Previous invalid Camera/GNSS mappings can be inspected only diagnostically; their probability-space UQ cannot now be treated as scientifically validated.

# Focused Tests

Final focused result: 313 passed, 0 failures, including 17 added tests. Historical separable fixtures explicitly request diagnostic evaluation. New mathematical fixtures test finite acceptance, complete/quasi separation, default rejection, deterministic/duplicate/permutation behavior, bounds/direction, both-class validation, constant/inverted non-identifiability, lifecycle invalidation and read-only legacy cache migration. The earlier schema assertion failure was corrected before final validation.

# Broad Regression

The exact requested broad suite ran under Python 3.12.10 with PYTHONHASHSEED=0: **482 passed, 0 failures**, up from 465 by 17 new tests. Runtime 94.12 seconds; 14 existing third-party matplotlib/pyparsing deprecation warnings. See `broad_tests.txt`.

# Two-Scenario Protocol Freeze

Scenario-1/scenario-10 are frozen as integration validation, protocol development and diagnostics. Their results must not select severities, slope caps, regularization, GAT hyperparameters, escalation thresholds or a split seed. Preserve all four caches and all old/new analyses. No more tuning on these scenarios.

# 10–20 Scenario Acquisition Plan

Measured: compressed prefix 3.001 GB already contains two complete scenarios and partial scenario-11; extracted scenario-1=1.337 GB, scenario-10=1.717 GB, total=3.054 GB. The two v2 cache builds together took 1,496.1 seconds: average 748 seconds per two-scenario build, about 374 seconds/scenario (6.23 minutes). Download logs show roughly 1.9–2.3 MB/s including scan overhead. Current C: free space was approximately 323 GB during this audit. Sizes below use decimal GB/MB.

| Resource | 10 total scenarios | 20 total scenarios |
|---|---|---|
| Compressed prefix total, extrapolated | 15–22 GB | 30–44 GB |
| Additional compressed data beyond existing 3 GB | 12–19 GB | 27–41 GB |
| Extracted raw data total, extrapolated | 15–25 GB | 30–50 GB |
| Compact normal + CAL pseudo cache, one pseudo/tick | 30–50 MB | 60–100 MB |
| Preprocessing, one production build | 1–2 hours | 2–4 hours |
| Additional network time at measured rate | 1.5–3 hours | 3.5–6 hours |
| Free-space reserve allowing staging and preservation | 60–80 GB | 120–160 GB |

These are conservative two-scenario extrapolations, not archive guarantees. Double preprocessing for a deliberate independent determinism build. No remaining archive index/size was fetched. Whole archive TRAIN scope is preserved. Next observed scenario is Town01/scenario-11 (partial RGB only); lexicographic ordering suggests scenario-12, ... scenario-19, scenario-2, scenario-20, but this remains an inference until headers prove the order. Town transitions and actual scenario counts are unknown.

Progressive prefix acquisition is practical at 10–20 scenarios if Range responses are checked and one running gzip/tar decoder processes appended bytes. Re-scanning the entire growing prefix after every ~268 MB chunk causes quadratic work and should be replaced later. Stage the expanded prefix separately so the original is preserved; validate complete RGB/Seg tick ranges and required sensor tables, finish at a proven next-scenario boundary, and never mistake a partial next scenario for completion. Select the first N complete TRAIN scenarios by archive order before metrics. Extraction/handcrafted preprocessing can remain local under the measured free-space/runtime budget. Revisit AWS only for storage pressure or materially slower/more extensive extraction; no cloud setup, transfer or purchase is part of this task.

# Pre-Registered FIT/CAL Design

Proposed, not executed: choose N=10 or N=20 before acquisition, take first N complete official TRAIN scenarios in archive order, canonicalize IDs, sort them, apply NumPy PCG64 seed 2026 permutation, take first max(1,round(0.25·N)) as CAL_NORMAL and the rest FIT_NORMAL. Ties use Python round-to-even: 10 gives 8 FIT/2 CAL (80/20); 20 gives 15 FIT/5 CAL (75/25). This deliberately retains the existing split rule and seed instead of optimizing a seed. Freeze the acquired list and manifest before fitting; the final 20-scenario split is formed once, not changed after viewing 10-scenario results. The two-scenario study stays a distinct frozen protocol. The current builder calls FIT rows TRAIN_NORMAL; the proposed artifact records FIT_NORMAL with an explicit name mapping. Generating FIT-derived graph pseudo rows is a later adapter extension, not implemented in this task.

One-class models use clean FIT only. Future graph pseudo examples originate from FIT only. The calibrator uses CAL clean + CAL-derived pseudo only. Whole scenarios, causal windows and every parent remain isolated across FIT/CAL. Record canonical ID, town, archive ordinal, complete tick coverage, partition, source hashes, feature/cache identity, recipe/default severity, seeds, window provenance and target scope. `proposed_protocol.json` contains the full manifest/schema proposal. TEST data influences nothing.

# Kaggle Handoff Design

Export compact features and fitted outputs only; never the raw archive or frames. Dictionary-encode scenario/town/recipe IDs; retain 65 float64 features, 20 member scores (five per agent), validity masks/status, nullable four-agent probabilities and entropy decompositions, scenario/tick/partition, constructed target and scope (per-agent labels refer only to the corrupted modality; graph-level any-pseudo labels are a distinct constructed target), pseudo parent/modality/recipe/severity/seed/window bounds, schema/protocol/code versions, hashes and complete calibration audit. Retain float64 scientific scores; optional float32 graph tensors are a separately versioned derivative. Invalid calibration probability/UQ is null, not zero or a diagnostic value masquerading as a validated probability.

With ~2,999 clean ticks/scenario and one pseudo per tick for both FIT graph training and CAL calibration, total rows are about 60k (10 scenarios) or 120k (20). Numeric payload is about 53/107 MB (65 features + 20 scores + 16 probability/UQ outputs, float64, plus ~80 bytes numeric provenance/validity). Budget 80–150 MB / 160–300 MB including manifests, identifiers and predictions; compression may reduce this but is not promised. Generating all seven recipes per tick instead increases row count fourfold from clean+pseudo to clean+7pseudo; budget ~0.3–0.6 GB / 0.6–1.2 GB. This is a sampling/storage option to pre-register later, not a change made now.

Later T4 workflow: verify compact artifact hashes/schema → construct graph samples → StandardGAT → EpistemicGAT on identical samples and paired seeds [101,202,303,404,505] → save checkpoints/configs → save per-seed metrics and row-keyed predictions. Use FIT-derived graph examples only; if graph validation is needed, register whole-scenario validation inside FIT before training. CAL remains probability calibration data. Graph settings and evaluation design require a separate preregistration; no GAT implementation/training occurred here.

# Domain-Agnostic Core Integrity

Final verification: all 12,054 prior probe files have unchanged size/mtime; all four caches match their preserved content hashes; every generic/frozen source and synthetic test matches its initial SHA-256; tracked Git diff is empty; bare import cognix loads no adapters. `integrity_results.json` lists the exact nine modified pre-existing files and one new test file. Generic APIs and entropy mathematics were not edited. Official TEST data and raw downloads were not accessed; unit tests exercise TEST gates using synthetic temporary fixtures only. No GAT, conformal, threshold, severity, regularization or slope-cap change; no commits, pushes, branch changes, cleanup or stash.

# Ready / Not Ready to Acquire More TRAIN Data

Ready to plan a separately authorized 10-scenario TRAIN acquisition after freezing the acquisition target and manifest. Not ready to treat Camera/GNSS probability outputs as scientifically calibrated or start final graph training. Expansion is data collection under the frozen rule, not a fix by calibration tuning.

# Recommended Next Step

Approve the 10-scenario acquisition manifest/rule in a separate request, then acquire only the additional official TRAIN prefix while preserving the original artifacts. Reassess calibration validity on the preregistered multi-scenario distribution. Stop here: no downloading or graph training.

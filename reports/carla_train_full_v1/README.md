# COGNIX Experiment 2A: FULL-TRAIN Expanded-Training Study

Preparation and preregistration only. No acquisition, fitting, pseudo generation, graph construction, simulator generation or performance evaluation is authorized or executed by this milestone. All material here was prepared offline on 2026-10-05 (Asia/Calcutta).

Baseline: `866637b1b73615af3178d5f242ce8b53e74d5115` (Seal successful CARLA final evaluation Attempt001). Experiment 1 is CLOSED. Historical reports and states remain immutable. The historical N=20 protocol remains a separate frozen experiment. No branch operation, staging, commit or push is part of this preparation.

## Why FULL TRAIN

H2A tests whether more official normal TRAIN data and environmental diversity improve transfer of agent-level normality evidence beyond the narrow N=20 distribution. The intentional intervention is N=20 versus **every complete official TRAIN scenario**, with fixed core methodology. N=60, town quotas, balancing by dropping scenarios, performance-based stopping and architecture redesign would change that question. FULL-TRAIN N and town count are unknown until complete verified archive traversal. No acquisition count is predicted here.

Experiment 1 established the historical N=20 pipeline and its scientific definitions; its GNSS calibrated probability/UQ channel was structurally invalid under the accepted model. Historical poor final generalization motivates the breadth question, but detailed old TEST results are not read or used as a design/tuning signal. Broader data is not assumed to repair the representations or pseudo-task mismatch.

## What changes and what stays fixed

The source expands to all complete official CARLAanomaly TRAIN scenarios in actual parsed archive order. After the inventory is frozen, sort canonical IDs, apply NumPy PCG64 seed 2026 once, take `n_cal=max(1,round(0.25*N))` with Python ties-to-even, and assign the first n_cal permuted IDs CAL_NORMAL and the rest FIT_NORMAL. The old 15/5 membership is never manually carried over. Freeze complete membership, town/environment summaries, source hashes and exact partition bytes/hash before fitting.

Keep the 12-tick causal window, Camera/Seg/IMU feature definitions, one-class/bootstrap scores, uncertainty decomposition, pseudo definitions/default severities and target semantics fixed. Active downstream modalities are Camera, Seg and IMU. GNSS probability/UQ is excluded; no repair or re-enablement is part of 2A. Compare NoGraph, StandardGAT and EpistemicGAT with paired seeds 101, 202, 303, 404 and 505. Keep every seed and negative result; no best-seed selection or prediction ensemble.

Outputs mean **P(target = normal)** under the TRAIN-derived constructed clean-versus-pseudo calibration task. They are not physical safety probabilities or guaranteed official anomaly posteriors. Good pseudo discrimination is a development diagnostic, not evidence of real anomaly detection.

## Offline reuse bindings

The local N=20 prefix exists at `C:/Users/Aditya/AppData/Local/Temp/carla_train_expand20/train_prefix_expand20.tar.gz`.

- Exact bytes: 26,205,017,360.
- Full-file SHA-256 measured this session: `015ef22f4312b70336f92214ac069979df9c57a42c26e0aec913863917961698`.
- Accepted next scenario: Town02/scenario-5; TAR header offset 28,207,053,312.
- N20 acquisition config, manifest and compute-reuse policy hashes match detached bindings, sealed-commit bindings and recorded cross-references. Seven scientific source files and three generic graph source files match accepted hashes.
- Historical source record: official TRAIN archive total 146,453,559,283 compressed bytes; quoted ETag `"6a200b7a-22194ff7f3"`. No new source request or identity check was made. No full-archive authoritative SHA-256 is recorded locally.

These checks authorize no reuse execution. Revalidate before later use. Any required failure means STOP, with no automatic repair, overwrite, redownload or substitution.

Partition-independent clean feature blocks are eligible only with exact source-member/scenario/code/schema/window/preprocessing bindings and no partition-dependent computation. New roles, all pseudo rows, one-class fitted state, calibrators, graph examples, checkpoints, thresholds and predictions must be recomputed. Prior TEST-derived scientific artifacts are forbidden development inputs. The reuse inventory assigns every preexisting repository artifact under reports/artifacts/results one whole-file category; mixed-cache clean projections require their own verification and new publication.

## Acquisition and resource recommendation

Later, if separately authorized, make an independent FULL-TRAIN staging copy of the verified prefix, reconstruct one live gzip/TAR decoder from byte zero and resume at compressed offset 26,205,017,360. Require HTTP 206, exact Content-Range, raw identity encoding, unchanged quoted ETag and unchanged total size. Never decompress an arbitrary later byte range independently or overwrite N20. A failed continuation cannot silently fall back to a fresh download; fresh acquisition requires separate authorization.

Completion requires the entire compressed stream, valid gzip trailers/EOF, complete TAR traversal, every accepted complete scenario inventoried and final source byte/hash records. The trailing partial scenario never counts. Only objective unreadable/corrupt required data can be excluded, with an evidence ledger and no model-based filtering. Budget interruption leaves an incomplete acquisition, never a reduced FULL-TRAIN target.

Measured free space: 264370925568 bytes (approximately 264.37 decimal GB). Planning from historical member bytes estimates full extracted raw members plus a separate compressed archive at 303580843050 bytes (approximately 303.58 GB), already exceeding current free space before reserve/overhead. This ratio estimate is not a true upper bound.

Recommend one separate retained compressed archive plus scenario-at-a-time raw verification/clean compact extraction. The admitted incremental budget is 182960781299 bytes (approximately 182.96 GB): the 146.45 GB archive, 8 GiB scenario temp cap, 4 GiB compact/provenance admission cap, 2 GiB decoder/IO allowance and 20 GiB reserve. Current local budget admission passes; disk must be remeasured at execution. Caps are fail-closed pause points, not permission to exclude large scenarios. No finite true raw/compact bound is known from current evidence.

Estimated clean compact size is 215063621 bytes from historical new-clean blocks, with a 2x planning allowance recorded; full scenario count is not inferred. Historical clean-extraction timing gives a rough 12.71-hour planning estimate, excluding replay, integrity work, future pseudo/fitting and training. Future runtime may differ substantially.

**Retain the full compressed source.** Camera/Seg pixel pseudo recipes and temporal sensor recipes require raw observations after the new partition is known; compact clean features alone are insufficient. After inventory/partition freeze and later pseudo authorization, replay the retained local archive scenario-by-scenario. Temporary raw data may be released only after exact reproduction/provenance is durable, using owned NEW Experiment-2 paths only. Never delete historical experiment evidence or unrelated data. No such cleanup occurs now.

## Three gates

1. **Data/integrity:** complete source traversal, readable required modalities, aligned/finite sensor data, source/member hashes, diversity report, frozen deterministic partition and zero TEST access. No performance threshold. Failure stops fitting.
2. **TRAIN-only scientific health:** authorized FIT-clean one-class fitting and CAL-clean plus CAL-derived pseudo calibration under unchanged validity audit. Each of Camera/Seg/IMU must receive VALID, INVALID_STRUCTURAL, INVALID_NUMERIC or INVALID_OTHER. Report score/probability saturation, member disagreement, UQ, separation and constant-output checks. Any active invalid channel stops before GAT; a repair is a separately registered Experiment 2B.
3. **Pre-final development:** separately frozen graph protocol and authorized TRAIN-derived three-method/five-seed development. Preserve fixed graph architecture/training rules; bind new whole-scenario graph membership and all new state/preprocessing/threshold/metric/aggregation/analysis/failure hashes before any final evaluation. Null or negative results remain reportable; EpistemicGAT winning is not a gate requirement.

The planned graph role rule retains the historical 20% FIT validation fraction and PCG64 seed 2027; its new membership must be frozen under the separate graph protocol before graph outcomes. This is supervised graph development conditional on upstream features fitted on all FIT scenarios, not independent full-pipeline evidence. CAL is reserved from graph development.

## Retrospective versus genuinely fresh evaluation

Protect all four old official TEST roles: 400 FINAL_EVAL, 100 fresh conformal CAL, 125 prior exposed CAL and 2 exclusions. None can influence training, calibration, features/representation/pseudo design, hyperparameters, thresholds, selection, early stopping, acquisition size or town choices.

Any later old-400 model comparison is descriptive **RETROSPECTIVE_LOCKED_OLD_TEST**, run only after a complete model/analysis freeze and separate authorization. It is never a fresh or untouched Experiment-2 test set. No tuning follows viewing those scores.

Prefer separately generated CARLA Simulator evidence. Later preregister a DEV-SIM family for protocol health checks and a disjoint FINAL-SIM seed/scenario specification, generation code/version/assets, fixed sizes, integrity rules and analysis. Generate FINAL-SIM after development protocol freeze and evaluate it once after model freeze. Keep final outputs unavailable to development; never relabel DEV as final. No simulator data is generated now. Concrete simulator sizes/seeds and any confirmatory material-effect margin require a separate outcome-blind protocol before new performance is observed, not post-hoc choice.

Report data-breadth, graph and epistemic-prior effects separately. FULL-TRAIN can improve all methods while the epistemic prior remains unsupported. Near-chance performance can show breadth alone was insufficient. Retain failures, negative results and limits of the constructed pseudo task. Any rescue of representation, calibration, recipe, threshold, architecture, seed or analysis after final access starts requires a new numbered experiment.

## Files and next authorization

The twelve files are initial_state.json, historical_bindings.json, full_train_preregistered_protocol.json, acquisition_plan.json, storage_runtime_plan.json, reuse_policy.json, development_gate_spec.json, evaluation_plan.json, risk_register.json, README.md, PREPARATION_SHA256SUMS and PREPARATION_SHA256SUMS.sha256.

initial_state.json contains the complete preexisting git-status and metadata-preservation snapshots plus the final verification receipt. PREPARATION_SHA256SUMS seals all ten preparation documents; its detached SHA-256 seals that list. The seal proves the exact preparation bytes, not completion of future gates.

Review this preparation before separately authorizing acquisition/staging/extraction. Fitting, graph work, DEV-SIM/FINAL-SIM generation and final/retrospective evaluation each require their later specified authorization and gates. A fresh-download fallback requires separate authorization. STOP here: zero network/TEST requests or TEST payload bytes, zero TRAIN acquisition and no scientific execution, staging, commit or push.

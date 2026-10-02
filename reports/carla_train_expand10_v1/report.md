# Frozen Acquisition Protocol

Acquisition config SHA-256: `e038ff835bd468a0894d37f23a922bacde39936a05a0bbc14147d7af9a25ca4a`. Frozen before the first network request: first 10 complete official TRAIN scenarios in parsed archive order, whole scenarios, PCG64 seed 2026, 8 FIT / 2 CAL, causal window length 12, unchanged recipe definitions/default severities, feature/code hashes and production calibration audit.

intermediate multi-scenario checkpoint, not final Experiment-1 calibration dataset; no final performance/generalization claims even if every mapping validates. Preferred later target is N=20, 15 FIT / 5 CAL, under separate authorization. No final performance/generalization claims or automatic graph work.

# Network / Prefix Integrity

Original prefix before/after: 3,001,024,512 bytes, SHA-256 `baffcf34a5d80ae2cf023afcd112a52f3dffab61d7107a245aa9bd0ccf798e6f`. Independent staging copy; no hardlink/symlink for the prefix. One replay of the staged original prefix reconstructed the live gzip/TAR state. Further bytes came through sequential HTTP Range streaming with 206/start/range/length/encoding checks and a stable server validator '"6a200b7a-22194ff7f3"'.

Downloaded 9.546 GB additional compressed data. Expanded prefix is 12.548 GB, SHA-256 `c565f5f416b264b7725c15e79d0fe7bee92e8439866647826d8c8e58afb02be0`. Remote TRAIN archive total advertised by Content-Range was 146.454 GB; the rest was not acquired. All selected members were completely consumed; the full archive gzip trailer was deliberately not fetched.

Stopping evidence: `{"kind": "subsequent_scenario_header", "next_scenario_id": "Town01/scenario-5", "tar_header_offset": 13534343168, "member": "train/Town01/scenario-5/rgb-front"}`. No scenario-ID prediction or performance-based selection. The parser stopped at that proven boundary; at most one 256 KiB compressed read chunk includes boundary read-ahead. No trailing scenario was extracted or classified complete. Logs: `C:\Users\Aditya\AppData\Local\Temp\carla_train_expand10/acquisition_log.jsonl`.

# 10 Complete TRAIN Scenarios

| Archive ordinal | Scenario | Partition | RGB/Seg frames | GNSS/IMU rows | Reused | First TAR offset |
|---|---|---|---|---|---|---|
| 1 | Town01/scenario-1 | FIT_NORMAL | 3000/3000 | 3000/3000 | True | 0 |
| 2 | Town01/scenario-10 | CAL_NORMAL | 3000/3000 | 3000/3000 | True | 1341529088 |
| 3 | Town01/scenario-11 | FIT_NORMAL | 3000/3000 | 3000/3000 | False | 3063567360 |
| 4 | Town01/scenario-12 | FIT_NORMAL | 3000/3000 | 3000/3000 | False | 4532770304 |
| 5 | Town01/scenario-13 | FIT_NORMAL | 3000/3000 | 3000/3000 | False | 5710794752 |
| 6 | Town01/scenario-14 | FIT_NORMAL | 3000/3000 | 3000/3000 | False | 7095044608 |
| 7 | Town01/scenario-15 | FIT_NORMAL | 3000/3000 | 3000/3000 | False | 8190871552 |
| 8 | Town01/scenario-2 | FIT_NORMAL | 3000/3000 | 3000/3000 | False | 9625392128 |
| 9 | Town01/scenario-3 | FIT_NORMAL | 3000/3000 | 3000/3000 | False | 10786467840 |
| 10 | Town01/scenario-4 | CAL_NORMAL | 3000/3000 | 3000/3000 | False | 12248765440 |

Every scenario has contiguous matching RGB/Seg tick coverage, complete member payloads, readable Base feather tables, finite required GNSS/IMU columns, aligned sensor rows/indexes, a strict TRAIN loader manifest and subsequent-header completion evidence. The frozen manifest records every source file size/SHA-256, table schemas/row counts, town, archive ordinal and source path. Original scenario-1/scenario-10 were verified against streamed archive hashes and never recopied or rewritten. Directory junctions provide a logical TRAIN-only view for the existing production builder.

# Frozen 8/2 FIT-CAL Manifest

Dataset manifest SHA-256: `c1e1e9fb31b63e15ebb4422214a01579be943bd9d7730bbe9a1954b4ccb0ff89`. It was frozen after all ten manifests validated, before any real model fitting.

FIT_NORMAL: Town01/scenario-1, Town01/scenario-11, Town01/scenario-12, Town01/scenario-13, Town01/scenario-14, Town01/scenario-15, Town01/scenario-2, Town01/scenario-3.

CAL_NORMAL: Town01/scenario-10, Town01/scenario-4.

The existing production rule sorted canonical IDs and applied NumPy PCG64 seed 2026; first round(0.25×10)=2 permuted IDs are CAL. No seed optimization. Production cache label TRAIN_NORMAL maps explicitly to FIT_NORMAL; CAL_NORMAL is unchanged. The accepted earlier proposal remains byte-identical.

# Multi-Scenario Cache

Exactly one production cache build: 29,990 clean rows (23,992 FIT / 5,998 CAL), plus 5,998 CAL-derived pseudo rows. Schema 2, content SHA-256 `e8dc6d712faca3ef0d0de7a073d98dffe3ef37cfce0ab0aa2c6119ed76655ad2`. NPZ size 25,292,406 bytes; preprocessing wall time 67.92 minutes.

Verified scenario/tick/partition coverage, finite features, complete hashes, CAL-only pseudo parents, target_normal=0 on pseudo rows, end_tick=parent_tick, start_tick=max(0,t−11), and exact parent features in every untouched modality block. No future/cross-scenario windows, FIT/CAL leakage or second determinism rebuild. Cache binding sidecar connects the frozen acquisition/dataset identities.

# Four-Agent Calibration Validity

| Agent | FIT scenarios/rows | CAL scenarios/normal/pseudo rows | Numerical convergence | Gradient norm | Minimum curvature | Pooled member structure | Boundary trend | finite_optimum |
|---|---|---|---|---|---|---|---|---|
| Camera | 8/23992 | 2/5998/1714 | False | 3.50994208e-09 | 3.50995009e-09 | overlapping | decreasing | False |
| Seg | 8/23992 | 2/5998/858 | True | 8.35772761e-17 | 0.0102904401 | overlapping | not_decreasing | True |
| GNSS | 8/23992 | 2/5998/1714 | False | 0.0312586884 | -1.9161577e-06 | completely_separated | decreasing | False |
| IMU | 8/23992 | 2/5998/1712 | True | 5.65156214e-16 | 0.00118423772 | overlapping | not_decreasing | True |

| Agent | Normal member range | Pseudo member range | Member overlap | Mean-score gap | Best mean threshold | Violations / observations |
|---|---|---|---|---|---|---|
| Camera | [7.47607287e-08, 0.745587295] | [8.58504547e-24, 0.903496358] | [7.47607287e-08, 0.745587295] | -0.902873653 | 8.55347056e-08 | 859/7712 |
| Seg | [5.34183768e-07, 0.936512087] | [0, 0.852543731] | [5.34183768e-07, 0.852543731] | -0.851764783 | 0.0571171324 | 298/6856 |
| GNSS | [2.5597391e-06, 0.9517129] | [0, 0] | none | 0.0053138325 | 0.00265691625 | 0/7712 |
| IMU | [7.96917012e-08, 0.98193594] | [0, 0.777178086] | [7.96917012e-08, 0.777178086] | -0.770948612 | 7.29860778e-07 | 294/7710 |

`calibration_results.json` records complete mean/member distributions and overlap, all threshold diagnostics, fitted mapping parameters, Jensen diagnostics, fixed/relative slope curves and optimization status. Thresholds are diagnostics only. Production ensemble-predictive BCE and validity rule are unchanged. Accepted finite candidates are numerical stationary evidence, not global-existence proofs. Invalid scientific probability/UQ is null; raw scores and objective diagnostics remain available.

# 2-Scenario vs 10-Scenario Diagnosis

| Agent | Two-scenario geometry / violations | Ten-scenario geometry / violations | Two-scenario valid | Ten-scenario valid |
|---|---|---|---|---|
| Camera | overlapping; 22/3856 (0.5705%) | overlapping; 859/7712 (11.1385%) | False | False |
| Seg | overlapping; 159/3428 (4.6383%) | overlapping; 298/6856 (4.3466%) | True | True |
| GNSS | completely_separated; 0/3856 (0.0000%) | completely_separated; 0/7712 (0.0000%) | False | False |
| IMU | overlapping; 139/3855 (3.6057%) | overlapping; 294/7710 (3.8132%) | True | True |

| Agent / CAL class | Two-scenario mean score p10/median/p90 | Ten-scenario mean score p10/median/p90 |
|---|---|---|
| Camera / normal | 0.00496945/0.0466207341/0.272370765 | 0.000517029702/0.232783606/0.56758653 |
| Camera / pseudo | 7.00629283e-19/3.28313533e-11/4.63267807e-09 | 8.74782251e-16/8.77189946e-07/0.67954156 |
| Seg / normal | 0.255433837/0.6607745/0.784600117 | 0.475919647/0.691168216/0.851600954 |
| Seg / pseudo | 0/1.60598506e-07/0.182053135 | 0/1.33542405e-06/0.350845239 |
| GNSS / normal | 0.29260546/0.85856578/0.93811345 | 0.434494278/0.806792862/0.934905366 |
| GNSS / pseudo | 0/0/0 | 0/0/0 |
| IMU / normal | 0.171060401/0.97867285/0.978857267 | 0.282250579/0.968162337/0.96859115 |
| IMU / pseudo | 0/0/0.0712066597 | 0/0/0.146881967 |

Ten-scenario accepted finite candidates: Seg, IMU. Invalid: Camera, GNSS. Camera/GNSS: Camera: old valid=False, new valid=False, new member structure=overlapping; GNSS: old valid=False, new valid=False, new member structure=completely_separated.

Additional FIT/CAL diversity is the only scientific change. This descriptive checkpoint comparison has different training/CAL scenarios and row counts; it does not establish generalization or causal performance improvement. No model/recipe/calibration changes follow from the comparison.
Camera's earlier suspected-separation flag is absent at N=10, but that does not validate its mapping: overlap expanded substantially, mean-score threshold violations increased from 22/3856 (0.5705%) to 859/7712 (11.1385%), and the optimizer exhausted 300 iterations without convergence. Its fitted positive slope is about 8.59e-7, with very small gradient/curvature; the unchanged validity rule rejects this numerical result. No claim about mathematical nonexistence of a Camera finite optimum follows. GNSS remains strictly separated at member level, with zero threshold violations and a positive gap. Therefore the unregularized mathematical MLE remains non-finite. Additional scenario diversity did not produce valid Camera/GNSS mappings. Seg and IMU retain accepted finite stationary candidates with positive local curvature. All ten scenarios are Town01; this is scenario diversity within one town, not cross-town validation.

# UQ Status

Unchanged: total=H(mean p_t), aleatoric=mean H(p_t), epistemic=max(0,total−aleatoric). The following distributions use validated mappings only, on CAL clean versus modality-matched pseudo rows.

| Agent / CAL class | prob_normal p10/median/p90 | Total median | Aleatoric median | Epistemic p10/median/p90/max |
|---|---|---|---|---|
| Seg / normal | 0.945466853/0.992896108/0.998491618 | 0.0422223528 | 0.0422185277 | 9.29521057e-08/3.12476216e-06/7.44149308e-05/0.00318025146 |
| Seg / pseudo | 0.147065583/0.147067207/0.83778847 | 0.417585094 | 0.417585094 | 0/1.02992614e-12/6.65905303e-05/0.00112734246 |
| IMU / normal | 0.912529493/0.999998182/0.999998195 | 2.58447139e-05 | 2.58446767e-05 | 3.48594428e-11/3.71954513e-11/9.532678e-05/0.00492399859 |
| IMU / pseudo | 0.106575722/0.106575722/0.550219695 | 0.339295619 | 0.339295619 | 0/0/0.000282082261/0.00276609709 |

The previous near-zero probability-space epistemic finding persists for typical observations, while nonzero tails are now explicitly visible. Seg medians are 3.12476e-6 (normal) and 1.02993e-12 (pseudo); IMU medians are 3.71955e-11 (normal) and exactly 0 (pseudo). Normal/pseudo p90 epistemic values are 7.44149e-5/6.65905e-5 for Seg and 9.53268e-5/2.82082e-4 for IMU. Maxima are 0.00318025/0.00112734 for Seg and 0.00492400/0.00276610 for IMU. Entropies use the unchanged natural-log production convention. Thus this is not identically zero disagreement, but typical probability-space epistemic remains small despite additional scenarios. No alternative epistemic definition or selected cutoff is introduced, and invalid Camera/GNSS mappings contribute no validated UQ.

# Kaggle Handoff

Prepared local compact checkpoint artifact: `C:\Users\Aditya\AppData\Local\Temp\carla_train_expand10\kaggle_compact_train10_v1`, 16,241,885 bytes, 35,988 rows. No raw archive, images, graph samples or graph training. It contains compact features, five member scores per agent, scenario/town/tick/partition dictionaries, per-agent validity/status, nullable probabilities/UQ, clean-vs-pseudo and per-agent targets, pseudo parent/recipe/severity/seed/window provenance, fitted one-class parameters, calibration parameters/audits and version/hash manifests.

Invalid probability/total/aleatoric/epistemic use NaN with an explicit validity mask (scientific null). Target semantics are constructed normality, never physical safety. Only CAL pseudo examples exist at this milestone; they are reserved for calibration and must not be repurposed as graph training data. FIT-derived graph pseudo generation and paired-seed StandardGAT/EpistemicGAT training require a later stage.

# Storage / Timing

Acquisition including copy/replay/validation: 62.79 minutes. Preprocessing: 67.92 minutes. Pipeline total: 68.23 minutes. Expanded prefix: 12.548 GB; new raw extraction: 10.434 GB; cache total: 25.297 MB; compact handoff: 16.242 MB. Existing original prefix/scenarios/caches remain separately preserved.

# Tests / Core Integrity

Focused: 336 passed in 79.33s (0:01:19).
Broad requested regression: 505 passed, 14 warnings in 98.99s (0:01:38).

23 new acquisition tests cover partial compressed/TAR state, boundary-based completeness, safe paths/links/checksums, verified reuse without overwrite, PAX/GNU names, Range rejection and valid gzip termination. Broad total 505 = accepted baseline 482 + 23 new acquisition tests; zero failures. Existing source/test hashes match the frozen initial inventory; generic/frozen code and accepted calibration/recipe implementations are unchanged. Bare import cognix loads no adapters. Original prefix and all prior probe/cache artifacts are preserved. No TEST access, GAT/conformal, severity/threshold/calibration tuning, commits or pushes.

New source/test files: `cognix/adapters/carla/train_acquisition.py`, `tests/unit/test_carla_train_acquisition.py`. New milestone scripts/manifests/logs/results/report are explicitly listed in `integrity_results.json`. No pre-existing source/test file was modified.

# Ready / Not Ready for 20-Scenario Expansion

Engineering path is ready for a separately authorized N=20 protocol if storage/transport checks remain satisfied. Preserve this intermediate checkpoint and freeze a new 15/5 manifest without optimizing seeds or reacting to performance. No expansion to 20 has begun.

# Ready / Not Ready for GAT

Not ready to start GAT at this intermediate checkpoint. Preferred development dataset is N=20 first, subject to separate authorization; any remaining invalid calibrations must stay unavailable. Graph target/sampling/evaluation decisions require their own preregistration.

# Recommended Next Step

Review this ten-scenario diagnostic checkpoint, then separately authorize the twenty-scenario TRAIN expansion if desired. Keep calibration and recipes unchanged; preserve all checkpoint artifacts. Stop here: no further download, graph training, TEST access or tuning.

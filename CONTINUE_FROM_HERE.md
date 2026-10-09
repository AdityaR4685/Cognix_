# COGNIX — Continue From Here

## Scientific status
COGNIX graph experiment development is complete and the final held-out TEST phase has not been executed yet.

Experiment 1 is closed/immutable. For the current graph experiment, all 15 development-selected checkpoints are frozen before TEST access: methods NoGraph, StandardGAT, and EpistemicGAT for seeds 101, 202, 303, 404, and 505. No best-seed selection, seed replacement, retraining, threshold retuning, recalibration, or post-hoc checkpoint selection is allowed.

Probability semantics remain: q_normal is P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task. Do not describe it as probability of safe driving, physical safety, or guaranteed official anomaly posterior.

## Authoritative frozen seals
- Graph checkpoint freeze: 9f787f8c3f8cb093b1daaea1a3bf414265456f5f0b139ac7f1d5e176c9d9a62c
- Pre-TEST evaluation protocol: d4ceb25023f49acc96ab2a62f05f7fdc92c88c608705ce5fbc34dfe64367691e
- Pre-download TEST source declaration: ddaa40d5d0dd04afefda9154b0203eb89155f67498c9b400f986b85a24db2dba
- Graph development data seal: 633b5801b86523c0a9396e45c02ddd23ae927a41ee032528eb0285bdbf296066
- Graph scientific-content SHA256: c8a82c361514fdad9807e23fbfd691c354db081cecdb22506a068a950a688a13

Never modify sealed bundle contents. Verify their SHA256SUMS/SHA256SUMS.sha256 before use.

## Repository
Personal fork: https://github.com/AdityaR4685/Cognix_.git
Branch: main
Pre-handoff base HEAD: 1e636800438a681ecfba7630ed372581abc4c8c7

The Git commit containing this file should include the compact scientific/provenance bundles, not the large raw/generated datasets.

## External data and large artifacts
TRAIN archive already exists on Aditya's machine:
E:\carlanomaly-base-train.tar.gz
Bytes: 146453559283
SHA256: 6cf22ecf7d2910b45ed71127525834d1a4a26a65cc65918e7b5e27da95a38683

The final TEST archive is NOT stored in Git and must be downloaded separately by each teammate who needs to execute TEST:
https://data.carlanomaly.de/v1/carlanomaly-base-test.tar.gz

Expected filename:
carlanomaly-base-test.tar.gz

Latest verified HTTP HEAD metadata before download:
Content-Length: 91538225599 bytes
Accept-Ranges: bytes
ETag: "6a2e752f-15501a79bf"
Last-Modified: Sun, 14 Jun 2026 09:32:31 GMT

Preferred local path:
E:\carlanomaly-base-test.tar.gz

Do not extract TEST immediately after downloading. First record the exact local byte count and SHA256, bind them to the frozen TEST-source declaration, and preserve download evidence. If the byte count/source/structure is incompatible with the frozen protocol, fail closed.

## Important omitted artifacts
Large or reproducible artifacts are intentionally not committed, including graph_development_data_v3, kaggle_graph_training_results_v1/results.zip, large Kaggle upload payloads, old recovery bundles, and the raw TEST archive.

A fresh clone is scientifically resumable but may not yet be fully self-contained for TEST feature generation. Before executing TEST, verify that every frozen TRAIN-derived upstream artifact needed to construct TEST features is locally available and hash-bound. If any required upstream model/calibration/representation artifact is missing, STOP and reconstruct or package it from TRAIN only before TEST execution. Do not fit anything on TEST.

## Frozen TEST evaluation rules
Evaluate all 15 frozen runs. Use each run's frozen GRAPH_VAL corruption threshold. No TEST-time training, optimizer steps, checkpoint selection, threshold tuning, calibration fitting, feature-scaler fitting, representation fitting, hyperparameter changes, result-dependent retry, seed replacement, or dropping negative/null results.

Scores:
q_normal = frozen model mean sigmoid output
p_corruption = 1 - q_normal
predicted_corruption iff p_corruption >= frozen per-run GRAPH_VAL threshold

Metrics are computed per official TEST scenario and then macro-averaged across scenarios:
BCE on q_normal versus target_normal
AUROC on p_corruption versus target_corruption
F1 for corruption using the frozen threshold

Primary paired comparisons by seed:
StandardGAT - NoGraph
EpistemicGAT - StandardGAT

For each comparison report all five paired seed differences plus mean, sample SD, minimum, and maximum. No inferential significance claim is predeclared; n=5 results are descriptive.

## Exact next phase
1. Finish downloading the TEST archive to a fresh local path.
2. Without extracting it, record byte count and SHA256.
3. Preserve and seal download/source-binding evidence.
4. Verify all frozen checkpoint/protocol/source seals again.
5. Verify required TRAIN-derived upstream runtime artifacts are present and immutable.
6. Inspect TEST archive structure read-only and fail closed on incompatibility.
7. Prepare TEST feature-generation/evaluation execution bundle with zero fitting/tuning on TEST.
8. Execute all 15 frozen runs, preserve raw predictions/per-scenario metrics/failures, then aggregate only according to the frozen protocol.

Do not skip directly to model execution just because the TEST archive is available.

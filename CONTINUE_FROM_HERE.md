# COGNIX — Continue From Here

## Scientific status
Experiment 2B is FINISHED, CLOSED, and IMMUTABLE. Final held-out TEST evaluation is complete. No TEST rerun, Phase A/B rerun, retraining, threshold tuning, recalibration, checkpoint reselection, prediction/label alteration, or post-hoc AUROC rescue is authorized. All five paired seeds and negative/undefined outcomes are retained.

Experiment 1 is closed/immutable. For the current graph experiment, all 15 development-selected checkpoints are frozen before TEST access: methods NoGraph, StandardGAT, and EpistemicGAT for seeds 101, 202, 303, 404, and 505. No best-seed selection, seed replacement, retraining, threshold retuning, recalibration, or post-hoc checkpoint selection is allowed.

Probability semantics remain: q_normal is P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task. Do not describe it as probability of safe driving, physical safety, or guaranteed official anomaly posterior.

The existing comprehensive report is [docs/research/COGNIX_FINAL_STATUS.md](docs/research/COGNIX_FINAL_STATUS.md#6-experiment-2b-closed-held-out-graph-evaluation). Section 6 contains the setup, metrics, TEST tables, paired-difference figures, discussion, limitations, and final conclusion. No LaTeX paper source is present in this checkout.

## Final authenticated artifacts and findings

- Final Phase-B TEST result seal: `ec5ea056cb53050c284816c470845f233d4fe8c566917195658639d20bdee15e`
- Final Experiment 2B conclusion bundle seal: `cff86433be91a3fdabd3b43568b7fb2c24c0760fa78fddeb298b9ae9cbb0cad8`
- Final conclusion JSON SHA-256: `6abef23828dd6a05fa05ab7e56e8be74072d5674417aafe9e321ee959b0e2f33`

Read [final_conclusion.json](reports/carla_train_full_v1/graph_test_final_conclusion_bundle_v1/final_conclusion.json) and [sealed Phase-B metrics](reports/carla_train_full_v1/graph_test_phase_B_results_v1/metrics.json) as authoritative scientific sources. The conclusion JSON's historical `FINAL_TEST_CONCLUSION_STAGED_FOR_FREEZE` string is preserved; the authenticated seal binds the final bytes.

All 15 frozen runs used 627 official TEST scenarios and 187,473 scored timesteps per run: ticks 1..299, with tick 0 reserved for synchronization validation. StandardGAT reduced BCE across all five seeds but also reduced frozen-threshold corruption F1 across all five seeds. The graph benefit was metric-dependent. EpistemicGAT showed no descriptive evidence of improvement over StandardGAT: BCE was mixed and slightly worse on average, and F1 was essentially neutral to slightly worse. No inferential significance test was prespecified or performed.

All 107 normal TEST scenarios had 299 normal ticks and zero corruption ticks; all 520 anomaly scenarios contained both classes. Under the frozen fail-closed protocol, no undefined scenario was excluded. The prespecified 627-scenario macro-AUROC was undefined and paired AUROC comparisons were unavailable.

GRAPH_VAL was development-only because upstream representation fitting used all 76 FIT scenarios before the 15 GRAPH_VAL scenarios were separated. Improved development F1 did not generalize to TEST. Threshold-transfer effects or score-distribution shift remain possible interpretations, not confirmed causes; thresholds were transferred unchanged and no post-hoc investigation occurred.

Treat all experiment-result directories as read-only, including Phase A predictions, Phase B results, the final conclusion bundle, and scoring bundle v2. Preserve `graph_test_phase_B_scoring_bundle_v1_superseded_prelabel` and its failure/supersession records. Historical preparation READMEs inside sealed bundles describe their preparation state, not the current experiment status or authorization.

## Earlier frozen seals retained for provenance
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

Leave reporting edits in the working tree for review. Do not stage, commit, push, pull, merge, or rebase without an explicit later instruction.

## External data and large artifacts
TRAIN archive already exists on Aditya's machine:
E:\carlanomaly-base-train.tar.gz
Bytes: 146453559283
SHA256: 6cf22ecf7d2910b45ed71127525834d1a4a26a65cc65918e7b5e27da95a38683

The final TEST archive is not stored in Git. Its historical source is retained below for provenance; this closed reporting task does not require acquisition or TEST execution:
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

The download/source-binding and completed execution evidence are preserved in the experiment directories. Reporting uses authenticated compact outputs without reacquisition or extraction.

## Important omitted artifacts
Large or reproducible artifacts are intentionally not committed, including graph_development_data_v3, kaggle_graph_training_results_v1/results.zip, large Kaggle upload payloads, old recovery bundles, and the raw TEST archive.

A fresh clone may not contain all large local upstream artifacts. Preserve existing archives, checkpoints, fitted inputs, and generated data. Missing artifacts do not authorize reconstruction, refitting, or rerunning this closed experiment; use the locally authenticated final outputs for reporting.

## Frozen TEST evaluation rules
All 15 frozen runs were evaluated using each run's frozen GRAPH_VAL corruption threshold. No TEST-time training, optimizer steps, checkpoint selection, threshold tuning, calibration fitting, feature-scaler fitting, representation fitting, hyperparameter changes, result-dependent retry, seed replacement, or dropping negative/null results occurred. The following records the completed protocol, not instructions to rerun it.

Scores:
q_normal = frozen model mean sigmoid output
p_corruption = 1 - q_normal
predicted_corruption iff p_corruption >= frozen per-run GRAPH_VAL threshold

Metrics were computed per official TEST scenario and then macro-averaged across all 627 scenarios:
BCE on q_normal versus target_normal
AUROC on p_corruption versus target_corruption
F1 for corruption using the frozen threshold

Primary paired comparisons by seed:
StandardGAT - NoGraph
EpistemicGAT - StandardGAT

For each comparison report all five paired seed differences plus mean, sample SD, minimum, and maximum. No inferential significance claim is predeclared; n=5 results are descriptive.

## Current next steps

Review the integrated report and [packaging audit](docs/research/EXPERIMENT_2B_PACKAGING_AUDIT.md). Report-only figures can be regenerated with `python docs/research/scripts/plot_experiment_2b.py`, which reads authenticated stored numbers without opening raw TEST labels. Experiment 2B has no remaining scientific execution phase. Any new scientific investigation requires a separate experiment and cannot alter this closed result.

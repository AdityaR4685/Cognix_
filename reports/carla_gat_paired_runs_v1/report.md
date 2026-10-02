# Frozen Execution Verification

All 15 preregistered real graph runs completed using the unchanged gated launcher independent commands, one visible T4, the exact sealed environment lock 0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d, graph protocol 68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203, graph scientific artifact baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237, trainer bundle 9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b. Rows: 71,976 GRAPH_TRAIN and 17,994 GRAPH_VALIDATION. Three nodes Camera/IMU/Seg, frozen p/E/A, six directed edges and no self-loops. No CAL or TEST arrays were loaded.

# 15-Run Completion Ledger

15/15 complete; run_ledger.json was sealed before any cross-method aggregation, SHA-256 cfc49769c3fb25c294afc11669ed3e50afa5c0655d156b0a2a98db61c2cc1568. Every independent run retains all strict-minimum checkpoints, selected checkpoint, full history, predictions, metrics, observer pairing evidence, final executed state, logs, runtime and environment. No runs replaced or selected by seed. Per-run raw SHA256SUMS verified before analysis.

# Per-Seed Results

Scenario-macro validation AUROC:

| Seed | NoGraph | StandardGAT | EpistemicGAT | Epi − Standard |
|---|---:|---:|---:|---:|
| 101 | 0.749402389 | 0.801201175 | 0.801199692 | -0.000001482 |
| 202 | 0.747197567 | 0.800852776 | 0.800995704 | +0.000142929 |
| 303 | 0.745990003 | 0.800493926 | 0.800485309 | -0.000008617 |
| 404 | 0.748315220 | 0.802422748 | 0.802419894 | -0.000002854 |
| 505 | 0.747872851 | 0.803676325 | 0.803678604 | +0.000002279 |

All eight frozen metrics, pooled and scenario-macro, selected epoch, minimum BCE and validation-selected threshold are recorded for every run in per_seed_metrics.json.

# Primary EpistemicGAT vs StandardGAT Contrast

Mean Δ=+2.64509643706e-05; median=-1.48246962983e-06; sample SD=6.52305623103e-05; min=-8.61685472353e-06; max=+0.000142928603188; paired dz=0.4054995608469083; preregistered t4 conditional seed interval=[np.float64(-5.4543476426879355e-05), np.float64(0.00010744540516811291)]. This interval describes initialization variability conditional on the fixed validation set.

# NoGraph Comparison

NoGraph values and both GAT-minus-NoGraph contrasts are retained per seed in paired_comparison.json. These comparisons remain secondary/exploratory. NoGraph has 48 parameters; both GATs have 50 and receive identical p/E/A.

# Scenario-Level Results

- Town01/scenario-12: paired AUROC differences across seeds (101, 202, 303, 404, 505): -0.000003780, +0.000943796, +0.000001779, +0.000000222, +0.000002946
- Town01/scenario-14: paired AUROC differences across seeds (101, 202, 303, 404, 505): -0.000000167, -0.000222037, -0.000027796, -0.000002557, +0.000003725
- Town02/scenario-2: paired AUROC differences across seeds (101, 202, 303, 404, 505): -0.000000500, -0.000292973, +0.000000167, -0.000006226, +0.000000167

All per-scenario AUROC, AUPRC, F1, accuracy, balanced accuracy, Brier, BCE, ECE, confusion counts and bin evidence are retained in scenario_metrics.json.

# Recipe-Level Results

- camera_brightness_shift: all method/seed pooled, scenario and macro metrics retained in recipe_metrics.json.
- camera_occlusion: all method/seed pooled, scenario and macro metrics retained in recipe_metrics.json.
- imu_bias_scale: all method/seed pooled, scenario and macro metrics retained in recipe_metrics.json.
- imu_spike: all method/seed pooled, scenario and macro metrics retained in recipe_metrics.json.
- seg_region_corruption: all method/seed pooled, scenario and macro metrics retained in recipe_metrics.json.

Diagnostics retain each recipe's pseudo rows together with the same pairs' clean parents. Undefined values retain null and frozen reasons.

# Calibration Metrics

Per-run pooled and scenario-macro Brier, BCE and 15-bin ECE are in calibration_metrics.json; per-scenario and per-recipe values are retained. No calibration was refitted. Thresholds maximize validation scenario-macro F1 over unique corruption probabilities union {0,1}, with largest-threshold exact tie rule; selection and reporting reuse validation and therefore remain development statistics.

# Exact Sign-Flip Result

Exact two-sided p=0.875; all 32 assignments were enumerated using the unchanged frozen metric function. With five pairs the minimum attainable exact two-sided p is 0.0625. No p < 0.05 significance claim is made.

# Whole-Scenario Bootstrap

10,000 whole-scenario draws, PCG64 seed 606, percentile interval=[-5.9873242175512155e-05, 0.00018899264076133182]. All ticks, clean/pseudo pairs and methods remain aligned within each scenario. Only three graph-validation scenarios are available, so this is descriptive fragility evidence and is not an independent generalization guarantee. No tick bootstrap was performed.

# Attention / Epistemic Diagnostics

All ten selected GAT checkpoints passed frozen attention normalization/finite/self-edge checks, and audited probabilities exactly matched saved selected-checkpoint validation predictions. Descriptive paired attention differences and raw E/sender-weight/log-adjustment summaries are retained in attention_summary.json. The prior remains 1/(1+E); neutral rounding or weak effects are accepted without amplification. Attention never selected a checkpoint.

# Seed Stability

Paired difference signs, in preregistered seed order: ['negative', 'positive', 'negative', 'negative', 'positive']. All five values and variation summaries are retained without ranking/selecting seeds. Each actual trainer construction was checked for byte-identical explicitly cloned GAT tensors with independent storage, actual first-forward RNG and optimizer, and identical plans for all 100 frozen epoch permutations/batch boundaries before optimization; each executed epoch was checked against that plan.

# Negative / Positive Findings

Observed Epi-minus-Standard mean Δ=+2.64509643706e-05, range [-8.61685472353e-06, +0.000142928603188]. Positive, negative and zero differences are all retained. This fixed experiment supports only descriptive statements about these frozen methods, constructed corruptions and validation scenarios; it establishes no statistical superiority at p < 0.05.

# Scientific Interpretation

These are fixed-representation graph-development results on frozen GRAPH_VALIDATION scenarios, under artificial paired 1:1 clean/pseudo prevalence. Validation is reused for early stopping and threshold selection; upstream representation saw FIT scenarios that include graph validation. Results are not official CarlAnomaly TEST/anomaly performance, physical safety performance or independent population generalization. No feature, recipe, architecture or prior was tuned after observing outcomes.

# Environment / Determinism

Exact Python/PyTorch/NumPy/CUDA/cuDNN and T4 specifications matched the sealed lock. CUDA_VISIBLE_DEVICES=0 was verified before PyTorch import in fresh trainer children; Kaggle allocated two T4s, one was visible, and no distributed execution occurred. Deterministic algorithms and cuDNN deterministic enabled, benchmark/TF32/AMP disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8. Adam lr .001, weight decay .0001, batch 256, max 100 epochs, dropout .1, no scheduler/gradient clipping, patience 10/min_delta 1e-5; selected checkpoint is strict observed minimum validation macro BCE with earliest exact tie.

# Core / Artifact Integrity

Portable bundle/source/graph/protocol seals and exact environment lock verified after all runs. The first optional audit attempt stopped because validation-only evaluation rebatched saved full-dataset predictions. The failed attempt is retained. Attention audit was recovered by using the original full-dataset batch boundaries and then selecting validation rows; all predictions were required to match bit-for-bit. No new tolerance, training, scientific source change or output replacement occurred. Local input/code/upstream preservation is verified when downloaded evidence is imported. Raw run artifacts and final result artifacts are separately sealed. N=20, calibration, feature extraction, recipes, generic COGNIX and frozen synthetic results were not edited. No commit or push occurred.

# Ready / Not Ready for Next Pipeline Stage

Ready for review of the complete frozen primary experiment. TEST, conformal and secondary GNSS remain unauthorized and unexecuted.

# Recommended Next Step

Review the sealed development-validation findings and separately preregister/authorize the next pipeline stage, resolving calibration independence and scenario/temporal exchangeability before any conformal or held-out evaluation. Stop this milestone without additional scientific runs.

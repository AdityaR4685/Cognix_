# Frozen Graph Trainers v1

## Frozen Trainer Specification Verification

Protocol `68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203` and graph scientific artifact `baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237` match the frozen identities. All sealed files, graph file/schema/manifest hashes and N20 dependencies were verified before editing and after validation. No model/data/split/metric decisions were changed. Trainer bundle SHA-256: `9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b`.

## Dataset Loader

Read-only float64 [89970,3,3] scientific storage; Camera/IMU/Seg and [prob_normal,epistemic,aleatoric]. New float32 model inputs and separate raw float64 E are materialized per batch. Exact frozen FIT split, pair/row provenance, six directed edges and zero diagonal verified. No tick resplitting, upstream refitting or raw/TEST loader. Training indices contain only GRAPH_TRAIN; all three frozen validation scenarios are required.

## NoGraph Implementation

Shared bias-free Linear(3,12) -> ELU -> hidden Dropout(0.1) -> bias-free Linear(12,1), with Xavier initialization and mean node sigmoid pooling. Exactly 48 trainable parameters, no message passing. Generic identity NoGraph remains unchanged.

## StandardGAT Implementation

Two unchanged generic EpistemicGATLayerPT instances, 3 -> 8 -> 1, single head, 50 parameters. Adapter-side batch algebra uses the original W/a, LeakyReLU(0.2), ELU hidden output, linear final output and post-softmax dropout 0.1. Standard prior is all ones. Batched/single-graph execution agrees with the generic forward path within atol=1e-7, rtol=1e-6; generic code remains unchanged.

## EpistemicGAT Implementation

The identical 50-parameter architecture uses the original sender's canonical E at both layers. Prior is float64 1/(1+E), stored float32 before the unchanged log operation, matching generic compute_epistemic_weights including neutral rounding for tiny E. No scaling, normalization, learned strength or amplification. Nonfinite/negative E rejected. E=0 gives exact matched-weight/RNG equivalence in inference and fixture training.

## Paired Initialization / Batch Integrity

StandardGAT state is explicitly deep-cloned into EpistemicGAT, with byte equality and independent storage tested for each of 101,202,303,404,505. NoGraph uses the corresponding seed. All methods reset dropout RNG before optimization. PCG64(seed+zero-based epoch) permutations and 256-row boundaries are independent of Torch RNG and identical for paired methods. Adam lr=0.001, weight_decay=0.0001, max 100 epochs, no scheduler, AMP, TF32 or clipping.

## Early Stopping

Equal-scenario validation BCE is the arithmetic mean of three scenario means, with no pooled substitute. Tests demonstrate opposing pooled/macro trends. Patience 10 resets only when loss < tracked_best-1e-5. Independently save every strict observed minimum, including improvements smaller than min_delta; exact ties retain the earliest epoch. Unit sequences exercise stopping, tiny improvements and ties. Learnable smoke runs used their permitted 100-epoch maximum.

## Metrics / Thresholding

Implemented pooled, per-scenario, equal-scenario macro and per-recipe paired clean/pseudo diagnostics for corruption AUROC, stepwise AP/AUPRC, F1, accuracy, balanced accuracy, normal-probability Brier/BCE and 15-bin ECE. Threshold uses only graph validation, candidates unique p_corrupt plus {0,1}, macro F1 objective, largest threshold on exact ties, and >= comparison. Undefined class-dependent metrics are null with reasons; the frozen zero-denominator F1 rule explicitly returns 0. ECE uses [k/15,(k+1)/15), with final bin including 1 and zero contribution from empty bins. Frozen five-pair summaries, exact sign flips and whole-scenario bootstrap helpers are implemented but no scientific experiment aggregate was calculated.

## Checkpoint / Prediction Schema

Exclusive per-method/seed directories retain checkpoint_epoch_NNN.pt for observed minima. Metadata binds method, seed, epoch, selected epoch, best macro BCE, model/training configuration, graph/protocol/split and trainer source identity, plus explicit synthetic/real training-data identity. Saved model/Adam state, early-stop trackers, history and Python/NumPy/Torch/available-CUDA RNG states support exact continuation; epoch-50 continuation reproduced the full fixture trajectory. A deterministic scientific content hash is separate from Torch ZIP bytes. Row-keyed predictions include row_key, pair_id, scenario, tick, target_normal, recipe (empty string for clean), split, method, seed, selected epoch, checkpoint content hash, q_normal and p_corrupt. No checkpoint overwrites are permitted.

## Attention Diagnostics

Optional validation_attention.npz records row-keyed, audit-only pre-dropout incoming attention [rows,2 layers,3 receivers,3 senders]; single head is implicit. Finite values, incoming sum 1 and zero diagonal verified. Fixed-logit fixtures show increased sender E reduces its contribution at both layers. Attention is never optimized as a diagnostic objective or selected by visual favorability.

## Smoke Training Results

Only synthetic engineering fixtures were trained: seed 101, 128 rows (32 train, 96 validation across three scenarios). These results are implementation evidence and do not measure COGNIX performance. Earlier candidate smoke outputs remain preserved separately.

| Method | Epochs | Selected epoch | First / last train BCE | Finite gradients | Updated |
|---|---:|---:|---|---|---|
| nograph | 100 | 100 | 0.64997280 / 0.53483009 | yes | yes |
| standard_gat | 100 | 100 | 0.64115536 / 0.56476361 | yes | yes |
| epistemic_gat | 100 | 100 | 0.64115536 / 0.56476361 | yes | yes |

## Determinism

All three methods were repeated in focused tests: exact epoch order/loss trajectory, selected epoch, checkpoint scientific contents, predictions and metrics. An independent persistent StandardGAT CLI repeat is also exact. CPU uses one thread and deterministic algorithms with errors enabled. CUDA was not exercised locally; no CUDA operation's determinism is claimed. Future CUDA smoke validation must pass before full execution; nondeterministic-operation errors stop without fallback.

## Kaggle Environment Lock

Proposed lock records actual local Python 3.12.10, PyTorch 2.14.0+cpu, NumPy 1.26.4, installed scikit-learn 1.9.0 (not used), CUDA availability=False, CUDA runtime=None, cuDNN=None. No Kaggle package versions are invented. Deterministic algorithms, cuDNN deterministic=true/benchmark=false, both TF32 settings false, AMP=false and CUBLAS_WORKSPACE_CONFIG=:4096:8 are recorded. validate_environment.py prints/checks actual future Kaggle versions, verifies hashes, and optionally runs deterministic GPU fixture repeats on one T4 to create a validated execution lock. Full commands are supplied in future_commands.md but were not invoked.

## Focused Tests

60 passed, zero failures, covering all 28 requested categories plus source-hash refusal, oversized smoke protection, generic forward equivalence, checkpoint continuation and frozen statistics helpers. Evidence: focused_tests_validated.txt and test_coverage.json.

## Broad Regression

Ran PYTHONHASHSEED=0 py -3.12 -m pytest tests/unit tests/regression tests/integration tests/test_interfaces.py tests/test_mathematics.py -q. 608 passed (548 baseline + 60 trainer cases), zero failures, 14 baseline warnings. Evidence: broad_tests.txt.

## Core / Artifact Integrity

All 325 pre-existing repository files and 132 intentional untracked files remain unchanged. The float64 graph artifact, preregistration, N20 cache/handoff, exporter, calibration, features, recipes, generic core/graphs and frozen synthetic benchmark retain their hashes. No official TEST, CAL optimization, full graph training, Kaggle execution/upload, conformal, scientific tuning, commits or pushes. GNSS secondary condition remains deferred. File inventory contains 25 new repository files and 738 persistent synthetic smoke/checkpoint files; initial work inventory is retained in initial_audit.json.

New repository files:

- `cognix/adapters/carla/graph_training.py`
- `cognix/adapters/carla/graph_training_data.py`
- `cognix/adapters/carla/graph_training_metrics.py`
- `cognix/adapters/carla/graph_training_models.py`
- `reports/carla_graph_trainers_v1/broad_tests.txt`
- `reports/carla_graph_trainers_v1/environment_console.txt`
- `reports/carla_graph_trainers_v1/environment_console_validated.txt`
- `reports/carla_graph_trainers_v1/file_inventory.json`
- `reports/carla_graph_trainers_v1/focused_tests.txt`
- `reports/carla_graph_trainers_v1/focused_tests_validated.txt`
- `reports/carla_graph_trainers_v1/future_commands.md`
- `reports/carla_graph_trainers_v1/initial_audit.json`
- `reports/carla_graph_trainers_v1/integrity_console.txt`
- `reports/carla_graph_trainers_v1/integrity_results.json`
- `reports/carla_graph_trainers_v1/proposed_environment_lock.json`
- `reports/carla_graph_trainers_v1/proposed_environment_lock_validated.json`
- `reports/carla_graph_trainers_v1/repeat_console.txt`
- `reports/carla_graph_trainers_v1/report.md`
- `reports/carla_graph_trainers_v1/smoke_console.txt`
- `reports/carla_graph_trainers_v1/smoke_console_validated.txt`
- `reports/carla_graph_trainers_v1/test_coverage.json`
- `reports/carla_graph_trainers_v1/train.py`
- `reports/carla_graph_trainers_v1/validate_and_report.py`
- `reports/carla_graph_trainers_v1/validate_environment.py`
- `tests/unit/test_carla_graph_trainers.py`

## Ready / Not Ready for Full Paired-Seed Run

Trainer implementation is validated on CPU fixtures. Full paired execution is NOT READY to launch in this milestone: it requires a future validated single-T4 runtime/CUDA fixture lock and separately authorized execution. The full-run guard is active; no five-seed experiment was launched.

## Ready / Not Ready for Kaggle

NOT READY for execution/upload. Actual Kaggle versions and CUDA fixture reproducibility remain unverified. The proposed lock and future validation/execution commands are prepared.

## Recommended Next Step

Authorize only future Kaggle environment and CUDA fixture validation first. Once that lock passes, separately authorize the frozen full paired-seed run. Preserve every planned seed and failure; keep TEST, conformal and secondary GNSS outside that execution.


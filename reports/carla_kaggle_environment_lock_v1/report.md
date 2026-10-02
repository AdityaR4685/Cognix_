## Kaggle Runtime

Actual notebook: https://www.kaggle.com/code/adityar4685/notebook694cee9113.
Python 3.12.13; PyTorch 2.10.0+cu128; NumPy 2.0.2; scikit-learn 1.6.1 (unused); CUDA runtime 12.8; cuDNN 91002; driver 580.178.04.
Kaggle allocated two Tesla T4 GPUs. The user explicitly authorized CUDA_VISIBLE_DEVICES=0 before importing PyTorch. Only GPU 0 was visible/used; no distributed execution. Selected UUID GPU-34af2d32-ab64-85ba-f040-9e12069b1053; compute capability 7.5; 40 multiprocessors; PyTorch-reported memory 15,636,037,632 bytes. Full platform/image identity is in environment_lock.json.

## Frozen Hash Verification

Graph protocol: 68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203.
Graph scientific artifact: baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237.
Trainer bundle: 9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b.
All matched before and after CUDA fixtures. NPZ/schema/manifest, preregistration seals, generic sources, node order Camera/IMU/Seg, feature order prob_normal/epistemic/aleatoric, [89970,3,3] shape, 71976/17994 split sizes, 12 training/3 validation scenarios verified. Real arrays were used only for read-only integrity.

## CUDA Determinism Settings

Effective deterministic_algorithms=True; warn_only=False; cuDNN_deterministic=True; benchmark=False; matmul/cuDNN TF32=False; AMP and CPU/CUDA autocast=False; CUBLAS_WORKSPACE_CONFIG=:4096:8. Frozen settings were never relaxed. CPU threads=1; DataLoader workers=0.

## Forward Validation

All methods had finite five-row outputs, expected shape and finite BCE. Both GATs had finite normalized incoming attention, zero sum error and zero self-edge attention. Batched/generic CUDA outputs matched with measured max error 0 under existing atol=1e-7, rtol=1e-6. E=0 Standard/Epistemic output equality was bitwise exact in inference and matched-RNG training mode. Raising sender E from 0 to 2 reduced fixed-logit sender attention from 0.5 to 0.25 at both layers. CPU/GPU max output difference was 5.960464477539063e-08 for each method; cross-device bitwise equality was not required and no tolerance was introduced/loosened.

## CUDA Smoke Training

Only the frozen synthetic 128-row fixture was trained: 32 training rows and 96 validation rows. Each of three methods ran twice in separate clean processes at seed 101. Each completed 100 tiny epochs/100 Adam updates and selected epoch 100. Gradients and trajectories were finite; parameters updated; validation, checkpoint saving/reloading and row-keyed prediction export passed. The frozen specification was preserved. Fixture metrics remain engineering-only.

## GPU Reproducibility

Classification 1: bitwise exact for all three methods. Initial parameters, first dropout RNG, every row permutation and batch boundary, loss/validation histories, selected epoch, final parameters and optimizer, every saved checkpoint payload, best checkpoint, row-keyed prediction arrays and fixture metrics matched. Measured train/validation/prediction repeat errors were 0. Accepted repeat tolerance=0. No fallback or weaker settings. Downloaded checkpoint/prediction contents and repeat evidence were independently verified locally.

## Paired Standard/Epistemic Integrity

Explicit deepcopy/load_state_dict cloning gave byte-exact parameters with independent storage. Initial dropout RNG, Adam configuration, row orders/batches, E=0 training/validation histories, selected epoch, final parameter/optimizer states, every saved model checkpoint tensor and row-keyed probabilities matched exactly. The nonzero-E diagnostic used one fixed-logit model with identical features, changing only sender E.

## GPU Memory / Runtime

One synthetic 256-row forward/backward/Adam batch per method. Peak reserved memory: 69,206,016 bytes (66 MiB, 0.443% of reported device capacity). Peak allocated: NoGraph 67,251,200 bytes; both GATs 67,499,520 bytes. Observed batch times: NoGraph 0.232553451 s; StandardGAT 0.117867622 s; EpistemicGAT 0.029423070 s. These are single engineering observations including startup effects, not comparative performance benchmarks. Batch size 256 comfortably fits the frozen shape. No OOM, batch tuning, representative real-batch training or full-dataset epochs. nvidia-smi reported 0% utilization at environment inspection, before fixtures.

## Environment Lock

Original Kaggle environment_lock.json is preserved byte-for-byte, with its .sha256 seal and evidence bindings. SHA-256: 0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d.
Bundle SHA256SUMS seal: 61544ab57409112695582fbfe170a9d832a550ae35f11336eabde9da6a5cc1f1.
Downloaded full evidence archive SHA-256: d65837d1be6e2b8bb2566836ebd46b8e79d25fc7b0fe527e7d8fb7f40cd64d73.
All hashes were independently reverified locally. All six fixture run directories/checkpoints/history/predictions/metrics and logs are retained.

## Full-Run Commands Prepared

future_full_commands.md and future_full_plan.json define nograph, standard_gat and epistemic_gat for seeds 101,202,303,404,505. The future launcher verifies frozen source/artifact hashes, sealed environment lock, bound evidence, single-visible-T4 exposure, package/CUDA/cuDNN versions and deterministic settings before execution. Runs have independent directories and retain failures/checkpoints/history/row-keyed predictions/metrics; no seed replacement or best-seed selection. Actual Kaggle default preflight passed: all 15 runs printed, execution_requested=False, no training directory created. Commands are prepared only; full execution remains separately authorized and was not run. No scientific aggregation commands are invoked.

## Core / Artifact Integrity

276 preexisting files plus sealed upstream/N20/graph dependencies preserved. Graph, protocol, trainer bundle, model architecture, features, recipes, metrics, calibration, seeds and N=20 unchanged. No portability patch to frozen sources was needed. No official TEST, full graph training, conformal fit, scientific tuning/aggregation, commit or push. Frozen regression baseline: 608 passed from the previous milestone; six new engineering guard tests passed. No claim of rerunning all 608 tests in this milestone.

## Ready / Not Ready for Full Paired-Seed Experiment

READY: actual authorized single-visible-T4 CUDA engineering gates passed and the final lock is sealed. Full paired-seed execution is not authorized by this milestone and was not executed.

## Recommended Next Step

Review the sealed lock and engineering evidence. Separately authorize the full paired-seed milestone, then run the prepared gated launcher in the same validated environment and GPU-exposure configuration. Stop here; do not execute full runs or scientific aggregation in this milestone.

# Future full experiment commands — prepared, never executed

Status: DRAFT. Actual Kaggle single-T4 validation and a sealed execution lock are still required.

This preflight verifies the frozen bundle, graph, lock seal, evidence and actual environment, then prints all 15 planned runs:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1
```

Only in a separately authorized full experiment milestone, the exact all-run command is:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1 --execute-full-experiment
```

That command retains nograph, standard_gat and epistemic_gat for every seed 101, 202, 303, 404, 505.
The 15 equivalent independent invocations below are alternatives to the all-run command; do not run both.
Each creates its own run root and nested method/seed directory. Existing directories are refused.

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/nograph_101 --method nograph --seed 101 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/standard_gat_101 --method standard_gat --seed 101 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/epistemic_gat_101 --method epistemic_gat --seed 101 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/nograph_202 --method nograph --seed 202 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/standard_gat_202 --method standard_gat --seed 202 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/epistemic_gat_202 --method epistemic_gat --seed 202 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/nograph_303 --method nograph --seed 303 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/standard_gat_303 --method standard_gat --seed 303 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/epistemic_gat_303 --method epistemic_gat --seed 303 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/nograph_404 --method nograph --seed 404 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/standard_gat_404 --method standard_gat --seed 404 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/epistemic_gat_404 --method epistemic_gat --seed 404 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/nograph_505 --method nograph --seed 505 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/standard_gat_505 --method standard_gat --seed 505 --execute-full-experiment
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python /kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py --environment-lock /kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json --output /kaggle/working/cognix_full_paired_graph_v1_individual/epistemic_gat_505 --method epistemic_gat --seed 505 --execute-full-experiment
```

Frozen orchestration preserves checkpoints, history, row-keyed predictions, metrics and failures.
It does not replace a failed seed or choose a best seed. No scientific aggregation is invoked.

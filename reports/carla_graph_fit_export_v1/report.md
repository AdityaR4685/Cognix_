# Frozen FIT Graph Compact Export v1

## Frozen Protocol Verification

Protocol SHA-256: `68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203`. All sealed preregistration files and checksum-list seal verified unchanged before editing and after export. Existing N20 cache/handoff/source hashes match the sealed bindings. The 117 pre-existing untracked files and 193 tracked files remain byte-identical; all 20 frozen raw scenario directories remain available.

## Exporter Implementation

Added only `cognix/adapters/carla/graph_fit_export.py`, `tests/unit/test_carla_graph_fit_export.py`, and this report directory. Production entry point: `run_export.py --workers 3`. Three bounded local processes performed one full corrupted-image pass. Per-scenario compact checkpoints allow restart with identical dependency and code bindings. Existing outputs are never overwritten. No upstream fitting, training, acquisition, calibration changes or generic-framework edits.

## Clean Graph Construction

Every frozen FIT parent tick 1..2999 was considered. Accepted clean graphs copy the sealed N20 clean checkpoint in canonical node order Camera/IMU/Seg (handoff indices 0/3/1), with features [prob_normal, epistemic, aleatoric]. All retained clean node blocks were compared byte for byte with N20. The single schema-bound adjacency is [[0,1,1],[1,0,1],[1,1,0]], with no self-loops or spatial topology. Clean target_normal=1.

## FIT Pseudo Generation

Used unchanged `_pseudo_for_scenario(raw, cached_parent_features, scenario_id, frozen_five_recipe_tuple, 0, 1, None)`. Rotation is recipes[t % 5]; base seed 0, call seed t, existing defaults and causal window 12. Only the 15 sealed FIT scenarios were passed to raw/pseudo generation; no CAL pseudo arrays were reused. IMU corruption consumes [max(0,t-11),t]; Camera/Seg corruption consumes tick t. Pseudo target_normal=0. GNSS contributes neither primary nodes nor recipes nor calibrated exported quantities.

## Untouched-Node Integrity

Restored five production Mahalanobis bootstrap members and the accepted shared calibrator for each primary modality from hash-verified state, with no `.fit` call. The affected compact block passes through the unchanged production agent normality scores, member calibration and canonical Bernoulli UQ. Untouched compact blocks are verified byte-identical to cached parents; untouched p/E/A nodes are copied directly and verified byte-identical for every pair. Focused spies prove five member-score and five calibration calls per accepted pseudo, only for its affected modality. Clean inference anchors and a ten-parent real FIT recipe replay across train/validation agree with the export.

## No-Effect / Pairing Results

The unchanged compact-feature np.allclose(rtol=1e-12,atol=1e-12) guard skipped 0 parents, removing both pair members without replacements. Skip decisions precede node inference and do not depend on predictions. 15 material-feature corruptions had identical affected-node p/E/A; these remain in the dataset and collision ledger. Every retained pair has exactly one clean target 1 and one pseudo target 0, a stable protocol-bound SHA256 pair ID, the same parent scenario/tick/split, and one affected modality.

## Graph Dataset Counts

| Quantity | Count |
|---|---:|
| candidate_parents | 44,985 |
| retained_parents | 44,985 |
| skipped_parents | 0 |
| clean_rows | 44,985 |
| pseudo_rows | 44,985 |
| graph_rows | 89,970 |
| node_output_collisions | 15 |
| GRAPH_TRAIN rows | 71,976 |
| GRAPH_VALIDATION rows | 17,994 |

Preregistered row maxima are 71,976 train and 17,994 validation. Realized counts follow only eligibility and the frozen guard; maxima are never forced.

## Per-Recipe / Per-Scenario Counts

Counts below are retained pseudo graphs (one matching clean graph each).

| Recipe | Train | Validation | Total |
|---|---:|---:|---:|
| camera_brightness_shift | 7,188 | 1,797 | 8,985 |
| camera_occlusion | 7,200 | 1,800 | 9,000 |
| seg_region_corruption | 7,200 | 1,800 | 9,000 |
| imu_spike | 7,200 | 1,800 | 9,000 |
| imu_bias_scale | 7,200 | 1,800 | 9,000 |

| Modality | Pseudo graphs |
|---|---:|
| Camera | 17,985 |
| IMU | 18,000 |
| Seg | 9,000 |

| Scenario | Split | Candidates | Retained pairs | Skipped | Clean | Pseudo | Rows |
|---|---|---:|---:|---:|---:|---:|---:|
| Town01/scenario-10 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-11 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-12 | GRAPH_VALIDATION | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-13 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-14 | GRAPH_VALIDATION | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-15 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-2 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-3 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-5 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-6 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town01/scenario-7 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town02/scenario-1 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town02/scenario-10 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town02/scenario-2 | GRAPH_VALIDATION | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |
| Town02/scenario-4 | GRAPH_TRAIN | 2,999 | 2,999 | 0 | 2,999 | 2,999 | 5,998 |

`counts.json` also records per-recipe/per-modality counts within each scenario and split.

## Compact Artifact Schema

Artifact directory: `C:\Users\Aditya\AppData\Local\Temp\carla_graph_fit_export_v1`. `graphs.npz` contains no pickle/object arrays, raw frames, archive bytes, GNSS calibrated values, model parameters or raw feature blocks. Provenance common to all rows and dictionaries live in `schema.json`; graph row fields reconstruct using array indices and parent scenario/tick. Scientific storage remains float64. The sealed per-graph hash retains its specified float32 model-view bytes plus float64 canonical epistemic side channel; the separate artifact content hash covers the full float64 storage.

| Array | Dtype | Shape |
|---|---|---|
| calibration_valid | bool | [3] |
| corrupted_node_index | int8 | [89970] |
| corruption_end_tick | int32 | [89970] |
| corruption_start_tick | int32 | [89970] |
| graph_content_sha256 | <U64 | [89970] |
| is_pseudo | bool | [89970] |
| node_features | float64 | [89970, 3, 3] |
| pair_id | <U64 | [89970] |
| parent_scenario_index | uint16 | [89970] |
| parent_tick | int32 | [89970] |
| recipe_index | int8 | [89970] |
| recipe_seed | int64 | [89970] |
| row_key | <U64 | [89970] |
| scenario_index | uint16 | [89970] |
| severity | float64 | [89970] |
| split | uint8 | [89970] |
| target_normal | uint8 | [89970] |
| target_normal_per_node | uint8 | [89970, 3] |
| tick | int32 | [89970] |
| town_index | uint8 | [89970] |
| window_end_tick | int32 | [89970] |
| window_start_tick | int32 | [89970] |

Clean recipe/seed/node indices and actual corruption bounds use -1; clean severity uses NaN. Reconstructed scientific recipe/severity/seed/modality are null. Window bounds always describe parent causal provenance. For the sealed graph hash, clean row provenance carries the deterministically selected recipe scope in corruption_start_tick/corruption_end_tick (the same scope as its paired pseudo); actual compact clean corruption bounds remain -1. `validate_export.py` reconstructs and checks every row hash from these rules.

## Artifact Hashes

Scientific artifact content SHA-256: `baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237`.

Schema file SHA-256: `5b678571d4ed112e5756e1364ff7966567a1338d87143ca120b3aede24d768fd`.

Manifest file SHA-256: `9f8c971b72ac260b9f48b7f3ab45561d62535c22002d38b53cf4712536f6e81d`.

| Artifact file | SHA-256 |
|---|---|
| counts.json | `44e98889b3de49f4840a35a3be9201977a0954ce0763a3a6cae91ec32fdf78fc` |
| graphs.npz | `5ba30ad81f6cd23b78d2e51ac094e05c7201d30b47cf2e76fe2a64940ffba20e` |
| node_output_collisions.json | `1d6ca434b59e1f6548d96f96826e88e9bd984f6379b961cf8f23327a1c2c07a7` |
| schema.json | `5b678571d4ed112e5756e1364ff7966567a1338d87143ca120b3aede24d768fd` |
| skip_ledger.json | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |
| upstream_dependencies.json | `4505874649696afec84b71b9ab8d74fda0ce53cd6e0d308554ccc32cb6a4ab11` |

N20 cache scientific content SHA-256: `c2df480a712e819d85437a9bb141bca91b47b68e90ae84fcf78d9151b93b5539`. Complete upstream dependency hashes are in `upstream_dependencies.json`; sealed protocol identities are embedded in schema common provenance.

## Focused Tests

43 passed, zero failures. Covers every requested exporter integrity category, using small local fixtures, deterministic recipe replay, fit-blocking spies, invalid/null mapping rejection, exact untouched nodes, collision retention, causal scope and serialization/content hash checks. Evidence: `focused_tests.txt`.

## Broad Regression

Ran the requested command with PYTHONHASHSEED=0: `py -3.12 -m pytest tests/unit tests/regression tests/integration tests/test_interfaces.py tests/test_mathematics.py -q`. Result: 548 passed (505 accepted baseline + 43 exporter tests), zero failures. Evidence: `broad_tests.txt`.

## Core / N=20 Integrity

All 193 existing tracked files and 117 pre-existing untracked files were preserved byte for byte. Sealed preregistration, N20 cache/handoff, calibration, feature, recipe, generic graph/core and frozen synthetic sources remain unchanged. FIT whitelist gates precede raw reads; official TRAIN raw paths only. No TEST, CAL graph rows, training, conformal, acquisition, scientific tuning, commits or pushes. Full file inventory is in `file_inventory.json`; original intentional untracked inventory is in `initial_audit.json`. Generated compact/checkpoint files are included explicitly.

New repository files:

- `cognix/adapters/carla/graph_fit_export.py`
- `reports/carla_graph_fit_export_v1/broad_tests.txt`
- `reports/carla_graph_fit_export_v1/export_console.txt`
- `reports/carla_graph_fit_export_v1/export_result.json`
- `reports/carla_graph_fit_export_v1/file_inventory.json`
- `reports/carla_graph_fit_export_v1/focused_tests.txt`
- `reports/carla_graph_fit_export_v1/initial_audit.json`
- `reports/carla_graph_fit_export_v1/integrity_console.txt`
- `reports/carla_graph_fit_export_v1/integrity_results.json`
- `reports/carla_graph_fit_export_v1/report.md`
- `reports/carla_graph_fit_export_v1/run_export.py`
- `reports/carla_graph_fit_export_v1/test_coverage.json`
- `reports/carla_graph_fit_export_v1/validate_export.py`
- `reports/carla_graph_fit_export_v1/write_report.py`
- `tests/unit/test_carla_graph_fit_export.py`

## Ready / Not Ready for Trainer Implementation

READY for the next separately authorized trainer implementation milestone: exported FIT data and integrity gates pass.

## Ready / Not Ready for Kaggle Training

NOT READY: no graph trainer, minibatching/early-stopping implementation or locked Kaggle execution environment has been implemented or validated. No upload or execution was performed.

## Recommended Next Step

Authorize only the frozen adapter/report-side trainers and their integrity tests against this validated artifact. Keep TEST and conformal outside that milestone, and do not execute Kaggle training before trainer validation and separate execution authorization.

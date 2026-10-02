"""Preservation/reproducibility audit and requested milestone report."""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla import graph_training as tr
from cognix.adapters.carla import graph_training_data as gd


def read(path):
    return json.loads(Path(path).read_text())


def main():
    initial = read(OUT / "initial_audit.json")
    changed = [rel for rel, h in initial["repository_sha256"].items()
               if not (ROOT / rel).is_file() or ge.sha256_file(ROOT / rel) != h]
    ge.require(not changed, "pre-existing repository work changed")
    for path, h in initial["graph_artifact_sha256"].items():
        ge.require(ge.sha256_file(path) == h, "graph artifact changed")
    ge.require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == initial["git_head"], "git HEAD changed")
    ge.verify_frozen_dependencies(ROOT)
    data = gd.load_graph_dataset(Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1", ROOT / "reports/carla_gat_preregistration_v1")
    lock = read(OUT / "proposed_environment_lock_validated.json")
    ge.require(lock["trainer_identity"] == tr.trainer_identity(), "current trainer identity differs from proposed lock")
    focused = (OUT / "focused_tests_validated.txt").read_text()
    broad = (OUT / "broad_tests.txt").read_text()
    ge.require("60 passed" in focused and "608 passed" in broad
               and not re.search(r"\d+ failed|\d+ errors", broad), "final test evidence incomplete")
    smoke_root = Path(os.environ["TEMP"]) / "carla_graph_trainers_v1_smoke_validated"
    repeat_root = Path(os.environ["TEMP"]) / "carla_graph_trainers_v1_repeat_validated"
    smoke = {}
    for method in gd.METHODS:
        directory = smoke_root / method / "seed_101"
        summary, history = read(directory / "run_summary.json"), read(directory / "history.json")
        ge.require(summary["data_kind"] == "synthetic_fixture" and summary["trainer_identity"] == tr.trainer_identity()
                   and summary["finite_gradients"] and summary["parameters_updated"], "final smoke validation")
        ge.require(history[-1]["train_BCE"] < history[0]["train_BCE"], "learnable fixture loss did not decrease")
        splits = read(ROOT / "reports/carla_gat_preregistration_v1/split_manifest.json")
        fixture = gd.make_smoke_fixture(splits)
        model, _, payload = tr.load_checkpoint(summary["selected_checkpoint"], fixture)
        prediction = tr.evaluate(model, fixture)
        with np.load(directory / "predictions.npz", allow_pickle=False) as z:
            ge.require(np.array_equal(z["q_normal"], prediction) and np.array_equal(z["row_key"], fixture.arrays["row_key"]), "persistent checkpoint/prediction roundtrip")
        if method != "nograph":
            with np.load(directory / "validation_attention.npz", allow_pickle=False) as z:
                alpha = z["attention"]
                ge.require(np.isfinite(alpha).all() and np.allclose(alpha.sum(-1), 1, atol=1e-6)
                           and np.all(np.diagonal(alpha, axis1=-2, axis2=-1) == 0), "persisted attention integrity")
        smoke[method] = {"seed": 101, "dataset_rows": len(fixture.arrays["target_normal"]),
            "train_rows": len(fixture.train_indices), "validation_rows": len(fixture.validation_indices),
            "epochs_executed": summary["epochs_executed"], "selected_epoch": summary["selected_epoch"],
            "first_training_BCE": history[0]["train_BCE"], "last_training_BCE": history[-1]["train_BCE"],
            "finite_gradients": True, "parameters_updated": True, "checkpoint_prediction_roundtrip": True,
            "checkpoint_content_sha256": summary["checkpoint_content_sha256"],
            "prediction_content_sha256": summary["prediction_content_sha256"], "output": str(directory)}
    reference = smoke_root / "standard_gat/seed_101"
    repeat = repeat_root / "standard_gat/seed_101"
    ra, rb = read(reference / "run_summary.json"), read(repeat / "run_summary.json")
    ge.require(ra["checkpoint_content_sha256"] == rb["checkpoint_content_sha256"]
               and ra["prediction_content_sha256"] == rb["prediction_content_sha256"], "independent repeat differs")
    for name in ("history.json", "metrics.json"):
        ge.require(read(reference / name) == read(repeat / name), "repeat history/metrics differ")
    ge.require(read(reference / "history.json") == read(smoke_root / "epistemic_gat/seed_101/history.json"), "E=0 training equivalence")
    evidence = {"status": "passed", "changed_preexisting_files": changed, "preserved_repository_files": len(initial["repository_sha256"]),
        "preserved_intentional_untracked_files": len(initial["intentional_untracked_files"]),
        "artifact_scientific_sha256": data.scientific_sha256, "protocol_sha256": ge.PROTOCOL_SHA256,
        "trainer_identity": tr.trainer_identity(), "smoke_runs": smoke,
        "independent_standard_repeat": {"seed": 101, "exact_loss_history": True, "exact_selected_epoch": True,
            "exact_checkpoint_contents": True, "exact_predictions": True, "exact_metrics": True, "output": str(repeat)},
        "CPU_fixture_reproducibility": "exact; all three methods repeated in focused tests",
        "CUDA_fixture_reproducibility": "not verified; local CUDA unavailable",
        "official_TEST_access": False, "full_graph_training": False, "Kaggle_execution": False,
        "conformal_fit": False, "scientific_tuning": False, "commit_push": False,
        "secondary_GNSS_condition": "deferred", "focused_passed": 60, "broad_passed": 608, "failures": 0}
    ge.save_json(OUT / "integrity_results.json", evidence)
    coverage = {
        "loader_hash_verification": "test_loader_verified_scientific_hash_and_source_precision",
        "node_feature_order": "test_loader_verified_scientific_hash_and_source_precision",
        "split_isolation": "test_scenario_split_isolation_and_no_cal",
        "NoGraph_48": "test_frozen_parameter_counts_and_biases", "StandardGAT_50": "test_frozen_parameter_counts_and_biases",
        "EpistemicGAT_50": "test_frozen_parameter_counts_and_biases",
        "identical_paired_initialization": "test_paired_initialization_is_explicit_byte_exact",
        "E_zero_equivalence": "test_zero_E_equivalence_with_matched_rng",
        "sender_E_suppression": "test_higher_E_suppresses_sender_at_both_layers_with_fixed_logits",
        "no_self_edge": "test_attention_incoming_normalization_no_self_edge",
        "incoming_attention_normalization": "test_attention_incoming_normalization_no_self_edge",
        "deterministic_epoch_shuffle": "test_deterministic_shuffle_and_batch_boundaries",
        "Standard_Epi_identical_batches": "test_standard_epistemic_same_batches_and_train_isolation",
        "BCE_normal_target": "test_BCE_normal_target_direction",
        "scenario_macro_BCE": "test_scenario_macro_not_pooled_controls_checkpoint",
        "early_stop_patience_min_delta": "test_early_stopping_dual_trackers_patience_and_min_delta",
        "earliest_exact_tie": "test_earliest_exact_tie_and_strict_min_delta",
        "AUROC_corruption_orientation": "test_AUROC_orientation_AP_and_ties",
        "F1_largest_threshold_tie": "test_threshold_largest_exact_tie_and_boundary",
        "ECE_15_boundaries": "test_ECE_exact_boundaries_zero_one_and_empty_bins",
        "undefined_null_reason": "test_undefined_metrics_null_with_reason_and_no_dropped_scenarios",
        "checkpoint_roundtrip": "test_checkpoint_restore_and_row_keyed_predictions",
        "row_key_alignment": "test_checkpoint_restore_and_row_keyed_predictions",
        "deterministic_repeat": "test_smoke_loss_gradients_updates_validation_and_exact_repeat",
        "no_CAL_optimization": "test_standard_epistemic_same_batches_and_train_isolation",
        "no_TEST_path": "test_no_TEST_raw_loader_or_CAL_code_path",
        "artifact_protocol_hash_refusal": "test_loader_hash_refusal",
        "full_run_protection": "test_full_training_protected_before_model_creation"}
    test_path = ROOT / "tests/unit/test_carla_graph_trainers.py"
    names = {n.name for n in ast.parse(test_path.read_text()).body if isinstance(n, ast.FunctionDef)}
    ge.require(set(coverage.values()) <= names, "coverage map references missing tests")
    ge.save_json(OUT / "test_coverage.json", {"required_categories": coverage, "test_file": str(test_path), "passed_cases": 60})
    command_doc = """# Future execution commands (not executed in this milestone)

All full commands require separately authorized execution and an observed environment lock.
The lock must pass artifact/source verification and deterministic CUDA fixture repeats on one T4.
Replace the placeholders with existing paths; no upload commands are supplied.

```text
python reports/carla_graph_trainers_v1/validate_environment.py --validate-kaggle --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --cuda-smoke-output <NEW_CUDA_FIXTURE_DIR> --output <NEW_VALIDATED_LOCK_FILE>
python reports/carla_graph_trainers_v1/train.py train --method nograph --seed 101 --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_NOGRAPH_RUN_ROOT>
python reports/carla_graph_trainers_v1/train.py train --method standard_gat --seed 101 --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_STANDARD_RUN_ROOT>
python reports/carla_graph_trainers_v1/train.py train --method epistemic_gat --seed 101 --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_EPISTEMIC_RUN_ROOT>
python reports/carla_graph_trainers_v1/train.py all --execute-full --device cuda --artifact <GRAPH_ARTIFACT_DIR> --preregistration <PREREGISTRATION_DIR> --environment-lock <VALIDATED_LOCK_FILE> --output <NEW_ALL_SEED_RUN_ROOT>
```

The all-run plan retains seeds 101,202,303,404,505, in that order, with all three primary methods.
Every run has its own method/seed directory; failed runs and reasons remain in orchestration.json.
No replacement seed, best-seed selection, implicit overwrite or retry occurs.
`all --smoke` runs only seed 101 on a synthetic 128-row engineering fixture.
Neither the original nor validated smoke outputs train on the 89,970-row artifact.
"""
    with (OUT / "future_commands.md").open("x", encoding="utf-8") as f:
        f.write(command_doc)
    old = set(initial["repository_sha256"])
    now = set(subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines())
    new = sorted((now - old) | {"reports/carla_graph_trainers_v1/file_inventory.json", "reports/carla_graph_trainers_v1/report.md"})
    roots = [Path(os.environ["TEMP"]) / name for name in ("carla_graph_trainers_v1_smoke", "carla_graph_trainers_v1_smoke_validated", "carla_graph_trainers_v1_repeat_validated")]
    external = [str(p) for r in roots for p in sorted(r.rglob("*")) if p.is_file()]
    ge.save_json(OUT / "file_inventory.json", {"changed_preexisting_files": changed, "new_repository_files": new,
        "new_persistent_smoke_files": external, "preserved_initial_candidate_smoke_root": str(roots[0]),
        "validated_smoke_root": str(smoke_root), "validated_repeat_root": str(repeat_root),
        "intentional_untracked_inventory": "initial_audit.json::intentional_untracked_files",
        "pytest_temporary_files": "small fixture checkpoints/predictions are produced under pytest's temporary directories; no user data is altered"})
    write_report(evidence, lock, new, len(external))
    print(json.dumps({"report": str(OUT / "report.md"), "integrity": "passed", "new_repository_files": len(new),
                      "preserved_repository_files": len(initial["repository_sha256"]), "focused": 60, "broad": 608}))


def write_report(evidence, lock, new, external_count):
    lines = ["# Frozen Graph Trainers v1", ""]
    def section(title, text):
        lines.extend(["## " + title, "", text, ""])
    section("Frozen Trainer Specification Verification",
        f"Protocol `{ge.PROTOCOL_SHA256}` and graph scientific artifact `{gd.ARTIFACT_SHA256}` match the frozen identities. All sealed files, graph file/schema/manifest hashes and N20 dependencies were verified before editing and after validation. No model/data/split/metric decisions were changed. Trainer bundle SHA-256: `{evidence['trainer_identity']['bundle_sha256']}`.")
    section("Dataset Loader",
        "Read-only float64 [89970,3,3] scientific storage; Camera/IMU/Seg and [prob_normal,epistemic,aleatoric]. New float32 model inputs and separate raw float64 E are materialized per batch. Exact frozen FIT split, pair/row provenance, six directed edges and zero diagonal verified. No tick resplitting, upstream refitting or raw/TEST loader. Training indices contain only GRAPH_TRAIN; all three frozen validation scenarios are required.")
    section("NoGraph Implementation",
        "Shared bias-free Linear(3,12) -> ELU -> hidden Dropout(0.1) -> bias-free Linear(12,1), with Xavier initialization and mean node sigmoid pooling. Exactly 48 trainable parameters, no message passing. Generic identity NoGraph remains unchanged.")
    section("StandardGAT Implementation",
        "Two unchanged generic EpistemicGATLayerPT instances, 3 -> 8 -> 1, single head, 50 parameters. Adapter-side batch algebra uses the original W/a, LeakyReLU(0.2), ELU hidden output, linear final output and post-softmax dropout 0.1. Standard prior is all ones. Batched/single-graph execution agrees with the generic forward path within atol=1e-7, rtol=1e-6; generic code remains unchanged.")
    section("EpistemicGAT Implementation",
        "The identical 50-parameter architecture uses the original sender's canonical E at both layers. Prior is float64 1/(1+E), stored float32 before the unchanged log operation, matching generic compute_epistemic_weights including neutral rounding for tiny E. No scaling, normalization, learned strength or amplification. Nonfinite/negative E rejected. E=0 gives exact matched-weight/RNG equivalence in inference and fixture training.")
    section("Paired Initialization / Batch Integrity",
        "StandardGAT state is explicitly deep-cloned into EpistemicGAT, with byte equality and independent storage tested for each of 101,202,303,404,505. NoGraph uses the corresponding seed. All methods reset dropout RNG before optimization. PCG64(seed+zero-based epoch) permutations and 256-row boundaries are independent of Torch RNG and identical for paired methods. Adam lr=0.001, weight_decay=0.0001, max 100 epochs, no scheduler, AMP, TF32 or clipping.")
    section("Early Stopping",
        "Equal-scenario validation BCE is the arithmetic mean of three scenario means, with no pooled substitute. Tests demonstrate opposing pooled/macro trends. Patience 10 resets only when loss < tracked_best-1e-5. Independently save every strict observed minimum, including improvements smaller than min_delta; exact ties retain the earliest epoch. Unit sequences exercise stopping, tiny improvements and ties. Learnable smoke runs used their permitted 100-epoch maximum.")
    section("Metrics / Thresholding",
        "Implemented pooled, per-scenario, equal-scenario macro and per-recipe paired clean/pseudo diagnostics for corruption AUROC, stepwise AP/AUPRC, F1, accuracy, balanced accuracy, normal-probability Brier/BCE and 15-bin ECE. Threshold uses only graph validation, candidates unique p_corrupt plus {0,1}, macro F1 objective, largest threshold on exact ties, and >= comparison. Undefined class-dependent metrics are null with reasons; the frozen zero-denominator F1 rule explicitly returns 0. ECE uses [k/15,(k+1)/15), with final bin including 1 and zero contribution from empty bins. Frozen five-pair summaries, exact sign flips and whole-scenario bootstrap helpers are implemented but no scientific experiment aggregate was calculated.")
    section("Checkpoint / Prediction Schema",
        "Exclusive per-method/seed directories retain checkpoint_epoch_NNN.pt for observed minima. Metadata binds method, seed, epoch, selected epoch, best macro BCE, model/training configuration, graph/protocol/split and trainer source identity, plus explicit synthetic/real training-data identity. Saved model/Adam state, early-stop trackers, history and Python/NumPy/Torch/available-CUDA RNG states support exact continuation; epoch-50 continuation reproduced the full fixture trajectory. A deterministic scientific content hash is separate from Torch ZIP bytes. Row-keyed predictions include row_key, pair_id, scenario, tick, target_normal, recipe (empty string for clean), split, method, seed, selected epoch, checkpoint content hash, q_normal and p_corrupt. No checkpoint overwrites are permitted.")
    section("Attention Diagnostics",
        "Optional validation_attention.npz records row-keyed, audit-only pre-dropout incoming attention [rows,2 layers,3 receivers,3 senders]; single head is implicit. Finite values, incoming sum 1 and zero diagonal verified. Fixed-logit fixtures show increased sender E reduces its contribution at both layers. Attention is never optimized as a diagnostic objective or selected by visual favorability.")
    section("Smoke Training Results", "Only synthetic engineering fixtures were trained: seed 101, 128 rows (32 train, 96 validation across three scenarios). These results are implementation evidence and do not measure COGNIX performance. Earlier candidate smoke outputs remain preserved separately.")
    lines.extend(["| Method | Epochs | Selected epoch | First / last train BCE | Finite gradients | Updated |", "|---|---:|---:|---|---|---|"])
    for method, r in evidence["smoke_runs"].items():
        lines.append(f"| {method} | {r['epochs_executed']} | {r['selected_epoch']} | {r['first_training_BCE']:.8f} / {r['last_training_BCE']:.8f} | yes | yes |")
    lines.append("")
    section("Determinism",
        "All three methods were repeated in focused tests: exact epoch order/loss trajectory, selected epoch, checkpoint scientific contents, predictions and metrics. An independent persistent StandardGAT CLI repeat is also exact. CPU uses one thread and deterministic algorithms with errors enabled. CUDA was not exercised locally; no CUDA operation's determinism is claimed. Future CUDA smoke validation must pass before full execution; nondeterministic-operation errors stop without fallback.")
    local = lock["local_observed"]
    section("Kaggle Environment Lock",
        f"Proposed lock records actual local Python {local['Python']}, PyTorch {local['PyTorch']}, NumPy {local['NumPy']}, installed scikit-learn {local['scikit_learn']} (not used), CUDA availability={local['CUDA_available']}, CUDA runtime={local['torch_CUDA_runtime']}, cuDNN={local['cuDNN_version']}. No Kaggle package versions are invented. Deterministic algorithms, cuDNN deterministic=true/benchmark=false, both TF32 settings false, AMP=false and CUBLAS_WORKSPACE_CONFIG=:4096:8 are recorded. validate_environment.py prints/checks actual future Kaggle versions, verifies hashes, and optionally runs deterministic GPU fixture repeats on one T4 to create a validated execution lock. Full commands are supplied in future_commands.md but were not invoked.")
    section("Focused Tests", "60 passed, zero failures, covering all 28 requested categories plus source-hash refusal, oversized smoke protection, generic forward equivalence, checkpoint continuation and frozen statistics helpers. Evidence: focused_tests_validated.txt and test_coverage.json.")
    section("Broad Regression", "Ran PYTHONHASHSEED=0 py -3.12 -m pytest tests/unit tests/regression tests/integration tests/test_interfaces.py tests/test_mathematics.py -q. 608 passed (548 baseline + 60 trainer cases), zero failures, 14 baseline warnings. Evidence: broad_tests.txt.")
    section("Core / Artifact Integrity",
        f"All {evidence['preserved_repository_files']} pre-existing repository files and {evidence['preserved_intentional_untracked_files']} intentional untracked files remain unchanged. The float64 graph artifact, preregistration, N20 cache/handoff, exporter, calibration, features, recipes, generic core/graphs and frozen synthetic benchmark retain their hashes. No official TEST, CAL optimization, full graph training, Kaggle execution/upload, conformal, scientific tuning, commits or pushes. GNSS secondary condition remains deferred. File inventory contains {len(new)} new repository files and {external_count} persistent synthetic smoke/checkpoint files; initial work inventory is retained in initial_audit.json.")
    lines.extend(["New repository files:", ""] + ["- `" + name + "`" for name in new] + [""])
    section("Ready / Not Ready for Full Paired-Seed Run",
        "Trainer implementation is validated on CPU fixtures. Full paired execution is NOT READY to launch in this milestone: it requires a future validated single-T4 runtime/CUDA fixture lock and separately authorized execution. The full-run guard is active; no five-seed experiment was launched.")
    section("Ready / Not Ready for Kaggle",
        "NOT READY for execution/upload. Actual Kaggle versions and CUDA fixture reproducibility remain unverified. The proposed lock and future validation/execution commands are prepared.")
    section("Recommended Next Step",
        "Authorize only future Kaggle environment and CUDA fixture validation first. Once that lock passes, separately authorize the frozen full paired-seed run. Preserve every planned seed and failure; keep TEST, conformal and secondary GNSS outside that execution.")
    with (OUT / "report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

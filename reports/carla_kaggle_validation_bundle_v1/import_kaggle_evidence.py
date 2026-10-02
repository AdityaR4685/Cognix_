"""Verify/downloaded engineering evidence; never trains or aggregates science."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT / "reports/carla_kaggle_environment_lock_v1"
sys.path.insert(0, str(ROOT))
EXPECTED_ARCHIVE = "d65837d1be6e2b8bb2566836ebd46b8e79d25fc7b0fe527e7d8fb7f40cd64d73"
EXPECTED_LOCK = "0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    require(digest(args.archive) == EXPECTED_ARCHIVE, "downloaded archive differs from Kaggle-observed checksum")
    require(not DEST.exists(), "preserve existing lock/evidence directory")
    DEST.mkdir()
    with zipfile.ZipFile(args.archive) as archive:
        require(archive.testzip() is None, "archive CRC failure")
        for info in archive.infolist():
            require((DEST / info.filename).resolve().is_relative_to(DEST.resolve()), "unsafe archive path")
        archive.extractall(DEST)
    shutil.copyfile(args.archive, HERE / "kaggle_evidence_download.zip")
    lock = read(DEST / "environment_lock.json")
    require(digest(DEST / "environment_lock.json") == EXPECTED_LOCK ==
            (DEST / "environment_lock.json.sha256").read_text().split()[0], "environment lock checksum")
    for name, expected in lock["evidence_sha256"].items():
        require(digest(DEST / name) == expected and read(DEST / name)["passed"], "evidence hash/gate: " + name)
    sys.path.insert(0, str(HERE / "bundle/reports/carla_kaggle_validation_bundle_v1"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("packaged_validation", HERE / "bundle/reports/carla_kaggle_validation_bundle_v1/validate_kaggle.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    current = harness.integrity()
    require(current == read(DEST / "hash_verification.json"), "downloaded frozen verification differs from local current bundle")
    import numpy as np
    import torch
    from cognix.adapters.carla import graph_training as tr
    from cognix.adapters.carla import graph_training_data as gd
    fixture = gd.make_smoke_fixture(read(ROOT / "reports/carla_gat_preregistration_v1/split_manifest.json"))
    results = {}
    for method in gd.METHODS:
        first = DEST / "smoke_runs" / method / "first"
        repeat = DEST / "smoke_runs" / method / "repeat"
        for directory in (first, repeat):
            observer = read(directory / "observer.json")
            for name, expected in observer["checkpoint_payload_hashes"].items():
                payload = torch.load(directory / name, map_location="cpu", weights_only=False)
                require(tr.recursive_content_hash(payload) == expected, "downloaded checkpoint payload: " + name)
                checkpoint_content = payload.pop("checkpoint_content_sha256")
                require(tr.recursive_content_hash(payload) == checkpoint_content, "checkpoint internal seal")
            summary = observer["summary"]
            # Local path substitution only for locating the unchanged downloaded bytes.
            selected = directory / Path(summary["selected_checkpoint"]).name
            _, _, payload = tr.load_checkpoint(selected, fixture, device="cpu")
            require(payload["checkpoint_content_sha256"] == summary["checkpoint_content_sha256"], "selected checkpoint hash")
            with np.load(directory / "predictions.npz", allow_pickle=False) as z:
                require(tr.recursive_content_hash(dict(z)) == summary["prediction_content_sha256"], "prediction content hash")
                require(np.array_equal(z["row_key"], fixture.arrays["row_key"]), "prediction row-key order")
        results[method] = harness.compare_runs(first, repeat)
        require(results[method]["passed"], "independent downloaded repeat comparison")
    environment = lock["validated_execution_environment"]
    require(environment["devices"] == ["Tesla T4"] and environment["CUDA_VISIBLE_DEVICES"] == "0"
            and environment["physical_GPU_count_observed"] == 2, "authorized T4 exposure evidence")
    require(all(environment[key] for key in ("CUDA_available", "deterministic_algorithms", "cuDNN_deterministic"))
            and not any(environment[key] for key in ("deterministic_warn_only", "cuDNN_benchmark", "matmul_TF32",
                            "cuDNN_TF32", "AMP", "autocast_CPU_enabled", "autocast_CUDA_enabled")), "actual deterministic settings")
    require(environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "actual CUBLAS settings")
    snapshot = read(HERE / "preservation_snapshot.json")
    for relative, expected in snapshot["files"].items():
        require(digest(ROOT / relative) == expected, "preexisting file changed: " + relative)
    for group in ("upstream", "artifact"):
        for name, expected in snapshot[group].items():
            path = ROOT / "reports/carla_gat_preregistration_v1/protocol.json" if name == "protocol.json" else (
                   Path(name) if Path(name).is_absolute() else ROOT / name)
            require(digest(path) == expected, "upstream/artifact changed: " + name)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == snapshot["git_head"], "HEAD changed")
    verification = {"passed": True, "downloaded_archive_sha256": EXPECTED_ARCHIVE,
        "environment_lock_sha256": EXPECTED_LOCK, "all_bound_evidence_files_verified": True,
        "all_downloaded_checkpoint_internal_and_observer_hashes_verified": True,
        "all_prediction_content_hashes_verified": True, "independent_repeat_comparisons": results,
        "preserved_preexisting_files": len(snapshot["files"]), "changed_preexisting_files": [],
        "N20_unchanged": True, "graph_artifact_unchanged": True, "protocol_unchanged": True,
        "trainer_bundle_unchanged": True, "TEST_access": False, "full_graph_training": False,
        "conformal_fit": False, "scientific_tuning": False, "scientific_aggregation": False,
        "commit_push": False, "local_CPU_GPU_equality_demanded": False}
    write(DEST / "local_download_verification.json", verification)
    # Original Kaggle report is retained. This local report adds measured details.
    shutil.copyfile(DEST / "report.md", DEST / "kaggle_original_report.md")
    smoke = read(DEST / "cuda_smoke_results.json")
    resource = read(DEST / "resource_results.json")
    report = """## Kaggle Runtime

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

future_full_commands.md and future_full_plan.json define nograph, standard_gat and epistemic_gat for seeds 101,202,303,404,505. The future launcher verifies frozen source/artifact hashes, sealed environment lock, bound evidence, single-visible-T4 exposure, package/CUDA/cuDNN versions and deterministic settings before execution. Runs have independent directories and retain failures/checkpoints/history/row-keyed predictions/metrics; no seed replacement or best-seed selection. Commands are prepared only; full execution remains separately authorized and was not run. No scientific aggregation commands are invoked.

## Core / Artifact Integrity

276 preexisting files plus sealed upstream/N20/graph dependencies preserved. Graph, protocol, trainer bundle, model architecture, features, recipes, metrics, calibration, seeds and N=20 unchanged. No portability patch to frozen sources was needed. No official TEST, full graph training, conformal fit, scientific tuning/aggregation, commit or push. Frozen regression baseline: 608 passed from the previous milestone; six new engineering guard tests passed. No claim of rerunning all 608 tests in this milestone.

## Ready / Not Ready for Full Paired-Seed Experiment

READY: actual authorized single-visible-T4 CUDA engineering gates passed and the final lock is sealed. Full paired-seed execution is not authorized by this milestone and was not executed.

## Recommended Next Step

Review the sealed lock and engineering evidence. Separately authorize the full paired-seed milestone, then run the prepared gated launcher in the same validated environment and GPU-exposure configuration. Stop here; do not execute full runs or scientific aggregation in this milestone.
"""
    (DEST / "report.md").write_text(report, encoding="utf-8")
    for name in ("future_full_commands.md", "future_full_plan.json"):
        content = (HERE / name).read_text()
        content = content.replace("DRAFT_REQUIRES_ACTUAL_VALIDATED_CUDA_LOCK", "PREPARED_WITH_VALIDATED_CUDA_LOCK")
        content = content.replace("Status: DRAFT. Actual Kaggle single-T4 validation and a sealed execution lock are still required.",
                                  "Status: PREPARED. Actual CUDA validation passed; lock SHA-256 " + EXPECTED_LOCK + ". Full execution requires separate authorization.")
        (DEST / name).write_text(content, encoding="utf-8")
    # Additional local final-report seal, without changing the original Kaggle lock.
    seal_files = ["environment_lock.json", "environment_lock.json.sha256", "cuda_smoke_results.json",
                  "hash_verification.json", "reproducibility_results.json", "resource_results.json",
                  "execution_readiness.json", "local_download_verification.json", "report.md",
                  "future_full_commands.md", "future_full_plan.json"]
    sums = "".join(digest(DEST / name) + "  " + name + "\n" for name in sorted(seal_files))
    (DEST / "SHA256SUMS").write_bytes(sums.encode())
    (DEST / "SHA256SUMS.sha256").write_bytes((digest(DEST / "SHA256SUMS") + "  SHA256SUMS\n").encode())
    print(json.dumps({"status": "READY_ENGINEERING_VALIDATION_ONLY", "environment_lock_sha256": EXPECTED_LOCK,
                      "methods_bitwise_exact": list(results), "full_training_executed": False,
                      "report": str(DEST / "report.md")}))


if __name__ == "__main__":
    main()

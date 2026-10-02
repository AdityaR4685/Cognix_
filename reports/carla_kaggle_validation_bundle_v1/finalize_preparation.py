"""Generate reviewable notebook/commands and audit preservation, never train."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    preparation = json.loads((HERE / "preparation_results.json").read_text())
    snapshot = json.loads((HERE / "preservation_snapshot.json").read_text())
    changed = [name for name, expected in snapshot["files"].items() if digest(ROOT / name) != expected]
    def protected_path(name):
        if name == "protocol.json":
            return ROOT / "reports/carla_gat_preregistration_v1/protocol.json"
        return Path(name) if Path(name).is_absolute() else ROOT / name
    changed += [name for group in ("upstream", "artifact") for name, expected in snapshot[group].items()
                if digest(protected_path(name)) != expected]
    if changed:
        raise RuntimeError("preexisting file changed: " + repr(changed))
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != snapshot["git_head"]:
        raise RuntimeError("Git HEAD changed")
    readiness = json.loads((HERE / "local_harness_check/execution_readiness.json").read_text())
    if readiness["status"] != "CPU_HARNESS_PASSED_KAGGLE_UNVERIFIED":
        raise RuntimeError("local harness check did not pass")
    notebook_code = '''# COGNIX VALIDATION ONLY. Never run future full commands in this notebook.
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile

# Change only this input path to the private compact bundle attached to this notebook.
BUNDLE_INPUT = Path('/kaggle/input/datasets/adityar4685/cognix-kaggle-validation-v1')
ROOT = Path('/kaggle/working/cognix_validation')
OUTPUT = Path('/kaggle/working/carla_kaggle_environment_lock_v1')
EXPECTED_ARCHIVE_HASH = 'ARCHIVE_HASH'
EXPECTED_BUNDLE_SEAL = 'BUNDLE_HASH'

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert BUNDLE_INPUT.is_dir(), 'Attach only the private compact validation input and set its exact path'
assert not ROOT.exists() and not OUTPUT.exists(), 'Use clean output paths; never overwrite evidence'
archive = BUNDLE_INPUT / 'cognix_kaggle_validation_v1.zip'
if archive.is_file():
    assert digest(archive) == EXPECTED_ARCHIVE_HASH, 'Archive SHA-256 mismatch'
    ROOT.mkdir()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            assert (ROOT / info.filename).resolve().is_relative_to(ROOT.resolve()), 'Unsafe archive member'
        z.extractall(ROOT)
else:
    # Kaggle may unpack a uploaded dataset archive. Verify its anchored seal first.
    assert (BUNDLE_INPUT / 'SHA256SUMS').is_file(), 'Expected compact bundle files missing'
    assert digest(BUNDLE_INPUT / 'SHA256SUMS') == EXPECTED_BUNDLE_SEAL, 'Input bundle seal mismatch'
    shutil.copytree(BUNDLE_INPUT, ROOT)
assert digest(ROOT / 'SHA256SUMS') == EXPECTED_BUNDLE_SEAL, 'Bundle seal mismatch'
for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
    expected, relative = line.split('  ', 1)
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT.resolve()) and digest(path) == expected, 'Bundle file mismatch: ' + relative
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # user authorized one visible T4 from T4 x2
command = [sys.executable, str(ROOT / 'reports/carla_kaggle_validation_bundle_v1/validate_kaggle.py'),
           '--output', str(OUTPUT)]
completed = subprocess.run(command, cwd=ROOT)
assert completed.returncode == 0, 'Validation failed; preserve failure evidence and stop without fallback'
readiness = json.loads((OUTPUT / 'execution_readiness.json').read_text())
assert readiness['ready'] is True and readiness['full_training_executed'] is False
print((OUTPUT / 'report.md').read_text())
shutil.make_archive('/kaggle/working/carla_kaggle_validation_evidence_v1', 'zip', OUTPUT)
print('Download carla_kaggle_validation_evidence_v1.zip and review. Do not run the full experiment.')
'''.replace("'ARCHIVE_HASH'", repr(preparation["archive_sha256"])).replace("'BUNDLE_HASH'", repr(preparation["bundle_SHA256SUMS_sha256"]))
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "markdown", "id": "scope", "metadata": {}, "source": [
            "# Frozen COGNIX Kaggle CUDA validation only\n",
            "User authorized GPU 0 isolation: two T4 allocated, one visible to PyTorch.\n",
            "No full graph training, TEST, conformal, scientific aggregation or source patching.\n"]},
            {"cell_type": "code", "id": "validation", "metadata": {}, "execution_count": None,
             "outputs": [], "source": notebook_code.splitlines(keepends=True)}]}
    write(HERE / "kaggle_validation.ipynb", notebook)
    launch = "/kaggle/working/cognix_validation/reports/carla_kaggle_validation_bundle_v1/launch_full.py"
    lock = "/kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json"
    run_root = "/kaggle/working/cognix_full_paired_graph_v1"
    plan = [{"method": method, "seed": seed, "frozen_batch_size": 256,
        "output": run_root + "/" + method + "/seed_" + str(seed)}
        for seed in (101, 202, 303, 404, 505) for method in ("nograph", "standard_gat", "epistemic_gat")]
    write(HERE / "future_full_plan.json", {"status": "DRAFT_REQUIRES_ACTUAL_VALIDATED_CUDA_LOCK",
        "executed": False, "runs": plan, "graph_scientific_sha256": preparation["frozen_hashes_verified"]["graph_scientific"],
        "separate_authorization_required": True, "retain_all_runs": True, "replace_failed_seeds": False,
        "best_seed_selection": False, "scientific_aggregation": False})
    lines = ["# Future full experiment commands — prepared, never executed", "",
        "Status: DRAFT. Actual Kaggle single-T4 validation and a sealed execution lock are still required.", "",
        "This preflight verifies the frozen bundle, graph, lock seal, evidence and actual environment, then prints all 15 planned runs:", "",
        "```bash", "CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python " + launch + " --environment-lock " + lock + " --output " + run_root, "```", "",
        "Only in a separately authorized full experiment milestone, the exact all-run command is:", "",
        "```bash", "CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python " + launch + " --environment-lock " + lock + " --output " + run_root + " --execute-full-experiment", "```", "",
        "That command retains nograph, standard_gat and epistemic_gat for every seed 101, 202, 303, 404, 505.",
        "The 15 equivalent independent invocations below are alternatives to the all-run command; do not run both.",
        "Each creates its own run root and nested method/seed directory. Existing directories are refused.", "", "```bash"]
    for row in plan:
        independent = run_root + "_individual/" + row["method"] + "_" + str(row["seed"])
        lines.append("CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python " + launch + " --environment-lock " + lock + " --output " + independent +
                     " --method " + row["method"] + " --seed " + str(row["seed"]) + " --execute-full-experiment")
    lines += ["```", "", "Frozen orchestration preserves checkpoints, history, row-keyed predictions, metrics and failures.",
              "It does not replace a failed seed or choose a best seed. No scientific aggregation is invoked."]
    (HERE / "future_full_commands.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(HERE / "core_integrity_results.json", {"passed": True, "preserved_files": len(snapshot["files"]),
        "preserved_upstream_files": len(snapshot["upstream"]), "changed_preexisting_files": [],
        "git_HEAD_unchanged": True, "graph_artifact_unchanged": True, "protocol_unchanged": True,
        "trainer_bundle_unchanged": True, "N20_unchanged": True, "TEST_access": False,
        "full_graph_training": False, "conformal_fit": False, "scientific_tuning": False,
        "scientific_aggregation": False, "commit_push": False})
    write(HERE / "runtime_access.json", {"notebook_url": "https://www.kaggle.com/code/adityar4685/notebook694cee9113",
        "observed_saved_version": "1 of 1", "observed_saved_runtime_label": "19s · GPU T4 x2",
        "connected_browser_authenticated": True, "actual_runtime_versions": None,
        "live_accelerator_options": ["None", "GPU T4 x2", "TPU v5e-8"],
        "live_selected_accelerator": "GPU T4 x2",
        "observation_scope": "authenticated notebook accelerator UI; not CUDA kernel inspection",
        "CUDA_execution": False, "blockers": ["no physical single-T4 accelerator option",
            "actual CUDA validation pending"],
        "user_authorized_GPU_exposure": "CUDA_VISIBLE_DEVICES=0 before PyTorch import",
        "user_authorized_private_archive_upload": True})
    report = '''## Kaggle Runtime

The linked notebook's saved Version 1 shows GPU T4 x2 (19 seconds). After user sign-in, the live editor offered None, GPU T4 x2 and TPU v5e-8, with GPU T4 x2 selected. The user explicitly authorized CUDA_VISIBLE_DEVICES=0 before PyTorch import and private archive upload. Actual Python/PyTorch/NumPy/CUDA/cuDNN versions and CUDA properties remain unobserved. CUDA validation is pending.

## Frozen Hash Verification

Graph scientific SHA-256: baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237.
Protocol SHA-256: 68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203.
Trainer bundle SHA-256: 9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b.
All matched. NPZ/schema/manifest, preregistration seals, generic graph sources, upstream N20 dependencies, node/feature order, shape [89970,3,3], 71976/17994 rows and 12/3 scenarios verified read-only. See preparation_results.json and local_harness_check/hash_verification.json.

## CUDA Determinism Settings

The external harness enforces strict deterministic algorithms with warn_only=False, deterministic cuDNN, benchmark=False, TF32=False, AMP=False and CUBLAS_WORKSPACE_CONFIG=:4096:8. Effective CPU harness settings checked; effective CUDA settings remain pending. Unsupported operations fail without fallback.

## Forward Validation

CPU harness checks passed for finite output/BCE, shape, finite incoming-normalized attention and zero self edges. E=0 matched-RNG outputs were exact in inference/training mode. Fixed-logit sender suppression passed at both layers. Existing generic-forward tolerances atol=1e-7/rtol=1e-6 and attention atol=1e-6/default rtol=1e-5 were retained. CPU/GPU differences will be measured, without introducing a new cross-device tolerance. T4 execution remains pending.

## CUDA Smoke Training

CUDA smoke not executed. The harness was tested on CPU using only the unchanged 128-row synthetic fixture (32 train, 96 validation) and frozen trainer settings. All methods had finite gradients, parameter updates, finite trajectories, working validation, checkpoint reloads and row-keyed prediction export. Six fresh CPU processes covered first/repeat runs. No scientific interpretation of fixture metrics.

## GPU Reproducibility

Unverified on GPU. CPU engineering repeats were bitwise exact for all three methods at seed 101: initial parameters/RNG, epoch row orders, batch boundaries, trajectories, selected epoch, final parameters/optimizer, all saved checkpoint contents, row-keyed predictions and metrics. The GPU gate preserves the existing exact-repeat requirement; any difference fails and requires diagnosis. See local_harness_check/reproducibility_results.json.

## Paired Standard/Epistemic Integrity

CPU observer confirmed explicit cloned parameters, byte equality, independent storage, identical dropout RNG, matching Adam configuration, row order, batches, validation trajectories, selected epoch, checkpoint model tensors and probabilities at E=0. The isolated nonzero-E fixed-logit diagnostic used the same model/input and changed only sender E. CUDA pairing remains pending.

## GPU Memory / Runtime

No GPU memory/runtime claim is made. CPU resource harness exercised one synthetic batch of 256 for each method. Actual T4 allocated/reserved/peak memory and synchronized batch runtime will be measured. No real dataset epochs, batch-size tuning or OOM fallback.

## Environment Lock

No validated Kaggle environment_lock.json was issued or sealed. Successful actual Kaggle execution will create reports/carla_kaggle_environment_lock_v1/ (notebook output path /kaggle/working/carla_kaggle_environment_lock_v1), bind CUDA evidence to the bundle seal and hash the lock. CPU harness outputs explicitly have ready=False.

## Full-Run Commands Prepared

future_full_commands.md and future_full_plan.json contain draft commands/configuration for all 15 method/seed combinations. They require an actual sealed CUDA lock and separate full-milestone authorization. The future launcher verifies hashes, evidence and environment first; retains independent runs/checkpoints/history/predictions/metrics/failures; never replaces a seed or selects a best seed. Commands were not executed.

## Core / Artifact Integrity

All preexisting files and upstream dependencies checked against the preservation snapshot remain unchanged. Graph, protocol, trainer bundle and N=20 unchanged. No TEST access, full graph training, conformal fit, scientific tuning/aggregation, commit or push. Prior validated regression baseline remains 608 passed; not represented as a new full regression run. Six new stdlib safeguard tests passed. See core_integrity_results.json.

## Ready / Not Ready for Full Paired-Seed Experiment

NOT READY. Actual single-T4 environment, CUDA forward/training/repeat/pairing and resource gates are unverified. The provided saved notebook uses T4 x2, which does not satisfy the frozen single-T4 requirement.

## Recommended Next Step

Use the approved private compact bundle and GPU-0 isolation with kaggle_validation.ipynb, run validation only and preserve/download its evidence. Review the actual sealed lock before separately authorizing the full experiment.
'''
    (HERE / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"core_integrity": "passed", "preserved_files": len(snapshot["files"]),
                      "commands_prepared": 15, "actual_Kaggle_CUDA_verified": False}))


if __name__ == "__main__":
    main()

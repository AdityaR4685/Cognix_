"""One-time OFFLINE repair builder. Never executes the scientific runner or fits models."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / "reports/carla_final_conformal_cal_bundle_v1"
FAILURE = ROOT / "reports/carla_final_conformal_cal_attempt001_failure_v1"
COMMIT = "1688bd79375e09d6a1fc78c4615b7b2745af489a"
V1_MANIFEST = "bbcdad0829bc68a6a69007da82073e682373a548e0422d92000fd0f80592acbe"
READY = "READY_FOR_SEPARATELY_AUTHORIZED_FINAL_CONFORMAL_CAL_ATTEMPT_002"
OUTPUT = "/kaggle/working/cognix_final_conformal_cal_attempt002_v1"
ZIP_NAME = "cognix_final_conformal_cal_kaggle_v2.zip"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
def text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
def git(*args): return subprocess.check_output(["git", *args], cwd=ROOT)
def inventory(directory): return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}
def seal(directory, name):
    files = [p for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in {name, name + ".sha256", ZIP_NAME, ZIP_NAME + ".sha256"}]
    text(directory / name, "".join(sha(p) + "  " + p.relative_to(directory).as_posix() + "\n" for p in files))
    text(directory / (name + ".sha256"), sha(directory / name) + "  " + name + "\n")
def offline(target, *args):
    result = subprocess.run([sys.executable, "-B", str(HERE / "offline_checks.py"), str(target), *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert rows[-1]["offline_network_guard"] == "PASSED"
    return {"command": [sys.executable, "-B", str(HERE / "offline_checks.py"), str(target), *args], "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr, "result": rows[-2], "network_guard": rows[-1]}

def main():
    resume = sys.argv[1:] == ["--finish-offline-build"]
    assert sys.argv[1:] in ([], ["--finish-offline-build"])
    assert not (HERE / "BUNDLE_SHA256SUMS").exists(), "Sealed v2 bundle is immutable"
    if resume:
        assert FAILURE.exists() and inventory(FAILURE) == inventory(HERE / "evidence/attempt001")
        for line in (FAILURE / "FAILURE_SHA256SUMS").read_text().splitlines():
            digest, rel = line.split("  ", 1)
            assert sha(FAILURE / rel) == digest
        assert (FAILURE / "FAILURE_SHA256SUMS.sha256").read_text().split() == [sha(FAILURE / "FAILURE_SHA256SUMS"), "FAILURE_SHA256SUMS"]
    else:
        assert not FAILURE.exists(), "Historical failure report already exists; never overwrite"
        assert set(inventory(HERE)) == {"offline_checks.py", "regression_verification.py", "prepare_bundle_v2.py"}, "Build once in a fresh v2 directory"
    initial_status = git("status", "--porcelain=v1").decode().splitlines()
    tracked_before = git("status", "--porcelain=v1", "--untracked-files=no")
    assert not tracked_before
    assert git("rev-parse", "HEAD").decode().strip() == COMMIT
    before = inventory(V1)
    assert sha(V1 / "BUNDLE_SHA256SUMS") == V1_MANIFEST
    # Prove every working-tree v1 byte matches the sealed commit, including historical ZIP/seals.
    for rel, digest in before.items():
        committed = git("show", COMMIT + ":reports/carla_final_conformal_cal_bundle_v1/" + rel)
        assert hashlib.sha256(committed).hexdigest() == digest, rel
    v1_verified = offline(V1 / "verify_bundle.py")
    v1_synthetic = offline(V1 / "synthetic_verification.py")
    evidence = {
        "failure_exception_audit.json": {"cal_completed": 0, "exception": "ValueError", "message": "FROZEN_IMAGE_SCHEMA_REQUIRES_UINT8_RGB", "status": "INCOMPLETE_NO_RETRY"},
        "access_ledger.json": {"automatic_retry": False, "cal_only_science": True, "cal_scenarios_completed": 0, "compressed_bytes_received": 840761354,
            "compressed_sha256_partial_or_complete": "2cd357e97c8a19ac57c28b94afe9082e780c5fce43dfbeed65c50ea5313d4f0f", "eval_scenarios_decoded": 0, "http_requests": 1,
            "opaque_body_bytes_discarded_by_role": {"CAL": 131589081, "EVAL_OPAQUE_DISCARD": 526258000, "EXCLUDED_OPAQUE_DISCARD": 240747299},
            "range_requests": 0, "raw_archive_retained": False, "scenario_roots_seen": 7, "transport_is_not_scientific_decoding": True},
        "result_manifest.json": {"must_not_use_partial_scores_as_final_thresholds": True, "status": "INCOMPLETE"}}
    if not resume:
        FAILURE.mkdir(exist_ok=False)
        for name, value in evidence.items(): write(FAILURE / name, value)
        shutil.copyfile(Path("C:/Users/Aditya/.codex/attachments/26c8ff3f-2f83-47a1-bcab-858158ccc46c/Pasted text.txt"), FAILURE / "user_supplied_request.txt")
        write(FAILURE / "historical_traceback_availability.json", {"historical_traceback": None, "status": "NOT_SUPPLIED", "reason": "User supplied exception audit lacks a traceback. Do not invent or substitute synthetic traceback for historical evidence."})
        write(FAILURE / "failure_record.json", {"attempt": 1, "status": "INCOMPLETE_NO_RETRY", "source": "User-supplied evidence; no original output directory or traceback supplied",
            "v1_commit": COMMIT, "v1_manifest_sha256": V1_MANIFEST, "pre_access_gates_passed_before_attempt": True,
            "pre_access_gate_evidence": "User attestation plus sealed v1 pre_access_validation.json; execution-time gate transcript not supplied",
            "http_GET_count": 1, "compressed_bytes_received": 840761354, "scenario_roots_seen": 7, "CAL_scenarios_completed": 0, "EVAL_scenarios_decoded": 0,
            "thresholds_finalized": 0, "partial_results_forbidden": True, "automatic_retry": False, "v1_immutable": True,
            "observed_exception": evidence["failure_exception_audit.json"], "failed_scenario_ID": None, "failed_image_properties": None,
            "root_cause": "Shared runner RGB gate is inconsistent with already-frozen segmentation contract. A legitimate 2D uint8 segmentation PNG is rejected before the unchanged histogram extractor. The actual failed image shape/mode/modality was not recorded in supplied evidence.",
            "ledger_limitation": "streaming.py increments body_bytes[role] for every non-directory body before decode/discard and before full body consumption. CAL=131589081 is accounted member-body size, not evidence of entirely opaque/discarded CAL bytes. EVAL/exclusion roles are opaque-discard only; on a failure counters need not prove complete consumption of the final counted body. Preserve historical ledger unchanged.",
            "scientific_failure": "No completed CAL scenario, no finalized thresholds. Compressed partial-or-complete digest is not a verified full archive receipt.",
            "repair_selection_basis": "Frozen source and TRAIN-compatible synthetic fixtures only; no CAL labels/outcomes, EVAL outcomes, anomaly performance or TEST payload"})
        text(FAILURE / "report.md", """# Historical Attempt 001 failure record

    Attempt 001 occurred after the pre-access gates passed (user attestation; sealed v1 validation corroborates preparation, but no execution-time gate transcript was supplied). Exactly one HTTP GET received 840761354 compressed bytes; seven scenario roots were seen. Zero CAL scenarios completed and zero EVAL scenarios were decoded. No thresholds were finalized. Partial scores/results are forbidden as final thresholds. No automatic retry occurred or is permitted by this record.

    The exception was ValueError: FROZEN_IMAGE_SCHEMA_REQUIRES_UINT8_RGB. The supplied audit contains no traceback, failed scenario ID, image mode, shape, dtype or failed modality. These facts are unavailable and are not inferred from synthetic fixtures. The separately labelled synthetic traceback in v2 is a reproduction, not the historical traceback.

    Frozen runner source used a shared RGB-only gate for camera and segmentation. Frozen segmentation_histogram_features accepts a 2-D class map and selects channel zero for 3-D arrays. Frozen TRAIN/cache source passes np.asarray(PIL.Image.open(path)) directly to this extractor. The old synthetic segmentation PNG was (16,16,3), so it did not expose the incompatibility. The contract mismatch is confirmed offline; the specific failed image representation remains unrecorded.

    Ledger nuance: opaque_body_bytes_discarded_by_role.CAL=131589081 must not be described as entirely opaque/discarded. streaming.py accounts all non-directory body sizes before decode/discard; these sizes can precede completed reads. EVAL=526258000 and exclusions=240747299 are restricted to opaque discard, with zero EVAL decoder entry. Preserve all supplied ledger values unchanged. The recorded compressed digest is partial-or-complete and does not establish full archive integrity.

    v1 commit 1688bd7 (1688bd79375e09d6a1fc78c4615b7b2745af489a), manifest bbcdad0829bc68a6a69007da82073e682373a548e0422d92000fd0f80592acbe, and all historical seals remain unchanged. This new directory is a sealed historical record; do not edit or overwrite it. Separate future records are required for additional evidence. No protocol history, including Amendment 003, is rewritten. This preparation performed zero TEST requests and accessed zero TEST payload bytes.
    """)
        seal(FAILURE, "FAILURE_SHA256SUMS")
    skipped = {"BUNDLE_SHA256SUMS", "BUNDLE_SHA256SUMS.sha256", "cognix_final_conformal_cal_kaggle_v1.zip", "cognix_final_conformal_cal_kaggle_v1.zip.sha256", "prepare_bundle.py", "README.md", "cognix_final_conformal_cal_v1.ipynb", "pre_access_validation.json", "synthetic_verification_results.json"}
    for rel in before:
        if rel in skipped: continue
        dest = HERE / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(V1 / rel, dest)
    for name in ("runner.py", "prepare_bundle.py", "pre_access_validation.json", "synthetic_verification_results.json", "scientific_bindings.json", "protected_eval_policy.json", "BUNDLE_SHA256SUMS", "BUNDLE_SHA256SUMS.sha256"):
        dest = HERE / "evidence" / ("v1_" + name + (".txt" if name.endswith(".py") else ""))
        shutil.copyfile(V1 / name, dest)
    shutil.copytree(FAILURE, HERE / "evidence/attempt001", dirs_exist_ok=resume)
    old = (V1 / "runner.py").read_text(encoding="utf-8")
    old_gate = '''            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("FROZEN_IMAGE_SCHEMA_REQUIRES_UINT8_RGB")
            feature = camera_embedding_features(array) if kind == "camera" else segmentation_histogram_features(array)'''
    new_gate = '''            if kind == "camera":
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError("FROZEN_CAMERA_SCHEMA_REQUIRES_UINT8_RGB")
                feature = camera_embedding_features(array)
            else:
                if array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] < 1):
                    raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP")
                feature = segmentation_histogram_features(array)'''
    assert old.count(old_gate) == 1
    new = old.replace(old_gate, new_gate).replace('/kaggle/working/cognix_final_conformal_cal_v1', OUTPUT)
    text(HERE / "runner.py", new)
    # Exact inverse verifies that every other runner byte/operation is preserved.
    assert new.replace(new_gate, old_gate).replace(OUTPUT, '/kaggle/working/cognix_final_conformal_cal_v1') == old
    verifier = (V1 / "verify_bundle.py").read_text().replace("cognix_final_conformal_cal_kaggle_v1.zip", ZIP_NAME).replace("READY_FOR_FINAL_CONFORMAL_CAL_EXECUTION", READY)
    text(HERE / "verify_bundle.py", verifier)
    policy = json.loads((V1 / "protected_eval_policy.json").read_text()); policy["output"] = OUTPUT
    write(HERE / "protected_eval_policy.json", policy)
    scientific = json.loads((V1 / "scientific_bindings.json").read_text())
    scientific["bundle_scientific_file_sha256"]["runner.py"] = sha(HERE / "runner.py")
    write(HERE / "scientific_bindings.json", scientific)
    from PIL import Image
    import numpy as np
    (HERE / "fixtures").mkdir(exist_ok=resume)
    Image.fromarray((np.arange(256).reshape(16,16) % 29).astype(np.uint8)).save(HERE / "fixtures/segmentation_2d_uint8.png", format="PNG")
    synthetic = offline(HERE / "synthetic_verification.py")
    regression = offline(HERE / "regression_verification.py")
    write(HERE / "synthetic_verification_results.json", synthetic["result"])
    write(HERE / "regression_verification_results.json", regression["result"])
    write(HERE / "offline_test_transcripts.json", {"v1_integrity": v1_verified, "v1_existing_synthetic": v1_synthetic, "v2_existing_synthetic": synthetic, "v2_regression": regression})
    unchanged = {rel: digest for rel, digest in before.items() if (HERE / rel).is_file() and sha(HERE / rel) == digest}
    changed = {rel: {"v1_sha256": before[rel], "v2_sha256": sha(HERE / rel)} for rel in before if (HERE / rel).is_file() and sha(HERE / rel) != before[rel]}
    assert set(changed) == {"runner.py", "verify_bundle.py", "protected_eval_policy.json", "scientific_bindings.json", "synthetic_verification_results.json"}
    assert json.loads((V1 / "synthetic_verification_results.json").read_text()) == synthetic["result"]
    write(HERE / "unchanged_scientific_bindings.json", {"v1_commit": COMMIT, "v1_manifest_sha256": V1_MANIFEST,
        "byte_identical_v1_files": unchanged, "changed_v1_files_before_new_validation": changed,
        "exact_runner_inverse_patch_verified": True, "frozen_CAL_count": 125, "frozen_EVAL_count": 500, "frozen_scorer_states": 15,
        "unchanged": ["125 CAL members", "500 EVAL members", "historical exclusions", "15 scorer identities, states and hashes", "one-class fitted parameters", "feature extractor bytes", "normality agents", "graph models", "checkpoints", "label semantics", "alpha=.05", "S_j=max_t(1-P_t(Y_t))", "augmented conformal order statistic k=120", "inclusive <=", "single byte-zero stream", "EVAL opaque discard", "one attempt, no automatic retry", "runtime lock", "Amendment 003 and prior protocol history"]})
    amendment = {"record_type": "POST_FAILURE_IMPLEMENTATION_COMPATIBILITY_REPAIR", "v1_commit": COMMIT, "v1_manifest_sha256": V1_MANIFEST,
        "attempt001_evidence_sha256": {name: sha(FAILURE / name) for name in evidence}, "attempt001_failure_manifest_sha256": sha(FAILURE / "FAILURE_SHA256SUMS"),
        "attempt001_evidence": evidence, "repair": "Separate camera/segmentation validation; accept uint8 2D map or 3D with channel zero available; preserve direct np.asarray semantics and frozen extractor; isolated Attempt 002 output",
        "selection_basis": "Frozen source/TRAIN-compatible synthetic fixtures; no TEST payload/CAL outcomes/CAL labels/anomaly performance/EVAL outcomes",
        "scientific_permissibility": "A separately authorized Attempt 002 is permissible as an implementation compatibility repair of an existing frozen contract. No method/data/model tuning; no partial results, no finalized thresholds or EVAL scientific decoding from Attempt 001. Prior physical TEST transport remains acknowledged; Attempt 002 is not relabelled as first access.",
        "separate_attempt002_authorization_required": True, "attempt002_authorized_by_this_preparation": False, "attempt002_executed": False,
        "output": OUTPUT, "prior_history_modified": False, "Amendment_003_modified": False, "new_protocol_revision": False}
    write(HERE / "post_failure_repair_record.json", amendment)
    text(HERE / "post_failure_repair_record.md", "# Post-failure implementation compatibility repair\n\n" + amendment["scientific_permissibility"] + "\n\nBound to v1 commit 1688bd7, manifest " + V1_MANIFEST + ". Exact supplied Attempt 001 evidence and its independent failure seal are bound in the JSON record. The only inference-path changes are modality validation and the isolated output path. Frozen scientific files and membership/score/quantile semantics are compared byte-for-byte. Amendment 003 and all prior records remain unchanged. This record authorizes preparation only; human authorization of Attempt 002 must be separate. No scientific execution has occurred during preparation.\n")
    notebook = json.loads((V1 / "cognix_final_conformal_cal_v1.ipynb").read_text())
    for cell in notebook["cells"]:
        cell["source"] = [s.replace("Frozen COGNIX FINAL_CONFORMAL_CAL", "Frozen COGNIX FINAL_CONFORMAL_CAL Attempt 002 (v2)").replace("/kaggle/working/cognix_cal_bundle", "/kaggle/working/cognix_cal_bundle_v2") for s in cell["source"]]
    notebook["cells"][0]["source"].append("Attempt 001 failed. This v2 notebook is unexecuted; Attempt 002 requires separate authorization. Output: " + OUTPUT + ".\n")
    write(HERE / "cognix_final_conformal_cal_v2.ipynb", notebook)
    readme = (V1 / "README.md").read_text().replace("execution bundle v1", "execution bundle v2").replace("READY_FOR_FINAL_CONFORMAL_CAL_EXECUTION", READY).replace("/kaggle/working/cognix_final_conformal_cal_v1", OUTPUT).replace("/kaggle/working/cognix_cal_bundle", "/kaggle/working/cognix_cal_bundle_v2")
    readme = readme.replace("Preparation helper prepare_bundle.py is local-only and requires repository evidence; it is never invoked by the notebook or runner.", "prepare_bundle_v2.py is a one-time offline builder and refuses an existing built bundle/failure report. Historical v1 builder is retained as .txt evidence. Neither is invoked by the notebook or runner.")
    readme += "\nAttempt 001 failed; see evidence/attempt001 and post_failure_repair_record.json. No original traceback or failed-image properties were supplied. Synthetic traceback is explicitly labelled. CAL body accounting in the old ledger does not mean opaque discard. v1 and prior history remain immutable.\n\nRepair: camera retains uint8 HxWx3; segmentation accepts uint8 2D or 3D with at least one channel, using the frozen extractor directly. Other channels are ignored by the frozen extractor, with its class-range check intact. No image conversion or feature change. regression_verification.py includes 2D-PNG end-to-end scoring of all 15 models and decoder spies. offline_checks.py enforces process-local network refusal for verification; it refuses scientific execution switches.\n\nReadiness is offline preparation only, not successful scientific execution or authorization. Local verification versions differ from locked Kaggle versions; execution-time runtime_check remains mandatory and unchanged. Verify the external manifest and ZIP hashes before upload/extraction. New output path cannot overwrite Attempt 001.\n"
    text(HERE / "README.md", readme)
    import platform, torch, PIL, pyarrow, pandas
    write(HERE / "pre_access_validation.json", {**synthetic["result"], "regression_groups_passed": regression["result"]["regression_groups_passed"], "regression_groups_failed": 0,
        "readiness": READY, "scientific_execution_success": False, "attempt002_executed": False, "attempt002_authorized": False,
        "v1_commit": COMMIT, "v1_manifest_sha256": V1_MANIFEST, "v1_integrity_verified": True, "v1_existing_synthetic_groups_passed": v1_synthetic["result"]["synthetic_verification_groups_passed"],
        "CAL_count": 125, "EVAL_count": 500, "overlap": 0, "union": 625, "frozen_scorer_states": 15, "scientific_bindings_unchanged_except_execution_schema_repair": True,
        "all_frozen_states_strictly_loadable": True, "no_fitting_training_tuning": True, "runtime_lock_unchanged": True,
        "local_offline_runtime": {"Python": platform.python_version(), "NumPy": np.__version__, "PyTorch": torch.__version__, "Pillow": PIL.__version__, "PyArrow": pyarrow.__version__, "pandas": pandas.__version__},
        "runtime_limitation": "Offline CPU tests do not certify locked Kaggle GPU numerics. Unchanged runtime_check must pass before separately authorized Attempt 002. No runtime pin changed.",
        "NO_TEST_PAYLOAD_ACCESSED": True, "TEST_network_requests": 0, "TEST_payload_bytes_accessed": 0, "archive_HTTP_GET_HEAD_range_requests": 0,
        "network_guard": "Process-local Python audit hook in each offline verifier/test subprocess; no OS-wide trace claim",
        "Kaggle_notebook_executed": False, "commit": False, "push": False, "tracked_worktree_clean": True, "outside_new_directories_git_status_unchanged": True,
        "v1_all_bytes_unchanged": True, "historical_failure_directory_sealed": True})
    assert inventory(V1) == before
    assert git("status", "--porcelain=v1", "--untracked-files=no") == tracked_before
    def outside(lines): return [line for line in lines if "reports/carla_final_conformal_cal_bundle_v2/" not in line and "reports/carla_final_conformal_cal_attempt001_failure_v1/" not in line]
    assert outside(git("status", "--porcelain=v1").decode().splitlines()) == outside(initial_status)
    seal(HERE, "BUNDLE_SHA256SUMS")
    verified = offline(HERE / "verify_bundle.py")
    preflight = offline(HERE / "runner.py", "--preflight")
    with zipfile.ZipFile(HERE / ZIP_NAME, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(HERE.rglob("*")):
            if not p.is_file() or p.name == ZIP_NAME: continue
            info = zipfile.ZipInfo(p.relative_to(HERE).as_posix(), (2026,10,3,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
    text(HERE / (ZIP_NAME + ".sha256"), sha(HERE / ZIP_NAME) + "  " + ZIP_NAME + "\n")
    with zipfile.ZipFile(HERE / ZIP_NAME) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(inventory(HERE)) - {ZIP_NAME, ZIP_NAME + ".sha256"}
        for name in archive.namelist(): assert hashlib.sha256(archive.read(name)).hexdigest() == sha(HERE / name)
    final_verified = offline(HERE / "verify_bundle.py")
    assert inventory(V1) == before
    assert inventory(FAILURE) == inventory(HERE / "evidence/attempt001")
    print(json.dumps({"readiness": READY, "v2_manifest_SHA256": sha(HERE / "BUNDLE_SHA256SUMS"), "v2_ZIP_SHA256": sha(HERE / ZIP_NAME), "ZIP_bytes": (HERE / ZIP_NAME).stat().st_size,
        "v2_integrity": final_verified["result"], "v2_preflight": preflight["result"], "TEST_network_requests": 0, "TEST_payload_bytes_accessed": 0}))

if __name__ == "__main__": main()

"""Preparation only: frozen metadata/model reads and synthetic checks. No TEST."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTO = ROOT / "reports/carla_final_evaluation_preregistration_v1"
PARTITION = ROOT / "reports/carla_final_partition_v1"
ZIP_NAME = "cognix_final_conformal_cal_kaggle_v1.zip"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(rel, obj):
    path = HERE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    blob = git("show", "HEAD:" + rel)
    assert blob.decode("utf-8-sig").replace("\r\n", "\n") == path.read_text(encoding="utf-8-sig"), rel
    return git("rev-parse", "HEAD:" + rel).decode().strip()


def copy(path, dest):
    dest = HERE / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    return dest


def verify_manifest(directory, expected):
    assert sha(directory / "SHA256SUMS") == expected
    checked = []
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = (directory / rel).resolve()
        assert path.is_relative_to(directory.resolve()) and sha(path) == digest, rel
        checked.append(rel)
    return checked


def extract_nodes(path, names):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parts = []
    found = set()
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            parts.append(ast.get_source_segment(source, node))
            found.add(name)
    assert found == set(names), (path, names, found)
    return "\n\n".join(parts) + "\n"


def main():
    assert not (HERE / "BUNDLE_SHA256SUMS").exists(), "Sealed bundle exists: no silent resealing"
    initial_head = git("rev-parse", "HEAD").decode().strip()
    initial_branch = git("branch", "--show-current").decode().strip()
    assert not git("status", "--porcelain=v1", "--untracked-files=no"), "Tracked worktree dirty"
    initial_status = [line for line in git("status", "--porcelain=v1").decode().splitlines()
                      if "reports/carla_final_conformal_cal_bundle_v1/" not in line]
    partition_files = verify_manifest(PARTITION, "e3047f2032896cc062f4b90eab9f7488151b9680e1b3c1234a81c112dcce577c")
    protocol_seal = (PROTO / "SHA256SUMS.sha256").read_text().split()[0]
    verify_manifest(PROTO, protocol_seal)
    evidence = {}
    for name in ("conformal_protocol.json", "frozen_method_bindings.json", "pipeline_map.json", "report.md",
                 "amendment_003_changelog.md", "amendment_003_record.json", "source_evidence.json"):
        path = PROTO / name
        evidence[name] = {"working_sha256": sha(path), "committed_blob": committed(path)}
        copy(path, "evidence/protocol/" + name)
    for name in partition_files + ["SHA256SUMS", "SHA256SUMS.sha256"]:
        committed(PARTITION / name)
        copy(PARTITION / name, "evidence/partition/" + name + (".txt" if name.endswith(".py") else ""))
    for name in ("final_conformal_cal.txt", "final_evaluation.txt"):
        copy(PARTITION / name, name)
    inventory_zip = ROOT / "reports/carla_test_structure_inventory_v1/kaggle_result_v1/cognix_carla_test_inventory_result_v1.zip"
    assert sha(inventory_zip) == "6f1bfa9bfb6a2401263c4b0642243f779632cf79c7f0b6cd7f07f95aaa6999b3"
    # This ZIP contains already-authorized structure-only metadata, no TEST payload.
    with zipfile.ZipFile(inventory_zip) as z:
        for name in ("SOURCE_INVENTORY_SHA256SUMS", "ELIGIBLE_INVENTORY_SHA256SUMS", "eligible_inventory_sorted.txt",
                     "eligible_inventory.json", "historical_exclusions.json", "scenario_inventory.json", "scenario_inventory_sorted.txt"):
            data = z.read("cognix_carla_test_inventory_result_v1/" + name)
            path = HERE / "evidence/inventory" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    cal = (HERE / "final_conformal_cal.txt").read_text().splitlines()
    evaluation = (HERE / "final_evaluation.txt").read_text().splitlines()
    eligible = (HERE / "evidence/inventory/eligible_inventory_sorted.txt").read_text().splitlines()
    assert len(cal) == len(set(cal)) == 125 and len(evaluation) == len(set(evaluation)) == 500
    assert not set(cal) & set(evaluation) and set(cal) | set(evaluation) == set(eligible)
    protocol = read(PROTO / "conformal_protocol.json")
    assert protocol["alpha"] == .05 and protocol["global_cutoffs"] == 15
    assert protocol["numerics"] == "Exact integer rank k=(19*(n_cal+1)+19)//20; +infinity augmentation and inclusive comparison. Serialize +infinity as null with quantile_is_infinite=true; no nonstandard JSON Infinity."
    assert protocol["class_order"] == ["official normal observation (0)", "official anomaly observation (1)"]
    source_map = read(PROTO / "source_evidence.json")["local_files"]
    graph_protocol = read(ROOT / "reports/carla_gat_preregistration_v1/protocol.json")
    bindings = graph_protocol["bindings"]
    frozen_sources = ["cognix/core/types.py", "cognix/core/interfaces.py", "cognix/graph/epistemic_gat.py",
                      "cognix/adapters/carla/graph_training_models.py", "cognix/adapters/carla/real_features.py",
                      "cognix/adapters/carla/normality.py", "cognix/adapters/carla/real_agents.py"]
    sources = {}
    for rel in frozen_sources:
        path = ROOT / rel
        expected = source_map.get(rel) or bindings["generic_graph_source_sha256"].get(rel)
        if expected:
            assert sha(path) == expected, rel
        sources[rel] = {"sha256": sha(path), "committed_blob": committed(path)}
        copy(path, "frozen_sources/" + rel)
    # Bind the exact existing restore function; exclude raw FIT exporter/cache/trainer code.
    ge_source = ROOT / "cognix/adapters/carla/graph_fit_export.py"
    ge_subset = ("import numpy as np\nfrom cognix.adapters.carla.normality import BootstrapNormalityEnsemble, EnsemblePredictiveCalibrator, MahalanobisNormality\n"
                 "from cognix.adapters.carla.real_agents import RealCameraAgent, RealIMUAgent, RealSegAgent\n\n" +
                 extract_nodes(ge_source, ["NODE_ORDER", "ADJACENCY", "OFFSETS", "GraphExportError", "require", "restore_agents"]))
    (HERE / "frozen_sources/cognix/adapters/carla/graph_fit_export.py").write_text(ge_subset, encoding="utf-8")
    sources[ge_source.relative_to(ROOT).as_posix()] = {"sha256": sha(ge_source), "committed_blob": committed(ge_source),
        "reused_AST_nodes": ["NODE_ORDER", "ADJACENCY", "OFFSETS", "GraphExportError", "require", "restore_agents"]}
    parser = ROOT / "reports/carla_test_structure_inventory_v1/extract_structure_inventory.py"
    amended = ROOT / "reports/carla_test_structure_inventory_v1/header_access_amendment_001/header_parser.py"
    helper = "import re\n\n" + extract_nodes(parser, ["TOWNS", "TYPES", "Stop", "require", "text_field", "octal", "classify"]) + "\n" + extract_nodes(amended, ["header_fields"])
    (HERE / "header_helpers.py").write_text(helper, encoding="utf-8")
    sources[parser.relative_to(ROOT).as_posix()] = {"sha256": sha(parser), "committed_blob": committed(parser), "reused": "strict path classification and TAR numeric/string guards"}
    sources[amended.relative_to(ROOT).as_posix()] = {"sha256": sha(amended), "committed_blob": committed(amended), "reused": "exact amended header_fields"}
    for rel in ("cognix/adapters/carla/cache_builder.py", "cognix/adapters/carla/carlanomaly_loader.py"):
        expected = bindings["source_code_sha256"][rel]
        assert sha(ROOT / rel) == expected
        copy(ROOT / rel, "evidence/sources/" + Path(rel).name + ".txt")
    label_source = ROOT / "reports/carla_public_provenance_audit_v1/sources/devkit/carlanomaly/datasets/anomaly_timestep.py"
    copy(label_source, "evidence/sources/official_anomaly_timestep.py.txt")
    # Frozen upstream states, not raw TRAIN, compact row arrays, or pseudo data.
    for filename in ("fitted_oneclass_parameters.npz", "manifest.json"):
        path = next(Path(s) for s in bindings["all_cache_handoff_file_sha256"] if s.endswith("/" + filename) or s.endswith("\\" + filename))
        assert sha(path) == bindings["all_cache_handoff_file_sha256"][str(path)]
        copy(path, "upstream/" + filename)
    from frozen_runtime import bootstrap, recursive_content_hash
    bootstrap()
    import numpy as np
    import torch
    from cognix.adapters.carla.graph_training_models import SharedNodeMLP, BatchedGAT, model_configuration
    rows = []
    for original in read(PROTO / "frozen_method_bindings.json")["runs"]:
        checkpoint = ROOT / original["checkpoint"]
        assert sha(checkpoint) == original["file_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        internal = payload.pop("checkpoint_content_sha256")
        assert internal == original["content_sha256"] == recursive_content_hash(payload)
        metadata = payload["metadata"]
        assert metadata["method"] == original["method"] and metadata["seed"] == original["seed"]
        assert metadata["epoch"] == original["selected_epoch"]
        assert metadata["model_configuration"] == model_configuration(original["method"])
        model = SharedNodeMLP() if original["method"] == "nograph" else BatchedGAT(original["method"] == "epistemic_gat")
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        scorer = f'{original["method"]}_{original["seed"]}'
        rel = "states/" + scorer + ".pt"
        (HERE / "states").mkdir(exist_ok=True)
        # Compact tensor-only exports, exact selected-state tensors; no optimizer/RNG.
        torch.save(payload["model_state"], HERE / rel)
        exported = torch.load(HERE / rel, map_location="cpu", weights_only=True)
        assert all(torch.equal(v, exported[k]) for k, v in payload["model_state"].items())
        rows.append({**original, "scorer_id": scorer, "bundle_state": rel, "bundle_state_sha256": sha(HERE / rel),
                     "original_content_hash_verified": True, "strict_load_passed": True,
                     "tensor_content_sha256": recursive_content_hash(exported)})
    assert len(rows) == 15
    write("model_registry.json", {"scorers": rows, "no_retraining": True, "no_reselection": True})
    write("partition_binding.json", {"partition_seal": sha(PARTITION / "SHA256SUMS"), "CAL_count": 125, "EVAL_count": 500,
        "overlap": 0, "union": 625, "union_equals_exact_eligible": True, "CAL_membership_sha256": sha(HERE / "final_conformal_cal.txt"),
        "EVAL_membership_sha256": sha(HERE / "final_evaluation.txt"), "membership_JSON_sha256": sha(PARTITION / "partition_membership.json"),
        "source_inventory_seal": "e5219b9dae9447bf8d818828e5ef9ee7987cc5d375f345c4be7d52ad045a333c",
        "eligible_inventory_seal": "9c08a8cc404b857d8030d316eac12f26fd9145e301ecdce1c081a4c82e981218",
        "seed_permanently_spent": True, "RNG_or_permutation_replayed": False,
        "historical_exclusions": ["test/anomaly/Town01/change-weather/scenario-1", "test/anomaly/Town01/change-weather/scenario-10"]})
    write("conformal_specification.json", {**{key: protocol[key] for key in ("alpha", "numerics", "score", "scenario_score", "eligible_ticks", "prediction_set", "calibration", "primary_guarantee", "scope_limits")},
        "n_cal": 125, "rank_1_based": 120, "threshold_count": 15, "source": "committed Amendment 003 conformal_protocol.json",
        "source_sha256": sha(PROTO / "conformal_protocol.json"), "fitted_in_preparation": False})
    write("label_semantics.json", {"resolved_before_TEST_access": True, "authoritative_field": "anomaly-observation.feather['anomaly']",
        "tick_index": "anomaly-observation.feather['tick']; integer exactly 0..n-1",
        "normal_scenarios": "Use official per-tick booleans. False -> normal=0; do not synthesize labels from directory names or missing files.",
        "anomaly_scenarios": "False -> normal=0; True -> anomaly=1. Anomaly directories may contain all-False annotations.",
        "normality_probability": "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration construction",
        "official_class_order": ["normal=0", "anomaly=1"], "P_y": "P(0)=q_normal, P(1)=1-q_normal",
        "tick_scoring": "1-P(Y_t) at every synchronized tick t>=1; tick 0 excluded; maximum over all such ticks per scenario",
        "alignment": "No interpolation, zero fill, missing-tick skips or anomaly directory substitution. IMU row i aligns tick i; explicit tick/frame if present must agree.",
        "evidence": ["evidence/sources/carlanomaly_loader.py.txt::load_timestep_labels/_check_synchronization",
                     "evidence/sources/official_anomaly_timestep.py.txt::_get_label_cache", "evidence/protocol/conformal_protocol.json::target"]})
    write("protected_eval_policy.json", {"membership_override": False, "training": False, "tuning": False,
        "archive_URL": "https://data.carlanomaly.de/v1/carlanomaly-base-test.tar.gz", "expected_compressed_size": 91538225599,
        "expected_archive_sha256": "267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a",
        "stream": "One sequential stream from byte zero. No GET/HEAD/range/retry during preparation; no compressed-offset seeking/resume.",
        "gate": "Strict structural root -> immutable CAL/EVAL/exclusion membership -> CAL-only decoder callback with second CAL gate",
        "eval_bytes": "Compressed transport and decompressed opaque body discard are transport only; never image, Feather, sensor, label, feature, or model interpretation.",
        "unknown_ID": "Stop before payload read or callback", "excluded_ID": "Opaque discard only; never scientific processing",
        "output": "/kaggle/working/cognix_final_conformal_cal_v1", "finalize_thresholds_only_after_full_archive_verified": True,
        "memory_caps": {"transport_chunk_bytes": 65536, "required_member_bytes": 16777216, "scenario_ticks": 100000,
                        "total_CAL_ticks": 2000000, "image_pixels": 16000000},
        "zero_decode_proof_limit": "Code gates, synthetic decoder spies, and future runtime ledger; not an OS-wide non-access proof"})
    import PIL
    import pyarrow
    write("runtime_lock.json", {"Python": "3.12.13", "NumPy": "2.0.2", "PyTorch": "2.10.0+cu128",
        "Pillow": PIL.__version__, "PyArrow": pyarrow.__version__, "CUDA": "12.8", "cuDNN": 91002,
        "device": "one visible Tesla T4", "decoder_pin_basis": "Existing local runtime, synthetic schema fixtures only; fixed before any TEST payload access"})
    (HERE / "runtime_requirements.txt").write_text(f'numpy==2.0.2\ntorch==2.10.0+cu128\nPillow=={PIL.__version__}\npyarrow=={pyarrow.__version__}\npandas==2.2.3\n', encoding="utf-8")
    scientific_paths = [p for folder in ("frozen_sources", "upstream", "states") for p in (HERE / folder).rglob("*") if p.is_file()]
    scientific_paths += [HERE / name for name in ("calibration_adapter.py", "header_helpers.py", "streaming.py", "runner.py", "frozen_runtime.py", "label_semantics.json", "conformal_specification.json", "runtime_lock.json")]
    write("scientific_bindings.json", {"git_HEAD": initial_head, "git_branch": initial_branch, "committed_protocol_evidence": evidence,
        "original_source_bindings": sources, "protocol_manifest_sha256": protocol_seal,
        "bundle_scientific_file_sha256": {p.relative_to(HERE).as_posix(): sha(p) for p in scientific_paths},
        "pipeline": {"feature_source": "cognix/adapters/carla/real_features.py", "functions": ["camera_embedding_features (18D handcrafted block means/variance/brightness)", "segmentation_histogram_features (29D R-channel relative frequencies; actual code normalizes by total despite L2 docstring)", "imu_window_features (10D acceleration-only)"],
            "L": 12, "eligible_ticks": "1..actual final tick; [max(0,t-11),t]; per-scenario reset", "node_order": ["Camera", "IMU", "Seg"],
            "agent_restore": "exact AST-bound graph_fit_export.restore_agents; no fit", "uq": "normality.ensemble_to_uncertainty: H(mean p), mean H(p), max(0,total-aleatoric)",
            "aleatoric_semantics": "Entropy decomposition term, not physical sensor-noise measurement", "GNSS": "Excluded: structurally invalid calibration",
            "graph": "exact graph_training_models.SharedNodeMLP/BatchedGAT; mean node sigmoids; raw float64 E reciprocal then float32 sender prior 1/(1+E) both layers",
            "execution": "float32 models, float64 upstream/E, eval/inference mode, fixed per-scenario batches of 256, no AMP/TF32",
            "probability": "P(target=normal) under TRAIN-derived clean-vs-pseudo calibration; not P(ACT), safety, or established official anomaly posterior"}})
    # Fresh subprocess: the build process used frozen namespaces to load states.
    test = subprocess.run([sys.executable, "-B", str(HERE / "synthetic_verification.py")], cwd=HERE,
                          capture_output=True, text=True, check=True)
    checks = json.loads(test.stdout.strip().splitlines()[-1])
    write("synthetic_verification_results.json", checks)
    assert git("rev-parse", "HEAD").decode().strip() == initial_head
    assert not git("status", "--porcelain=v1", "--untracked-files=no")
    current_status = [line for line in git("status", "--porcelain=v1").decode().splitlines()
                      if "reports/carla_final_conformal_cal_bundle_v1/" not in line]
    assert current_status == initial_status, "Unrelated untracked inventory changed"
    write("pre_access_validation.json", {**checks, "git_HEAD": initial_head, "git_branch": initial_branch,
        "all_15_frozen_scorers_resolved": True, "label_semantics_resolved": True, "committed_preregistration_matches": True,
        "partition_seal_verified": True, "CAL_count": 125, "EVAL_count": 500, "overlap": 0, "union": 625,
        "historical_exclusions_absent": True, "all_original_checkpoint_file_and_content_hashes_verified": True,
        "all_exported_states_strictly_loadable": True, "no_training_or_tuning_entry_point": True,
        "scientific_bindings_hashed": True, "output_paths_isolated": True, "outside_bundle_git_status_unchanged": True,
        "tracked_worktree_clean": True, "NO_TEST_PAYLOAD_ACCESSED": True, "archive_HTTP_GET_HEAD_range_requests": 0,
        "conformal_fitting_on_TEST": False, "Kaggle_notebook_executed": False, "commit": False, "push": False,
        "runtime_limitation": "Kaggle preflight must verify locked versions/device before future access; no Kaggle execution in preparation",
        "readiness": "READY_FOR_FINAL_CONFORMAL_CAL_EXECUTION"})
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Frozen COGNIX FINAL_CONFORMAL_CAL\n", "Preparation only. This notebook has not been executed. Run preflight first. The execution cell is disabled and requires separate later CAL authorization. Attach the sealed bundle as a Kaggle input and extract it into /kaggle/working/cognix_cal_bundle. Check the externally recorded ZIP SHA-256 before extraction. Runtime mismatch halts before access. No automatic retry, training, tuning or EVAL decoding.\n"]},
        {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": ["import subprocess, sys\n", "from pathlib import Path\n", "BUNDLE = Path('/kaggle/working/cognix_cal_bundle')\n", "subprocess.run([sys.executable, '-B', str(BUNDLE/'runner.py'), '--preflight'], check=True)\n"]},
        {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": ["EXECUTE_SEPARATELY_AUTHORIZED_FINAL_CAL = False\n", "if EXECUTE_SEPARATELY_AUTHORIZED_FINAL_CAL:\n", "    subprocess.run([sys.executable, '-B', str(BUNDLE/'runner.py'), '--execute-authorized-final-cal'], check=True)\n"]}]}
    write("cognix_final_conformal_cal_v1.ipynb", notebook)
    (HERE / "README.md").write_text('''# FINAL_CONFORMAL_CAL execution bundle v1

READY_FOR_FINAL_CONFORMAL_CAL_EXECUTION means the offline preparation gates passed. It does not authorize archive access. NO TEST PAYLOAD ACCESSED during preparation. The Kaggle notebook remains unexecuted and its execution cell defaults to False.

The frozen 125 CAL and 500 EVAL scenarios are bound to the committed partition. Both historical exclusions remain opaque. Unknown roots stop before payload decoding. The single gzip stream physically transports and decompresses bytes from the monolithic archive; EVAL bodies are chunk-discarded without image, Feather, sensor, annotation, feature, or model interpretation. CAL payload decoding has two membership gates. Strict header parsing is reused from the already validated inventory parser, including its old-GNU header amendment.

Offline verification: `python -B verify_bundle.py`. Synthetic fixtures: `python -B synthetic_verification.py` in a fresh process with NumPy/PyTorch/Pillow/PyArrow/pandas available. No fixture accesses the archive URL. `python -B runner.py --preflight` verifies seals without network access. The only execution switch is `--execute-authorized-final-cal`, for a later separately authorized milestone. There are no CAL-list, EVAL-list, alpha, checkpoint, URL, output, seed, training or tuning overrides. Changing bundle bytes invalidates the published manifest/ZIP seal. Those external hashes are the trust anchor; the detached manifest is an integrity check, not a cryptographic signature.

Upload the small ZIP as a Kaggle input. Verify its external SHA-256, safely extract the contents into `/kaggle/working/cognix_cal_bundle`, and use the notebook. The future runtime must match Python 3.12.13, NumPy 2.0.2, PyTorch 2.10.0+cu128, CUDA 12.8/cuDNN 91002 and one visible T4, plus the pinned decoder versions in runtime_lock.json. Install/validate requirements before enabling execution; package installation makes no archive request. Exact Kaggle runtime verification remains a mandatory execution-time gate. Run all commands with `-B` to avoid adding unsealed bytecode files. Preparation helper prepare_bundle.py is local-only and requires repository evidence; it is never invoked by the notebook or runner.

All fifteen model/seed checkpoints match the committed preregistration, including internal scientific hashes. Tensor-only states preserve the exact selected tensors without optimizer/RNG states. The existing feature, one-class probability/UQ, graph and readout implementations are reused. Only the exact existing agent-restoration function and static graph constants are extracted from the exporter to remove TRAIN/raw-data dependencies; AST-node bindings are recorded. Fitting methods are disabled in the inference loader, parameters have no gradients, and training mode is disabled. No trainer, acquisition, pseudo-corruption or generic orchestrator is bundled as an active import. Evidence scripts are archived as .txt. GNSS is excluded. Camera/IMU/Seg nodes receive p/E/A, with the unchanged float64 raw-E prior and float32 graph. Per-scenario batch size is fixed at 256 before access.

Normality means P(target=normal) under the TRAIN-derived clean-vs-pseudo construction. It is not safety, P(ACT), safe-driving probability, or an established official anomaly posterior. The aleatoric term is the frozen mean-member-entropy decomposition, not a physical sensor-noise measurement. The segmentation implementation divides histogram counts by total; its historical L2 wording does not change the frozen code.

Official tick labels are the anomaly boolean in anomaly-observation.feather, aligned by tick exactly 0..n-1. False maps normal=0 and True anomaly=1, including within anomaly directories. Normal directories also require official labels. Missing/misaligned tables or modalities halt; no directory fallback. Tick 0 is excluded, and causal IMU windows are [max(0,t-11),t] with no cross-scenario leakage. Each scorer uses P(0)=q, P(1)=1-q and one scenario maximum S_j=max_t(1-P_t(Y_t)). Alpha=.05; k=(19*(n_cal+1)+19)//20=120 for n_cal=125; Q is the augmented kth order statistic of the 125 scores plus infinity. Comparison is inclusive <=. Infinity is null with quantile_is_infinite=true. This is fifteen separate pooled thresholds, identical membership, no model ensemble, subgroup thresholds, tuning or method ranking. The guarantee and its finite-catalogue marginal limitations remain those of committed Amendment 003.

Future outputs are isolated at /kaggle/working/cognix_final_conformal_cal_v1. One attempt marker is written before the sole GET from byte zero. Existing outputs refuse restart; there is no retry/resume. SHA-256 and compressed size are verified incrementally through complete EOF; thresholds are not finalized until the official expected size/hash and complete scenario/CAL sets match. Raw archives and raw sensor files are never retained. Image features and per-tick calibration scores use bounded compact memory. Failures leave audit/ledger/result seals and require separate authorization before another scientific attempt.

Final output includes archive receipt, access/processing ledgers, zero-EVAL decoder receipt, per-scorer scenario scores, fifteen thresholds and receipts, CAL-only fit diagnostics, exception audit, result manifest and SHA-256 seals. No FINAL_EVALUATION performance or method ranking is computed. Zero-access assertions describe the executed code/tools and ledger, not an OS-wide access trace.

Manifest scope includes all bundle files except BUNDLE_SHA256SUMS, its detached digest, the upload ZIP and the ZIP detached digest. The ZIP contains the sealed files and manifest/digest, excluding itself. This avoids circular hash dependencies. Existing intentional untracked artifacts and frozen repository files remain unchanged. No commit or push.
''', encoding="utf-8")
    files = sorted(p for p in HERE.rglob("*") if p.is_file())
    assert all("__pycache__" not in p.parts for p in files)
    (HERE / "BUNDLE_SHA256SUMS").write_text("".join(sha(p) + "  " + p.relative_to(HERE).as_posix() + "\n" for p in files), encoding="utf-8")
    manifest_sha = sha(HERE / "BUNDLE_SHA256SUMS")
    (HERE / "BUNDLE_SHA256SUMS.sha256").write_text(manifest_sha + "  BUNDLE_SHA256SUMS\n", encoding="utf-8")
    with zipfile.ZipFile(HERE / ZIP_NAME, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + [HERE / "BUNDLE_SHA256SUMS", HERE / "BUNDLE_SHA256SUMS.sha256"]:
            info = zipfile.ZipInfo(p.relative_to(HERE).as_posix(), (2026, 10, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    zip_sha = sha(HERE / ZIP_NAME)
    (HERE / (ZIP_NAME + ".sha256")).write_text(zip_sha + "  " + ZIP_NAME + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-B", str(HERE / "verify_bundle.py")], cwd=HERE, check=True)
    print(json.dumps({"HEAD": initial_head, "manifest_sha256": manifest_sha, "upload_ZIP_sha256": zip_sha,
                      "ZIP_bytes": (HERE / ZIP_NAME).stat().st_size, "synthetic_groups_passed": checks["synthetic_verification_groups_passed"]}))


if __name__ == "__main__":
    main()

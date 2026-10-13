#!/usr/bin/env python3
"""
COGNIX Experiment 2B final TEST Phase A.

Label-blind final evaluation phase:
  * authenticates all pre-TEST frozen inputs before sensor-payload access;
  * reads only RGB, segmentation and IMU TEST payloads;
  * never opens anomaly-observation.feather or anomaly-front/*;
  * applies only frozen TRAIN-derived feature/UQ transformations;
  * evaluates all 15 frozen checkpoints in eval/no-grad mode;
  * writes and seals predictions before any TEST label reader may run.

This file is prepared and reviewed before Phase-A TEST payload access.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
import traceback
from pathlib import Path

EXPECTED_ARCHIVE = Path(r"E:\carlanomaly-base-test.tar.gz")
EXPECTED_ARCHIVE_BYTES = 91538225599
EXPECTED_ARCHIVE_SHA256 = "267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a"
EXPECTED_GATE2_RUNTIME_SEAL = "58cc01e1203b697e201276e044e69b118063801d9300569f59df97e74f0f912a"
EXPECTED_CHECKPOINT_FREEZE_SEAL = "9f787f8c3f8cb093b1daaea1a3bf414265456f5f0b139ac7f1d5e176c9d9a62c"
EXPECTED_PROTOCOL_SEAL = "d4ceb25023f49acc96ab2a62f05f7fdc92c88c608705ce5fbc34dfe64367691e"
EXPECTED_STRUCTURE_SEAL = "8e6493a29170a7effb6eec25ce2522f2c59f1ce28ea003af8766975bf7ed5c82"
EXPECTED_DOWNLOAD_SEAL = "45aa8860a0f3ca6423fffd0977181cfff2ea61f57d4772ba57c293d7e0cd43be"
EXPECTED_GRAPH_DATA_SEAL = "633b5801b86523c0a9396e45c02ddd23ae927a41ee032528eb0285bdbf296066"
EXPECTED_SCIENCE_SHA = "c8a82c361514fdad9807e23fbfd691c354db081cecdb22506a068a950a688a13"

NODE_ORDER = ("Camera", "IMU", "Seg")
DIMS = {"Camera": 18, "IMU": 10, "Seg": 29}
METHODS = ("NoGraph", "StandardGAT", "EpistemicGAT")
SEEDS = (101, 202, 303, 404, 505)
ADJACENCY = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
EXPECTED_SCENARIOS = 627
EXPECTED_FILES_PER_SCENARIO = 907
EXPECTED_FRAMES = 300
EXPECTED_EMITTED_TICKS = 299
EXPECTED_ROWS = EXPECTED_SCENARIOS * EXPECTED_EMITTED_TICKS
AUTHORIZATION = "PHASE_A_TEST_SENSOR_ACCESS_AUTHORIZED_V1"

FORBIDDEN_DIRECT = {"anomaly-observation.feather"}
FORBIDDEN_PREFIX = ("anomaly-front/",)
ALLOWED_DIRECT = {"imu.feather"}
ALLOWED_PREFIX = ("rgb-front/", "segmentation-front/")


class PhaseAError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise PhaseAError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, obj):
    path = Path(path)
    require(not path.exists(), "Refusing overwrite: " + str(path))
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path, chunk=8 << 20, progress=None):
    h = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            total += len(b)
            if progress is not None:
                progress(total)
    return h.hexdigest()


def verify_manifest_seal(root, expected=None):
    root = Path(root)
    sums = root / "SHA256SUMS"
    detached = root / "SHA256SUMS.sha256"
    require(sums.is_file() and detached.is_file(), "Missing sealed manifest: " + str(root))
    actual_seal = hashlib.sha256(sums.read_bytes()).hexdigest()
    recorded = detached.read_text(encoding="utf-8").strip().split()[0]
    require(actual_seal == recorded, "Detached bundle seal mismatch: " + str(root))
    if expected is not None:
        require(actual_seal == expected, "Unexpected bundle seal: " + str(root))
    rows = [line for line in sums.read_text(encoding="utf-8").splitlines() if line]
    seen = set()
    for line in rows:
        parts = line.split("  ", 1)
        require(len(parts) == 2, "Malformed SHA256SUMS row")
        digest, rel = parts
        require(rel not in seen, "Duplicate SHA256SUMS path")
        seen.add(rel)
        p = root / rel
        require(p.is_file(), "Missing sealed file: " + str(p))
        require(hashlib.sha256(p.read_bytes()).hexdigest() == digest, "Sealed file hash mismatch: " + str(p))
    return actual_seal


def seal_tree(root):
    root = Path(root)
    sums = root / "SHA256SUMS"
    detached = root / "SHA256SUMS.sha256"
    require(not sums.exists() and not detached.exists(), "Output already sealed")
    files = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256")],
        key=lambda p: p.relative_to(root).as_posix(),
    )
    lines = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + rel)
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    seal = hashlib.sha256(sums.read_bytes()).hexdigest()
    detached.write_text(seal + "  SHA256SUMS\n", encoding="utf-8", newline="\n")
    return seal, len(files)


def verify_runtime_versions(prep):
    import importlib.metadata as md
    current = {
        "numpy": md.version("numpy"),
        "pandas": md.version("pandas"),
        "pyarrow": md.version("pyarrow"),
        "Pillow": md.version("Pillow"),
        "torch": md.version("torch"),
    }
    recorded = prep["runtime"]
    for key in current:
        require(current[key] == recorded[key], "Runtime version drift: " + key)
    require(sys.version == recorded["python"], "Python version drift")
    require(hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == recorded["executable_sha256"],
            "Python executable hash drift")
    return current


def repo_paths(repo):
    report = repo / "reports" / "carla_train_full_v1"
    bundle = Path(__file__).resolve().parent
    return report, bundle


def verify_preexecution(repo, bundle, require_bundle_seal=None):
    report, _ = repo_paths(repo)
    prep = read_json(bundle / "preparation_initial.json")
    runtime = read_json(bundle / "runtime_and_constants_audit.json")
    design = read_json(bundle / "phase_A_design.json")
    scenario_manifest = read_json(bundle / "test_scenario_manifest.json")
    amendment = read_json(bundle / "test_scenario_manifest_order_amendment.json")
    inputs = read_json(bundle / "phase_A_input_bindings.json")

    require(prep["status"] == "PREPARATION_ONLY_NO_TEST_PAYLOAD_ACCESS", "Preparation status changed")
    require(runtime["status"] == "PASS" and runtime["exact_feature_runtime_match"] is True,
            "Runtime/constants audit not PASS")
    require(design["status"] == "FROZEN_DESIGN_PRE_EXECUTION_NO_TEST_PAYLOAD_ACCESS", "Phase-A design not frozen")
    require(amendment["status"] == "FROZEN_PRE_EXECUTION_PROVENANCE_CORRECTION_NO_TEST_PAYLOAD_ACCESS",
            "Scenario-order amendment not frozen")
    require(inputs["status"] == "PASS_PRE_EXECUTION_INPUTS_BOUND", "Input bindings not PASS")
    require(inputs["checkpoint_count"] == 15, "Checkpoint binding count changed")

    verify_runtime_versions(prep)
    require(verify_manifest_seal(report / "graph_test_checkpoint_freeze_bundle_v1",
                                 EXPECTED_CHECKPOINT_FREEZE_SEAL) == prep["checkpoint_freeze_seal"],
            "Checkpoint freeze binding changed")
    require(verify_manifest_seal(report / "graph_test_evaluation_protocol_freeze_bundle_v1",
                                 EXPECTED_PROTOCOL_SEAL) == prep["test_protocol_seal"],
            "Protocol binding changed")
    require(verify_manifest_seal(report / "graph_test_structure_inspection_v1",
                                 EXPECTED_STRUCTURE_SEAL) == prep["test_structure_seal"],
            "Structure binding changed")
    require(verify_manifest_seal(report / "graph_test_download_evidence_v1",
                                 EXPECTED_DOWNLOAD_SEAL) == prep["test_download_evidence_seal"],
            "Download evidence binding changed")
    verify_manifest_seal(report / "gate2_train_health_v3", EXPECTED_GATE2_RUNTIME_SEAL)

    if require_bundle_seal is not None:
        require(verify_manifest_seal(bundle, require_bundle_seal) == require_bundle_seal,
                "Execution bundle seal mismatch")

    for record in inputs["sources"]:
        p = repo / record["path"]
        require(p.is_file(), "Bound source missing: " + record["path"])
        require(hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"],
                "Bound source changed: " + record["path"])

    checkpoints = []
    for record in inputs["checkpoints"]:
        p = report / record["path"]
        require(p.is_file(), "Checkpoint missing: " + str(p))
        require(hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"],
                "Checkpoint hash mismatch: " + record["run_id"])
        checkpoints.append((record, p))
    require(len(checkpoints) == 15, "Not all checkpoints authenticated")

    roots = [r["scenario_root"] for r in scenario_manifest["scenarios"]]
    require(len(roots) == len(set(roots)) == EXPECTED_SCENARIOS, "Scenario identity set changed")
    require(set(r["files_in_scenario"] for r in scenario_manifest["scenarios"]) == {EXPECTED_FILES_PER_SCENARIO},
            "Scenario metadata file count changed")
    require(amendment["invalid_interpretation"].startswith("archive_scenario_ordinal"),
            "Scenario-order correction missing")
    return prep, design, inputs, frozenset(roots), checkpoints


def load_science_and_scorers(repo):
    report, _ = repo_paths(repo)
    gate2 = report / "gate2_execution_bundle_v3"
    if str(gate2) not in sys.path:
        sys.path.insert(0, str(gate2))
    import gate2_science
    science = gate2_science.load_science()

    scorers = {}
    for modality in NODE_ORDER:
        fit = report / "gate2_train_health_v3" / "units" / ("fit-" + modality)
        audit_root = report / "gate2_train_health_v3" / "units" / ("audit-" + modality)
        audit = read_json(audit_root / "audit.json")
        require(audit["status"] == "VALID", "Upstream calibration audit not VALID: " + modality)
        require(audit["optimization_info"]["finite_optimum"] is True, "Nonfinite frozen calibration: " + modality)
        members = []
        import numpy as np
        for i in range(5):
            with np.load(fit / ("member_%d.npz" % i), allow_pickle=False) as z:
                values = {k: z[k] for k in z.files}
            require(set(values) == {"mean", "precision", "d2_scale", "shrinkage"},
                    "Unexpected member state keys")
            require(values["mean"].shape == (DIMS[modality],), "Member mean shape mismatch")
            require(values["precision"].shape == (DIMS[modality], DIMS[modality]), "Member precision shape mismatch")
            require(values["mean"].dtype == np.float64 and values["precision"].dtype == np.float64,
                    "Member dtype mismatch")
            require(np.isfinite(values["mean"]).all() and np.isfinite(values["precision"]).all(),
                    "Nonfinite member state")
            require(float(values["d2_scale"]) > 0.0 and float(values["shrinkage"]) == 0.1,
                    "Member scalar state mismatch")
            members.append(values)

        with np.load(audit_root / "audit_arrays.npz", allow_pickle=False) as z:
            params = z["calibrator_parameters_a_b_score_mean_std"].astype(np.float64, copy=False)
        require(params.shape == (4,) and np.isfinite(params).all(), "Frozen calibration parameter shape/domain")
        expected_params = np.asarray(audit["calibrator_parameters"], dtype=np.float64)
        require(np.array_equal(params, expected_params), "Frozen calibration JSON/NPZ disagreement")

        ensemble = object.__new__(science["BootstrapNormalityEnsemble"])
        ensemble.n_members = 5
        ensemble.seed = 42
        ensemble.factory = science["MahalanobisNormality"]
        ensemble._members = []
        for values in members:
            member = object.__new__(science["MahalanobisNormality"])
            member._mean = values["mean"].copy()
            member._prec = values["precision"].copy()
            member._d2_scale = float(values["d2_scale"])
            member.shrinkage = float(values["shrinkage"])
            ensemble._members.append(member)

        calibrator = object.__new__(science["EnsemblePredictiveCalibrator"])
        calibrator.a = float(params[0])
        calibrator.b = float(params[1])
        calibrator._s_mean = float(params[2])
        calibrator._s_std = float(params[3])
        calibrator._fitted = True
        calibrator._validity = copy.deepcopy(audit["optimization_info"])
        for field in ("nll_initial", "nll_final", "n_iter", "converged"):
            setattr(calibrator, "_" + field, audit["optimization_info"][field])
        calibrator.require_valid(diagnostic=False)
        scorers[modality] = (ensemble, calibrator)

    def node(modality, vector):
        import numpy as np
        x = np.asarray(vector)
        require(x.dtype == np.float64 and x.shape == (DIMS[modality],) and np.isfinite(x).all(),
                "Invalid compact feature vector: " + modality)
        ensemble, calibrator = scorers[modality]
        scores = ensemble.predict_normality(x)
        require(scores.shape == (5,) and np.isfinite(scores).all(), "Invalid member scores")
        probs = np.asarray([calibrator.prob_normal(float(v), diagnostic=False) for v in scores], dtype=np.float64)
        require(np.isfinite(probs).all() and np.all(probs >= 0.0) and np.all(probs <= 1.0),
                "Invalid calibrated member probabilities")
        uq = science["ensemble_to_uncertainty"](probs)
        out = np.asarray([float(probs.mean()), float(uq.epistemic), float(uq.aleatoric)], dtype=np.float64)
        require(np.isfinite(out).all(), "Nonfinite node output")
        return out

    return science, node


def load_models(repo, checkpoints):
    report, _ = repo_paths(repo)
    kaggle = report / "kaggle_graph_training_bundle_v1"
    if str(kaggle) not in sys.path:
        sys.path.insert(0, str(kaggle))
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from kaggle_models import paired_models
    by_seed = {}
    for seed in SEEDS:
        models, _ = paired_models(seed)
        by_seed[seed] = models

    loaded = []
    for record, path in checkpoints:
        method = record["method"]
        seed = int(record["seed"])
        require(method in METHODS and seed in SEEDS, "Foreign checkpoint method/seed")
        model = by_seed[seed][method]
        chosen = torch.load(path, map_location="cpu", weights_only=True)
        require(chosen["method"] == method and int(chosen["seed"]) == seed, "Checkpoint identity mismatch")
        require(int(chosen["epoch"]) == int(record["epoch"]), "Checkpoint epoch mismatch")
        require(chosen["graph_seal"] == EXPECTED_GRAPH_DATA_SEAL, "Checkpoint graph-data seal mismatch")
        require(chosen["scientific_content_sha256"] == EXPECTED_SCIENCE_SHA, "Checkpoint science SHA mismatch")
        model.load_state_dict(chosen["model_state"], strict=True)
        model.eval()
        require(all(not p.requires_grad or torch.isfinite(p).all().item() for p in model.parameters()),
                "Nonfinite model parameter")
        loaded.append((record, model))
    require(len(loaded) == 15, "Model load count mismatch")
    return loaded


def scenario_root_from_name(name):
    parts = name.strip("/").split("/")
    if len(parts) >= 5 and parts[0] == "test" and parts[1] == "anomaly":
        return "/".join(parts[:5]), "/".join(parts[5:])
    if len(parts) >= 4 and parts[0] == "test" and parts[1] == "normal":
        return "/".join(parts[:4]), "/".join(parts[4:])
    return None, None


def member_class(relative):
    if relative in FORBIDDEN_DIRECT or relative.startswith(FORBIDDEN_PREFIX):
        return "FORBIDDEN"
    if relative in ALLOWED_DIRECT or relative.startswith(ALLOWED_PREFIX):
        return "ALLOWED"
    return "UNUSED"


def read_allowed_member(tf, member, relative):
    require(member_class(relative) == "ALLOWED", "Attempt to open non-allowed TEST payload: " + relative)
    max_size = (128 << 20) if relative.startswith(ALLOWED_PREFIX) else (64 << 20)
    require(member.size <= max_size, "Allowed TEST payload exceeds bounded size")
    stream = tf.extractfile(member)
    require(stream is not None, "Could not open allowed TEST payload")
    data = stream.read()
    require(len(data) == member.size, "Short TEST payload read")
    return data


def image_array(data):
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        arr = np.asarray(image).copy()
    require(arr.size > 0 and arr.ndim in (2, 3), "Malformed decoded image")
    return arr


class ScenarioAccumulator:
    def __init__(self, root, science, oov_stream):
        import numpy as np
        self.root = root
        self.science = science
        self.oov_stream = oov_stream
        self.files = 0
        self.camera = np.empty((EXPECTED_FRAMES, 18), dtype=np.float64)
        self.seg = np.empty((EXPECTED_FRAMES, 29), dtype=np.float64)
        self.camera_seen = np.zeros(EXPECTED_FRAMES, dtype=np.bool_)
        self.seg_seen = np.zeros(EXPECTED_FRAMES, dtype=np.bool_)
        self.imu_bytes = None
        self.allowed_reads = 0

    def observe_file(self):
        self.files += 1

    def consume(self, relative, data):
        import numpy as np
        cls = member_class(relative)
        require(cls == "ALLOWED", "Scenario consume called on non-allowed payload")
        self.allowed_reads += 1
        if relative == "imu.feather":
            require(self.imu_bytes is None, "Duplicate IMU payload")
            self.imu_bytes = data
            return
        if relative.startswith("rgb-front/"):
            name = Path(relative).name
            require(name.endswith(".jpg") and name[:-4].isdigit(), "Noncanonical RGB member")
            tick = int(name[:-4])
            require(tick in range(EXPECTED_FRAMES), "RGB tick outside frozen TEST structure")
            require(not self.camera_seen[tick], "Duplicate RGB tick")
            arr = image_array(data)
            feat = self.science["camera_embedding_features"](arr)
            require(feat.dtype == np.float64 and feat.shape == (18,) and np.isfinite(feat).all(),
                    "Camera feature contract failure")
            self.camera[tick] = feat
            self.camera_seen[tick] = True
            return
        if relative.startswith("segmentation-front/"):
            name = Path(relative).name
            require(name.endswith(".png") and name[:-4].isdigit(), "Noncanonical segmentation member")
            tick = int(name[:-4])
            require(tick in range(EXPECTED_FRAMES), "Seg tick outside frozen TEST structure")
            require(not self.seg_seen[tick], "Duplicate segmentation tick")
            arr = image_array(data)
            transformed, counts = self.science["sanitize"](arr)
            feat = self.science["segmentation_histogram_features"](transformed)
            require(feat.dtype == np.float64 and feat.shape == (29,) and np.isfinite(feat).all(),
                    "Seg feature contract failure")
            self.seg[tick] = feat
            self.seg_seen[tick] = True
            record = {
                "schema": "COGNIX-TEST-Seg-OOV-frame-v1",
                "scenario_root": self.root,
                "tick": tick,
                **counts,
                "descriptive_only": True,
                "decision_use": "none",
            }
            self.oov_stream.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            return
        raise PhaseAError("Unknown allowed relative payload: " + relative)

    def finalize(self):
        import numpy as np
        import pandas as pd
        require(self.files == EXPECTED_FILES_PER_SCENARIO, "Scenario file-count mismatch: " + self.root)
        require(self.allowed_reads == 601, "Allowed payload read count mismatch: " + self.root)
        require(self.camera_seen.all() and self.seg_seen.all(), "Incomplete image coverage: " + self.root)
        require(self.imu_bytes is not None, "Missing IMU: " + self.root)
        frame = pd.read_feather(io.BytesIO(self.imu_bytes))
        cols = ["acceleration_x", "acceleration_y", "acceleration_z"]
        require(all(c in frame.columns for c in cols), "IMU acceleration schema mismatch")
        table = frame[cols].to_numpy(dtype=np.float64)
        require(len(table) >= EXPECTED_FRAMES and table.shape[1] == 3 and np.isfinite(table).all(),
                "IMU table contract failure")
        imu = np.empty((EXPECTED_EMITTED_TICKS, 10), dtype=np.float64)
        for tick in range(1, EXPECTED_FRAMES):
            lo = max(0, tick - 11)
            feat = self.science["imu_window_features"](table[lo:tick + 1])
            require(feat.dtype == np.float64 and feat.shape == (10,) and np.isfinite(feat).all(),
                    "IMU feature contract failure")
            imu[tick - 1] = feat
        return self.camera[1:].copy(), imu, self.seg[1:].copy()


def build_nodes(camera, imu, seg, node_fn):
    import numpy as np
    require(camera.shape == (299, 18) and imu.shape == (299, 10) and seg.shape == (299, 29),
            "Scenario compact feature matrix shape mismatch")
    nodes = np.empty((299, 3, 3), dtype=np.float64)
    sources = {"Camera": camera, "IMU": imu, "Seg": seg}
    for row in range(299):
        for col, modality in enumerate(NODE_ORDER):
            nodes[row, col] = node_fn(modality, sources[modality][row])
    require(np.isfinite(nodes).all(), "Nonfinite canonical graph nodes")
    require(np.all(nodes[:, :, 0] >= 0.0) and np.all(nodes[:, :, 0] <= 1.0), "q_normal node domain failure")
    require(np.all(nodes[:, :, 1:] >= 0.0), "Negative UQ component")
    return nodes


def predict_nodes(nodes, loaded_models):
    import numpy as np
    import torch
    x = torch.tensor(nodes, dtype=torch.float32)
    e = torch.tensor(nodes[:, :, 1], dtype=torch.float64)
    q = np.empty((len(nodes), len(loaded_models)), dtype=np.float64)
    predicted = np.empty((len(nodes), len(loaded_models)), dtype=np.bool_)
    with torch.no_grad():
        for j, (record, model) in enumerate(loaded_models):
            values = model(x, e).detach().cpu().numpy().astype(np.float64)
            require(values.shape == (len(nodes),) and np.isfinite(values).all(), "Invalid model predictions")
            require(np.all(values >= 0.0) and np.all(values <= 1.0), "Prediction probability domain failure")
            q[:, j] = values
            predicted[:, j] = (1.0 - values) >= float(record["threshold"])
    return q, predicted


def self_test(repo):
    bundle = Path(__file__).resolve().parent
    prep, design, inputs, roots, checkpoints = verify_preexecution(repo, bundle)
    science, node_fn = load_science_and_scorers(repo)
    loaded = load_models(repo, checkpoints)

    import numpy as np
    rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    camera = science["camera_embedding_features"](rgb)
    require(camera.shape == (18,) and camera.dtype == np.float64, "Synthetic Camera self-test failed")

    seg_raw = np.zeros((8, 8, 3), dtype=np.uint8)
    seg_raw[..., 0] = 31
    seg_clean, counts = science["sanitize"](seg_raw)
    require(np.all(seg_clean == 22) and counts["count_remapped"] == 64, "Synthetic Seg OOV self-test failed")
    seg = science["segmentation_histogram_features"](seg_clean)
    require(seg.shape == (29,) and seg[22] == 1.0, "Synthetic Seg histogram self-test failed")

    imu_raw = np.arange(36, dtype=np.float64).reshape(12, 3) / 10.0
    imu = science["imu_window_features"](imu_raw)
    require(imu.shape == (10,) and np.isfinite(imu).all(), "Synthetic IMU self-test failed")

    nodes = np.stack([
        np.stack([node_fn("Camera", camera), node_fn("IMU", imu), node_fn("Seg", seg)])
        for _ in range(2)
    ])
    q, predicted = predict_nodes(nodes, loaded)
    require(q.shape == (2, 15) and predicted.shape == (2, 15), "Synthetic graph inference shape failure")

    print(json.dumps({
        "status": "SELF_TEST_PASS_NO_TEST_PAYLOAD_ACCESS",
        "scenario_identity_count": len(roots),
        "checkpoints_authenticated": len(checkpoints),
        "models_loaded": len(loaded),
        "synthetic_q_shape": list(q.shape),
        "test_archive_opened": False,
        "test_payload_opened": False,
        "test_labels_opened": False,
        "predictions_on_TEST_generated": False,
    }, sort_keys=True))


def execute_phase_a(repo, bundle_seal, authorization):
    require(authorization == AUTHORIZATION, "Explicit Phase-A TEST sensor authorization token required")
    report, bundle = repo_paths(repo)
    require(bundle == Path(__file__).resolve().parent, "Execution bundle location mismatch")
    prep, design, inputs, frozen_roots, checkpoints = verify_preexecution(
        repo, bundle, require_bundle_seal=bundle_seal
    )
    science, node_fn = load_science_and_scorers(repo)
    loaded_models = load_models(repo, checkpoints)

    archive = EXPECTED_ARCHIVE
    require(archive.is_file(), "TEST archive missing")
    require(archive.stat().st_size == EXPECTED_ARCHIVE_BYTES, "TEST archive byte-size mismatch")

    output = report / "graph_test_phase_A_predictions_v1_staging"
    final = report / "graph_test_phase_A_predictions_v1"
    require(not output.exists() and not final.exists(), "Fresh Phase-A output namespace required")
    output.mkdir()
    start = time.time()
    start_record = {
        "schema": "COGNIX-TEST-phase-A-execution-start-v1",
        "status": "STARTED_LABEL_BLIND_PHASE_A",
        "authorization": authorization,
        "execution_bundle_seal": bundle_seal,
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "test_labels_opened": False,
        "metrics_computed": False,
    }
    write_new_json(output / "execution_start.json", start_record)

    last_progress = [time.monotonic()]

    def progress_hash(n):
        now = time.monotonic()
        if now - last_progress[0] >= 30:
            print("Archive pre-execution SHA256 bytes:", n, flush=True)
            last_progress[0] = now

    try:
        measured_sha = sha256_file(archive, progress=progress_hash)
        require(measured_sha == EXPECTED_ARCHIVE_SHA256, "TEST archive SHA256 mismatch before Phase A")

        import numpy as np
        all_nodes = np.empty((EXPECTED_ROWS, 3, 3), dtype=np.float64)
        all_scenario_index = np.empty(EXPECTED_ROWS, dtype=np.int32)
        all_tick = np.empty(EXPECTED_ROWS, dtype=np.int16)
        all_q = np.empty((EXPECTED_ROWS, 15), dtype=np.float64)
        all_predicted = np.empty((EXPECTED_ROWS, 15), dtype=np.bool_)

        oov_path = output / "segmentation_oov.jsonl.gz"
        scenario_order = []
        seen_roots = set()
        current = None
        current_root = None
        row_cursor = 0
        file_count_total = 0
        allowed_payload_reads = 0
        forbidden_payload_opens = 0
        label_payload_opens = 0
        tar_members = 0
        last_print = time.monotonic()

        with gzip.GzipFile(filename=str(oov_path), mode="wb", mtime=0) as oov_stream:
            with archive.open("rb") as raw:
                with tarfile.open(fileobj=raw, mode="r|gz") as tf:
                    for member in tf:
                        tar_members += 1
                        if not member.isfile():
                            continue
                        root_name, relative = scenario_root_from_name(member.name)
                        require(root_name is not None, "Unexpected TEST file outside canonical scenario roots: " + member.name)
                        require(root_name in frozen_roots, "Unknown TEST scenario root: " + root_name)

                        if current_root is None:
                            require(root_name not in seen_roots, "Scenario re-entry before start")
                            current_root = root_name
                            current = ScenarioAccumulator(current_root, science, oov_stream)
                        elif root_name != current_root:
                            camera, imu, seg = current.finalize()
                            nodes = build_nodes(camera, imu, seg, node_fn)
                            q, predicted = predict_nodes(nodes, loaded_models)
                            require(row_cursor + 299 <= EXPECTED_ROWS, "Prediction row overflow")
                            sl = slice(row_cursor, row_cursor + 299)
                            all_nodes[sl] = nodes
                            all_scenario_index[sl] = len(scenario_order)
                            all_tick[sl] = np.arange(1, 300, dtype=np.int16)
                            all_q[sl] = q
                            all_predicted[sl] = predicted
                            row_cursor += 299
                            seen_roots.add(current_root)
                            scenario_order.append(current_root)

                            require(root_name not in seen_roots, "Noncontiguous/duplicate TEST scenario root: " + root_name)
                            current_root = root_name
                            current = ScenarioAccumulator(current_root, science, oov_stream)

                        current.observe_file()
                        file_count_total += 1
                        cls = member_class(relative)
                        if cls == "ALLOWED":
                            data = read_allowed_member(tf, member, relative)
                            current.consume(relative, data)
                            allowed_payload_reads += 1
                        else:
                            # Critically: no extractfile/read call is made for FORBIDDEN or UNUSED payloads.
                            if cls == "FORBIDDEN":
                                pass

                        now = time.monotonic()
                        if now - last_print >= 30:
                            print("Phase A stream:", tar_members, "members;",
                                  len(scenario_order), "scenarios finalized;",
                                  row_cursor, "prediction rows", flush=True)
                            last_print = now

            require(current is not None, "No TEST scenario encountered")
            camera, imu, seg = current.finalize()
            nodes = build_nodes(camera, imu, seg, node_fn)
            q, predicted = predict_nodes(nodes, loaded_models)
            require(row_cursor + 299 == EXPECTED_ROWS, "Final prediction row count mismatch")
            sl = slice(row_cursor, row_cursor + 299)
            all_nodes[sl] = nodes
            all_scenario_index[sl] = len(scenario_order)
            all_tick[sl] = np.arange(1, 300, dtype=np.int16)
            all_q[sl] = q
            all_predicted[sl] = predicted
            row_cursor += 299
            seen_roots.add(current_root)
            scenario_order.append(current_root)

        require(len(scenario_order) == EXPECTED_SCENARIOS and seen_roots == set(frozen_roots),
                "Final TEST scenario coverage mismatch")
        require(row_cursor == EXPECTED_ROWS, "Final TEST prediction-row coverage mismatch")
        require(file_count_total == EXPECTED_SCENARIOS * EXPECTED_FILES_PER_SCENARIO,
                "Final TEST file-count coverage mismatch")
        require(allowed_payload_reads == EXPECTED_SCENARIOS * 601, "Allowed payload read count mismatch")
        require(forbidden_payload_opens == 0 and label_payload_opens == 0, "Forbidden TEST payload opened")
        require(np.isfinite(all_nodes).all() and np.isfinite(all_q).all(), "Nonfinite final Phase-A arrays")

        run_ids = np.asarray([record["run_id"] for record, _ in loaded_models], dtype="U32")
        thresholds = np.asarray([record["threshold"] for record, _ in loaded_models], dtype=np.float64)
        p_corruption = 1.0 - all_q

        npz_path = output / "predictions_and_nodes.npz"
        with npz_path.open("xb") as f:
            np.savez_compressed(
                f,
                nodes=all_nodes,
                scenario_index=all_scenario_index,
                tick=all_tick,
                run_id=run_ids,
                threshold=thresholds,
                q_normal=all_q,
                p_corruption=p_corruption,
                predicted_corruption=all_predicted,
            )

        write_new_json(output / "scenario_stream_order.json", {
            "schema": "COGNIX-TEST-phase-A-scenario-stream-order-v1",
            "scenario_count": len(scenario_order),
            "order_semantics": "actual scenario order encountered during Phase-A TAR traversal",
            "scenario_roots": scenario_order,
        })

        manifest = {
            "schema": "COGNIX-TEST-phase-A-completion-v1",
            "status": "PHASE_A_LABEL_BLIND_PREDICTIONS_COMPLETE",
            "archive_sha256_remeasured_before_payload_access": measured_sha,
            "archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "execution_bundle_seal": bundle_seal,
            "scenario_count": len(scenario_order),
            "prediction_rows": row_cursor,
            "run_count": 15,
            "run_ids": run_ids.tolist(),
            "thresholds": thresholds.tolist(),
            "nodes_dtype": str(all_nodes.dtype),
            "model_input_dtype": "float32",
            "epistemic_prior_input_dtype": "float64",
            "inference_device": "CPU",
            "allowed_payload_reads": allowed_payload_reads,
            "forbidden_payload_opens": forbidden_payload_opens,
            "test_labels_opened": False,
            "anomaly_front_opened": False,
            "unused_payloads_opened": False,
            "target_columns_written": False,
            "metrics_computed": False,
            "training_executed": False,
            "optimizer_executed": False,
            "recalibration_executed": False,
            "threshold_tuning_executed": False,
            "representation_fitting_executed": False,
            "result_dependent_retry": False,
            "elapsed_seconds": time.time() - start,
        }
        write_new_json(output / "manifest.json", manifest)

        output_seal, payload_files = seal_tree(output)
        require(not final.exists(), "Final Phase-A output unexpectedly exists")
        os.rename(output, final)
        print(json.dumps({
            "status": manifest["status"],
            "output_path": str(final),
            "output_seal": output_seal,
            "sealed_payload_files": payload_files,
            "scenario_count": len(scenario_order),
            "prediction_rows": row_cursor,
            "run_count": 15,
            "test_labels_opened": False,
            "metrics_computed": False,
        }, sort_keys=True))
    except BaseException as exc:
        failure = {
            "schema": "COGNIX-TEST-phase-A-failure-v1",
            "status": "FAILED_PRESERVED_LABEL_BLIND_PHASE_A",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "test_labels_opened": False,
            "metrics_computed": False,
            "retry_requires_review": True,
            "result_dependent_retry": False,
        }
        try:
            if output.is_dir() and not (output / "failure.json").exists():
                write_new_json(output / "failure.json", failure)
        finally:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute-phase-a", action="store_true")
    parser.add_argument("--bundle-seal")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    require(args.self_test != args.execute_phase_a, "Choose exactly one mode")
    repo = Path.cwd().resolve()
    require((repo / "reports" / "carla_train_full_v1").is_dir(), "Run from COGNIX repository root")
    if args.self_test:
        require(args.bundle_seal is None and args.authorization is None, "Self-test takes no execution authorization")
        self_test(repo)
        return
    require(args.bundle_seal is not None, "Execution bundle seal required")
    execute_phase_a(repo, args.bundle_seal, args.authorization)


if __name__ == "__main__":
    main()

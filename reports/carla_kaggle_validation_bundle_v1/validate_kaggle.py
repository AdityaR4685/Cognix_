"""External engineering observer for unchanged frozen trainers.

Real graph arrays are read for integrity only. Training always constructs the
existing 128-row synthetic fixture. Each repeat uses a new Python process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
# Set before importing torch. An inherited incompatible setting is an error.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8; no fallback")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify_files(root=ROOT):
    """Stdlib preflight verifies source bytes before frozen modules are imported."""
    root = Path(root)
    expected = (root / "SHA256SUMS.sha256").read_text().split()[0]
    require(digest(root / "SHA256SUMS") == expected, "bundle hash manifest seal mismatch")
    records = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        h, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), "unsafe manifest path")
        require(path.is_file() and digest(path) == h, "bundle file mismatch: " + relative)
        records[relative] = h
    manifest = read(root / "bundle_manifest.json")
    require(all(records.get(name) == h for name, h in manifest["files"].items()), "manifest records differ")
    require(manifest["frozen_lock"]["trainer_identity"]["bundle_sha256"] ==
            "9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b", "trainer expected identity")
    require(manifest["frozen_lock"]["expected_hashes"]["protocol"] ==
            "68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203", "protocol expected identity")
    require(manifest["frozen_lock"]["expected_hashes"]["graph_scientific"] ==
            "baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237", "graph expected identity")
    return manifest, expected, records


def modules():
    import numpy as np
    import torch
    from torch.nn import functional as F
    from cognix.adapters.carla import graph_fit_export as ge
    from cognix.adapters.carla import graph_training as tr
    from cognix.adapters.carla import graph_training_data as gd
    from cognix.adapters.carla import graph_training_models as gm
    return np, torch, F, ge, tr, gd, gm


def integrity():
    manifest, seal, files = verify_files()
    np, torch, F, ge, tr, gd, gm = modules()
    require(tr.trainer_identity() == manifest["frozen_lock"]["trainer_identity"], "trainer source/bundle mismatch")
    data = gd.load_graph_dataset(ROOT / "graph_artifact", ROOT / "reports/carla_gat_preregistration_v1")
    require(list(data.arrays["node_features"].shape) == manifest["graph_shape"] == [89970, 3, 3], "shape mismatch")
    require(data.schema["node_order"] == manifest["node_order"] == ["Camera", "IMU", "Seg"], "node order")
    require(data.schema["feature_order"] == manifest["feature_order"] == ["prob_normal", "epistemic", "aleatoric"], "feature order")
    require(len(data.train_indices) == manifest["train_rows"] == 71976, "train row count")
    require(len(data.validation_indices) == manifest["validation_rows"] == 17994, "validation row count")
    require(sorted(set(data.scenarios[data.train_indices])) == manifest["train_scenarios"], "train scenarios")
    require(sorted(set(data.scenarios[data.validation_indices])) == manifest["validation_scenarios"], "validation scenarios")
    result = {"passed": True, "expected_hashes": manifest["frozen_lock"]["expected_hashes"],
              "trainer_identity": tr.trainer_identity(), "bundle_SHA256SUMS_sha256": seal,
              "verified_files": files, "graph_shape": manifest["graph_shape"],
              "node_order": manifest["node_order"], "feature_order": manifest["feature_order"],
              "train_scenarios": manifest["train_scenarios"], "validation_scenarios": manifest["validation_scenarios"],
              "graph_arrays_writeable": any(a.flags.writeable for a in data.arrays.values()),
              "real_graph_use": "read-only hash/schema/split verification; never passed to training"}
    del data
    return result


def fixture():
    np, torch, F, ge, tr, gd, gm = modules()
    data = gd.make_smoke_fixture(read(ROOT / "reports/carla_gat_preregistration_v1/split_manifest.json"))
    require(data.data_kind == "synthetic_fixture" and len(data.arrays["row_key"]) == 128,
            "only existing bounded 128-row synthetic fixture permitted")
    return data


def actual_environment(device):
    np, torch, F, ge, tr, gd, gm = modules()
    tr.configure_determinism(101)
    record = tr.environment_record()
    record["CUDA_validation"] = "external frozen-source engineering harness; see sealed CUDA evidence"
    record["autocast_CPU_enabled"] = torch.is_autocast_enabled("cpu")
    record["autocast_CUDA_enabled"] = torch.is_autocast_enabled("cuda")
    require(record["deterministic_algorithms"] and not record["deterministic_warn_only"]
            and record["cuDNN_deterministic"] and not record["cuDNN_benchmark"]
            and not record["matmul_TF32"] and not record["cuDNN_TF32"]
            and not record["autocast_CPU_enabled"] and not record["autocast_CUDA_enabled"]
            and record["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "effective determinism settings failed")
    record["runtime_metadata"] = {"executable": sys.executable, "system": platform.system(),
          "machine": platform.machine(), "release": platform.release(),
          "Kaggle_environment": {key: os.environ.get(key) for key in
              ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DOCKER_IMAGE")},
          "kaggle_working_exists": Path("/kaggle/working").is_dir(),
          "kaggle_input_exists": Path("/kaggle/input").is_dir()}
    record["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    record["GPU_exposure_policy"] = "user authorized GPU 0 isolation from Kaggle T4 x2 allocation; one visible GPU; no distributed execution"
    record["device_properties"] = []
    if device == "cuda":
        require(record["CUDA_available"] and len(record["devices"]) == 1
                and "T4" in record["devices"][0], "actual single T4 required; no CPU/multi-GPU fallback")
        require(platform.system() == "Linux" and Path("/kaggle/working").is_dir()
                and Path("/kaggle/input").is_dir(), "actual Kaggle runtime required")
        props = torch.cuda.get_device_properties(0)
        record["device_properties"] = [{"name": props.name, "major": props.major, "minor": props.minor,
            "total_memory": props.total_memory, "multi_processor_count": props.multi_processor_count,
            "uuid": str(getattr(props, "uuid", "unavailable"))}]
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            record["nvidia_smi"] = {"returncode": p.returncode, "output": p.stdout.strip(), "stderr": p.stderr.strip()}
            record["physical_GPU_count_observed"] = len(p.stdout.strip().splitlines()) if p.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            record["nvidia_smi"] = {"unavailable": repr(exc)}
    return record


def forward_checks(device):
    np, torch, F, ge, tr, gd, gm = modules()
    tr.configure_determinism(101)
    data = fixture()
    raw = np.random.Generator(np.random.PCG64(1)).uniform(0.1, 0.9, (5, 3, 3))
    raw[0, :, 1] = 1e-10
    x = torch.tensor(raw, dtype=torch.float32, device=device)
    e = torch.tensor(raw[:, :, 1], dtype=torch.float64, device=device)
    target = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32, device=device)
    pair = gm.paired_models(101)
    standard = pair["standard_gat"].to(device)
    epi = pair["epistemic_gat"].to(device)
    initial_equal = all(torch.equal(v, epi.state_dict()[key]) for key, v in standard.state_dict().items())
    independent = all(v.data_ptr() != epi.state_dict()[key].data_ptr() for key, v in standard.state_dict().items())
    require(initial_equal and independent, "explicit paired clone/storage integrity")
    result = {"passed": True, "methods": {}, "paired_parameters": {
        "explicit_clone": "frozen paired_models uses copy.deepcopy + load_state_dict",
        "byte_exact": initial_equal, "independent_storage": independent},
        "tolerance_source": "tests/unit/test_carla_graph_trainers.py existing generic/batch equivalence",
        "generic_equivalence_atol": 1e-7, "generic_equivalence_rtol": 1e-6,
        "attention_atol": 1e-6, "attention_rtol": 1e-5,
        "CPU_GPU_tolerance": None, "CPU_GPU_policy": "measure errors only; no new cross-device acceptance tolerance"}
    for method in gd.METHODS:
        model = pair[method].to(device).eval()
        cpu = gm.paired_models(101)[method].eval()
        cpu.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
        with torch.no_grad():
            q, attention = model(x, e, audit_attention=True)
            q_cpu = cpu(x.cpu(), e.cpu())
            require(q.shape == (5,) and torch.isfinite(q).all().item(), "forward shape/finite " + method)
            bce = F.binary_cross_entropy(q, target)
            require(torch.isfinite(bce).item(), "finite BCE " + method)
            item = {"finite_forward": True, "shape": list(q.shape), "finite_BCE": True, "BCE": bce.item(),
                    "CPU_GPU_max_abs_error": (q.cpu() - q_cpu).abs().max().item(),
                    "CPU_GPU_bitwise_equal": torch.equal(q.cpu(), q_cpu), "attention": []}
            for alpha in attention:
                err = (alpha.sum(-1) - 1).abs().max().item()
                require(torch.isfinite(alpha).all().item()
                        and torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-6)
                        and not torch.diagonal(alpha, dim1=-2, dim2=-1).any().item(), "attention invariant")
                item["attention"].append({"finite": True, "max_incoming_sum_error": err, "self_edges_zero": True})
            if method != "nograph":
                reference = []
                for i in range(len(raw)):
                    prior = model.gat.compute_epistemic_weights(dict(zip(ge.NODE_ORDER, raw[i, :, 1])), list(ge.NODE_ORDER))
                    h = x[i]
                    for j, layer in enumerate(model.gat.layers):
                        h, _ = layer(h, model.adjacency, torch.tensor(prior, device=device), final_layer=j == 1)
                    reference.append(torch.sigmoid(h[:, 0]).mean())
                ref = torch.stack(reference)
                require(torch.allclose(q, ref, atol=1e-7, rtol=1e-6), "frozen generic-forward tolerance failed")
                item["generic_forward_max_abs_error"] = (q - ref).abs().max().item()
            result["methods"][method] = item
    zero_x = torch.full((7, 3, 3), 0.4, device=device); zero_x[:, :, 1] = 0
    zero_e = torch.zeros((7, 3), dtype=torch.float64, device=device)
    equivalence = {}
    for training in (False, True):
        values = []
        for model in (standard, epi):
            model.train(training)
            tr.configure_determinism(8)
            values.append(model(zero_x, zero_e).detach())
        require(torch.equal(*values), "E=0 matched-RNG forward must be exact")
        equivalence[str(training)] = {"bitwise_exact": True, "max_abs_error": 0.0}
    result["E_zero_equivalence_training_false_true"] = equivalence
    diagnostic = gm.paired_models(101)["epistemic_gat"].to(device).eval()
    with torch.no_grad():
        for layer in diagnostic.gat.layers:
            layer.a.weight.zero_()  # isolated fixed-logit diagnostic; never a trained model
        diagnostic_x = torch.full((1, 3, 3), 0.2, device=device)
        low = torch.zeros((1, 3), dtype=torch.float64, device=device)
        high = low.clone(); high[:, 0] = 2
        _, a = diagnostic(diagnostic_x, low, audit_attention=True)
        _, b = diagnostic(diagnostic_x, high, audit_attention=True)
        suppress = []
        for first, second in zip(a, b):
            require(torch.all(second[:, 1:, 0] < first[:, 1:, 0]).item()
                    and torch.allclose(second[:, 1:, 0], torch.full((1, 2), 0.25, device=device)), "sender suppression")
            suppress.append({"low_E": first[:, 1:, 0].cpu().tolist(), "high_E": second[:, 1:, 0].cpu().tolist()})
    result["fixed_logit_sender_suppression"] = suppress
    return result


def sync(device):
    if device == "cuda":
        modules()[1].cuda.synchronize()


def train_worker(method, device, output):
    np, torch, F, ge, tr, gd, gm = modules()
    data = fixture()
    observation = {}
    def observer(frame, event, arg):
        if frame.f_globals.get("__name__") == gm.__name__ and frame.f_code.co_name == "forward" and event == "call":
            if frame.f_locals["self"].training and "initial_dropout_rng_sha256" not in observation:
                observation["initial_dropout_rng_sha256"] = tr.recursive_content_hash(tr.rng_state())
                observation["effective_autocast_CUDA"] = torch.is_autocast_enabled("cuda")
        if frame.f_code is tr.train_model.__code__ and event == "return" and "final_state" in frame.f_locals:
            for name in ("initial_state", "final_state"):
                observation[name + "_sha256"] = tr.recursive_content_hash(frame.f_locals[name])
            observation["final_optimizer_sha256"] = tr.recursive_content_hash(frame.f_locals["optimizer"].state_dict())
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device); start = time.perf_counter()
    sys.setprofile(observer)
    try:
        summary = tr.train_model(data, method, 101, output, device=device, audit_attention=True)
    finally:
        sys.setprofile(None)
    sync(device)
    observation["elapsed_seconds"] = time.perf_counter() - start
    require(summary["finite_gradients"] and summary["parameters_updated"] and summary["optimizer_updates"] > 0,
            "tiny training gradients/update failure")
    history = read(output / "history.json")
    require(all(np.isfinite(row["train_BCE"]) and np.isfinite(row["validation_scenario_macro_BCE"]) for row in history), "finite loss trajectories")
    require(all(set(row["validation_per_scenario_BCE"]) == set(data.splits["GRAPH_VALIDATION"]) for row in history), "validation loop")
    model, optimizer, payload = tr.load_checkpoint(summary["selected_checkpoint"], data, device=device)
    q = tr.evaluate(model, data, device=device)
    with np.load(output / "predictions.npz", allow_pickle=False) as z:
        require(np.array_equal(z["q_normal"], q) and np.array_equal(z["p_corrupt"], 1 - q)
                and np.array_equal(z["row_key"], data.arrays["row_key"]), "checkpoint/export roundtrip")
    observation.update({"passed": True, "seed": 101, "fixture_rows": 128, "train_rows": 32,
        "validation_rows": 96, "checkpoint_roundtrip": True, "prediction_export": True,
        "optimizer_configuration": {"optimizer": "Adam", "lr": optimizer.param_groups[0]["lr"],
              "weight_decay": optimizer.param_groups[0]["weight_decay"]},
        "checkpoint_payload_hashes": {p.name: tr.recursive_content_hash(torch.load(p, map_location="cpu", weights_only=False))
                                     for p in sorted(output.glob("checkpoint_epoch_*.pt"))},
        "summary": summary, "interpretation": "engineering fixture only"})
    require(observation.get("effective_autocast_CUDA") is False, "AMP observation")
    if device == "cuda":
        observation["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write(output / "observer.json", observation)
    return observation


def compare_runs(a, b):
    np, torch, F, ge, tr, gd, gm = modules()
    left, right = read(a / "observer.json"), read(b / "observer.json")
    ha, hb = read(a / "history.json"), read(b / "history.json")
    checks = {key: left[key] == right[key] for key in
        ("initial_state_sha256", "initial_dropout_rng_sha256", "final_state_sha256", "final_optimizer_sha256",
         "checkpoint_payload_hashes", "optimizer_configuration")}
    checks["epoch_row_permutations"] = [r["train_row_order"] for r in ha] == [r["train_row_order"] for r in hb]
    checks["batch_boundaries"] = [r["batch_sizes"] for r in ha] == [r["batch_sizes"] for r in hb]
    checks["full_loss_validation_history"] = ha == hb
    checks["selected_epoch"] = left["summary"]["selected_epoch"] == right["summary"]["selected_epoch"]
    checks["best_checkpoint_content"] = left["summary"]["checkpoint_content_sha256"] == right["summary"]["checkpoint_content_sha256"]
    checks["row_keyed_predictions"] = left["summary"]["prediction_content_sha256"] == right["summary"]["prediction_content_sha256"]
    checks["metrics"] = read(a / "metrics.json") == read(b / "metrics.json")
    errors = {}
    for key in ("train_BCE", "validation_scenario_macro_BCE"):
        errors[key] = max((abs(x[key] - y[key]) for x, y in zip(ha, hb)), default=0.0) if len(ha) == len(hb) else None
    with np.load(a / "predictions.npz", allow_pickle=False) as x, np.load(b / "predictions.npz", allow_pickle=False) as y:
        errors["q_normal"] = float(np.max(np.abs(x["q_normal"] - y["q_normal"])))
        checks["prediction_arrays_exact"] = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
    passed = all(checks.values())
    return {"passed": passed, "classification": "1_bitwise_exact" if passed else "3_nondeterministic_unacceptable",
            "checks": checks, "measured_max_abs_errors": errors, "accepted_repeat_tolerance": 0.0,
            "policy": "existing exact-repeat gate; no tolerance relaxation or deterministic fallback"}


def resource_checks(device):
    np, torch, F, ge, tr, gd, gm = modules()
    tr.configure_determinism(101)
    data = fixture()
    # Repeat synthetic inputs to exercise the frozen batch size; never real training.
    rows = np.resize(data.train_indices, 256)
    results = {}
    for method in gd.METHODS:
        model = gm.paired_models(101)[method].to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        if device == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        sync(device); start = time.perf_counter()
        x, e, y = data.batch(rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy(model(x, e), y)
        require(torch.isfinite(loss).item(), "resource fixture loss")
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), "resource fixture gradients")
        optimizer.step()
        sync(device)
        item = {"batch_size": 256, "synthetic_only": True, "one_batch_seconds": time.perf_counter() - start}
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            item.update({"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "total_device_bytes": total, "peak_reserved_fraction": torch.cuda.max_memory_reserved() / total,
                         "batch_256_comfortably_fits": torch.cuda.max_memory_reserved() < total / 2})
        results[method] = item
        del model, optimizer, x, e, y, loss
    return {"passed": True, "methods": results, "scope": "one synthetic 256-row forward/backward/Adam batch per method",
            "scientific_batch_size_changed": False, "real_epochs_benchmarked": False}


def report(output, success, local=False, error=None):
    sections = ["Kaggle Runtime", "Frozen Hash Verification", "CUDA Determinism Settings", "Forward Validation",
        "CUDA Smoke Training", "GPU Reproducibility", "Paired Standard/Epistemic Integrity", "GPU Memory / Runtime",
        "Environment Lock", "Full-Run Commands Prepared", "Core / Artifact Integrity",
        "Ready / Not Ready for Full Paired-Seed Experiment", "Recommended Next Step"]
    messages = ["CPU harness check only; Kaggle/T4 unverified." if local else "See environment_lock.json for actual observed Kaggle/T4 runtime.",
        "See hash_verification.json. Frozen bytes and scientific identity verified before fixtures.",
        "Strict deterministic algorithms; cuDNN deterministic; benchmark/TF32/AMP off; CUBLAS :4096:8. No fallback.",
        "See cuda_smoke_results.json: attention, generic equivalence, E=0 and sender suppression.",
        "Existing 128-row synthetic fixture only; frozen trainer/specification; separate clean processes.",
        "See reproducibility_results.json; every exact-repeat comparison is required.",
        "Explicit cloning, independent storage, matched batches/RNG/optimizer, E=0 trajectories verified.",
        "See resource_results.json; one synthetic batch of 256, no full epochs.",
        "CPU check creates no validated execution lock." if local else "Passed locks are SHA-256 sealed with evidence bindings.",
        "Future launcher prepared; execution requires separate authorization and a passed CUDA lock.",
        "No scientific source/data edits, TEST, full graph training, conformal fit, aggregation, commit or push.",
        "READY (technical validation only; full execution not authorized)." if success and not local else "NOT READY; actual Kaggle CUDA gates remain unverified or failed.",
        "Review the sealed CUDA evidence, then separately authorize the full milestone." if success and not local else
            "Run the validation bundle on an actual Kaggle single-T4 runtime; diagnose any failed gate before proceeding."]
    text = "\n\n".join("## " + section + "\n\n" + message for section, message in zip(sections, messages))
    if error:
        text += "\n\nValidation failure: `" + error + "`\n"
    (output / "report.md").write_text(text + "\n", encoding="utf-8")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    local = args.local_harness_check
    device = "cpu" if local else "cuda"
    try:
        hashes = integrity()
        write(output / "hash_verification.json", hashes)
        if args.integrity_only:
            write(output / "execution_readiness.json", {"ready": False, "status": "INTEGRITY_ONLY_KAGGLE_UNVERIFIED"})
            return
        environment = actual_environment(device)
        write(output / "observed_environment.json", environment)
        evidence = {}
        def worker(kind, destination, method=None):
            command = [sys.executable, str(SELF), "--worker", kind, "--device", device, "--output", str(destination)]
            if method:
                command += ["--method", method]
            with (output / (kind + ("_" + method if method else "") + "_" + destination.name + ".log")).open("x") as stream:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            require(completed.returncode == 0, "worker failed: " + kind + " " + str(method) + "; retained logs; no fallback")
        worker("forward", output / "forward.json")
        for method in ("nograph", "standard_gat", "epistemic_gat"):
            for repeat in ("first", "repeat"):
                worker("train", output / "smoke_runs" / method / repeat, method)
            a, b = (output / "smoke_runs" / method / repeat for repeat in ("first", "repeat"))
            evidence[method] = compare_runs(a, b)
        write(output / "reproducibility_results.json", {"passed": all(r["passed"] for r in evidence.values()),
            "seed": 101, "fresh_process_per_repeat": True, "methods": evidence,
            "interpretation": "engineering only; no multi-seed method comparison"})
        require(all(r["passed"] for r in evidence.values()), "repeat differs; diagnose before proceeding; no relaxed tolerance")
        np, torch, F, ge, tr, gd, gm = modules()
        sa = output / "smoke_runs/standard_gat/first"
        ea = output / "smoke_runs/epistemic_gat/first"
        so, eo = read(sa / "observer.json"), read(ea / "observer.json")
        pair_checks = {key: so[key] == eo[key] for key in
            ("initial_state_sha256", "initial_dropout_rng_sha256", "final_state_sha256", "final_optimizer_sha256", "optimizer_configuration")}
        pair_checks["history_rows_batches_validation_exact"] = read(sa / "history.json") == read(ea / "history.json")
        pair_checks["selected_epoch"] = so["summary"]["selected_epoch"] == eo["summary"]["selected_epoch"]
        for name in so["checkpoint_payload_hashes"]:
            s = torch.load(sa / name, map_location="cpu", weights_only=False)
            e = torch.load(ea / name, map_location="cpu", weights_only=False)
            require(tr.recursive_content_hash(s["model_state"]) == tr.recursive_content_hash(e["model_state"]), "paired checkpoint tensors differ")
        pair_checks["all_saved_checkpoint_model_tensors"] = True
        with np.load(sa / "predictions.npz", allow_pickle=False) as a, np.load(ea / "predictions.npz", allow_pickle=False) as b:
            pair_checks["row_keys_probabilities_exact"] = np.array_equal(a["row_key"], b["row_key"]) and np.array_equal(a["q_normal"], b["q_normal"])
        require(all(pair_checks.values()), "paired E=0 trajectories/RNG differ")
        forward = read(output / "forward.json")
        write(output / "cuda_smoke_results.json", {"passed": True, "device": device, "forward": forward,
            "training": {m: read(output / "smoke_runs" / m / "first/observer.json") for m in evidence},
            "paired_E_zero": pair_checks, "full_graph_training": False, "TEST_access": False,
            "conformal_fit": False, "scientific_aggregation": False})
        worker("resource", output / "resource_results.json")
        # Verify all scientific/package files again after GPU fixtures.
        require(integrity() == hashes, "frozen inputs changed during validation")
        if not local:
            lock = read(ROOT / "reports/carla_graph_trainers_v1/proposed_environment_lock_validated.json")
            lock.update({"status": "ARTIFACT_RUNTIME_AND_CUDA_FIXTURES_VALIDATED",
                "Kaggle_observed_versions": environment, "validated_execution_environment": environment,
                "CUDA_smoke_validation_passed": True, "CUDA_smoke_evidence": evidence,
                "bundle_SHA256SUMS_sha256": hashes["bundle_SHA256SUMS_sha256"],
                "evidence_sha256": {name: digest(output / name) for name in
                    ("hash_verification.json", "cuda_smoke_results.json", "reproducibility_results.json", "resource_results.json")},
                "full_training_authorized_this_milestone": False})
            write(output / "environment_lock.json", lock)
            (output / "environment_lock.json.sha256").write_text(digest(output / "environment_lock.json") + "  environment_lock.json\n", encoding="utf-8")
        write(output / "execution_readiness.json", {"ready": not local,
            "status": "CPU_HARNESS_PASSED_KAGGLE_UNVERIFIED" if local else "ALL_CUDA_GATES_PASSED",
            "full_training_authorized": False, "full_training_executed": False,
            "regression_baseline": "608 passed at frozen milestone"})
        report(output, True, local)
    except Exception as exc:
        write(output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
              "no_fallback": True, "ready": False, "retain_all_evidence": True})
        if not (output / "execution_readiness.json").exists():
            write(output / "execution_readiness.json", {"ready": False, "status": "FAILED", "error": repr(exc)})
        report(output, False, local, repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--local-harness-check", action="store_true", help="CPU engineering verification; never creates a CUDA lock")
    parser.add_argument("--worker", choices=("forward", "train", "resource"), help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("nograph", "standard_gat", "epistemic_gat"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        verify_files()
        actual_environment(args.device)
        if args.worker == "train":
            require(args.method is not None, "worker method required")
            train_worker(args.method, args.device, args.output)
        else:
            value = forward_checks(args.device) if args.worker == "forward" else resource_checks(args.device)
            write(args.output, value)
    else:
        run(args)


if __name__ == "__main__":
    main()

"""Frozen primary graph trainer. This milestone executes synthetic fixtures only."""
from dataclasses import dataclass
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

# Must precede CUDA initialization, including a future Kaggle launch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla.graph_training_data import (
    ARTIFACT_SHA256, MANIFEST_SHA256, SCHEMA_SHA256, SPLIT_SHA256, METHODS, SEEDS, epoch_batches,
)
from cognix.adapters.carla.graph_training_models import paired_models, model_configuration
from cognix.adapters.carla.graph_training_metrics import scenario_macro_bce, dataset_metric_report

TRAINER_VERSION = "carla_graph_trainers_v1"
TRAINING_SPEC = {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
                 "batch_size": 256, "maximum_epochs": 100, "patience": 10, "min_delta": 1e-5,
                 "scheduler": None, "AMP": False, "TF32": False, "gradient_clipping": None}


def configure_determinism(seed):
    ge.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS determinism configuration")
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    # Single-thread CPU fixtures avoid reductions depending on thread scheduling.
    torch.set_num_threads(1)


def trainer_identity():
    directory = Path(__file__).parent
    names = ("graph_training.py", "graph_training_data.py", "graph_training_models.py", "graph_training_metrics.py")
    hashes = {name: ge.sha256_file(directory / name) for name in names}
    return {"version": TRAINER_VERSION, "source_sha256": hashes,
            "bundle_sha256": hashlib.sha256(ge.canonical_json(hashes)).hexdigest()}


def environment_record():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {"Python": platform.python_version(), "PyTorch": torch.__version__, "NumPy": np.__version__,
            "scikit_learn": version("scikit-learn"), "scikit_learn_used": False,
            "platform": platform.platform(), "CUDA_available": torch.cuda.is_available(),
            "torch_CUDA_runtime": torch.version.cuda, "cuDNN_version": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuDNN_deterministic": torch.backends.cudnn.deterministic,
            "cuDNN_benchmark": torch.backends.cudnn.benchmark,
            "matmul_TF32": torch.backends.cuda.matmul.allow_tf32, "cuDNN_TF32": torch.backends.cudnn.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "AMP": False, "DataLoader_workers": 0, "CPU_threads": torch.get_num_threads(),
            "CUDA_validation": "not exercised on local CPU fixtures; deterministic-operation errors must stop future execution"}


def proposed_environment_lock():
    configure_determinism(0)
    return {"status": "PROPOSED_LOCAL_VALIDATED_KAGGLE_UNVERIFIED", "local_observed": environment_record(),
            "expected_hashes": {"graph_scientific": ARTIFACT_SHA256, "graph_schema": SCHEMA_SHA256,
                                "graph_manifest": MANIFEST_SHA256, "protocol": ge.PROTOCOL_SHA256, "split": SPLIT_SHA256},
            "trainer_identity": trainer_identity(), "full_training_authorized_this_milestone": False,
            "Kaggle_observed_versions": None, "required_runtime": {"single_GPU": "T4", "distributed": False,
                "deterministic_algorithms": True, "TF32": False, "cuDNN_deterministic": True,
                "cuDNN_benchmark": False, "AMP": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "future_validation": "Run validate_environment.py on the actual Kaggle runtime before enabling full training; never infer Kaggle package versions from local versions."}


def verify_environment_lock(lock, device):
    actual = environment_record()
    ge.require(lock["expected_hashes"] == proposed_environment_lock()["expected_hashes"], "environment artifact/protocol hash refusal")
    ge.require(lock["trainer_identity"] == trainer_identity(), "trainer code identity changed")
    expected = lock.get("validated_execution_environment")
    ge.require(expected is not None, "full training requires an observed, validated execution environment lock")
    ge.require(lock.get("CUDA_smoke_validation_passed") is True,
               "full execution requires future deterministic CUDA smoke validation")
    for key in ("Python", "PyTorch", "NumPy", "torch_CUDA_runtime", "cuDNN_version"):
        ge.require(actual[key] == expected[key], "execution environment version mismatch: " + key)
    ge.require(str(device).startswith("cuda") and actual["CUDA_available"]
               and len(actual["devices"]) == 1 and "T4" in actual["devices"][0], "frozen single-T4 execution")
    ge.require(actual["deterministic_algorithms"] and not actual["matmul_TF32"]
               and not actual["cuDNN_TF32"] and actual["cuDNN_deterministic"] and not actual["cuDNN_benchmark"], "deterministic execution settings")


@dataclass
class EarlyStopping:
    best_loss: float = float("inf")
    best_epoch: int | None = None
    tracked_best: float = float("inf")
    bad_epochs: int = 0

    def observe(self, loss, epoch):
        ge.require(np.isfinite(loss) and loss >= 0 and epoch >= 1, "invalid validation macro BCE")
        save = loss < self.best_loss
        if save:
            self.best_loss, self.best_epoch = float(loss), int(epoch)
        if loss < self.tracked_best - TRAINING_SPEC["min_delta"]:
            self.tracked_best, self.bad_epochs = float(loss), 0
        else:
            self.bad_epochs += 1
        return save, self.bad_epochs >= TRAINING_SPEC["patience"]


def recursive_content_hash(value):
    """Hash scientific checkpoint state, independent of torch's ZIP serialization."""
    h = hashlib.sha256()
    def walk(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            h.update(ge.canonical_json({"tensor": str(v.dtype), "shape": list(a.shape)})); h.update(a.tobytes())
        elif isinstance(v, np.ndarray):
            h.update(ge.canonical_json({"array": str(v.dtype), "shape": list(v.shape)})); h.update(v.tobytes())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                walk(k); walk(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(ge.canonical_json({"sequence": type(v).__name__, "length": len(v)}))
            for item in v:
                walk(item)
        else:
            h.update(ge.canonical_json({"type": type(v).__name__, "value": v}))
    walk(value)
    return h.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_CPU": torch.get_rng_state(),
            "torch_CUDA": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_CPU"])
    if state["torch_CUDA"]:
        ge.require(torch.cuda.is_available(), "CUDA RNG continuation needs CUDA")
        torch.cuda.set_rng_state_all(state["torch_CUDA"])


def save_checkpoint(path, model, optimizer, method, seed, epoch, stopper, dataset, history):
    payload = {"metadata": {"trainer_identity": trainer_identity(), "method": method, "seed": seed, "epoch": epoch,
        "best_validation_scenario_macro_BCE": stopper.best_loss, "selected_epoch": stopper.best_epoch,
        "graph_artifact_scientific_sha256": ARTIFACT_SHA256, "protocol_sha256": ge.PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256, "model_configuration": model_configuration(method),
        "training_specification": TRAINING_SPEC, "data_kind": dataset.data_kind,
        "training_data_content_sha256": ge.scientific_content_hash(dataset.arrays, dataset.schema)},
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()), "rng_state": rng_state(),
        "early_stopping_state": vars(stopper).copy(), "history": copy.deepcopy(history)}
    # Optimizer tensors use portable CPU storage, while preserving exact values.
    for state in payload["optimizer_state"]["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.detach().cpu().clone()
    content = recursive_content_hash(payload)
    payload["checkpoint_content_sha256"] = content
    with Path(path).open("xb") as f:
        torch.save(payload, f)
    return content


def load_checkpoint(path, dataset, device="cpu", continuation=False):
    # Checkpoints are local trainer-created files; tensor/scientific content is verified.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    content = payload.pop("checkpoint_content_sha256")
    ge.require(recursive_content_hash(payload) == content, "checkpoint content hash refusal")
    m = payload["metadata"]
    ge.require(m["graph_artifact_scientific_sha256"] == ARTIFACT_SHA256 and m["protocol_sha256"] == ge.PROTOCOL_SHA256
               and m["split_sha256"] == SPLIT_SHA256 and m["trainer_identity"] == trainer_identity(), "checkpoint dependency refusal")
    ge.require(m["data_kind"] == dataset.data_kind and m["training_data_content_sha256"] == ge.scientific_content_hash(dataset.arrays, dataset.schema), "checkpoint training-data identity")
    ge.require(m["model_configuration"] == model_configuration(m["method"]) and m["training_specification"] == TRAINING_SPEC, "checkpoint specification refusal")
    model = paired_models(m["seed"])[m["method"]].to(device)
    model.load_state_dict(payload["model_state"]); model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer.load_state_dict(payload["optimizer_state"])
    if continuation:
        restore_rng_state(payload["rng_state"])
    payload["checkpoint_content_sha256"] = content
    return model, optimizer, payload


def evaluate(model, dataset, indices=None, device="cpu", audit_attention=False):
    indices = np.arange(len(dataset.arrays["target_normal"])) if indices is None else np.asarray(indices)
    old_training = model.training
    model.eval()
    values, attention = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            rows = indices[start:start + 256]
            x, e, _ = dataset.batch(rows, device)
            if audit_attention:
                q, layers = model(x, e, audit_attention=True)
                if layers:
                    attention.append(torch.stack(layers, dim=1).cpu().numpy())
            else:
                q = model(x, e)
            values.append(q.cpu().numpy())
    model.train(old_training)
    result = np.concatenate(values).astype(np.float64)
    ge.require(np.isfinite(result).all() and np.all((result >= 0) & (result <= 1)), "invalid graph probability")
    if audit_attention:
        return result, np.concatenate(attention) if attention else None
    return result


def prediction_arrays(dataset, q, method, seed, epoch, checkpoint_content_sha256):
    a, n = dataset.arrays, len(q)
    ge.require(n == len(a["row_key"]), "row-keyed prediction alignment")
    recipes = np.array([ge.RECIPES[int(k)] if k >= 0 else "" for k in a["recipe_index"]])
    return {"row_key": a["row_key"], "pair_id": a["pair_id"], "scenario": dataset.scenarios,
        "tick": a["tick"], "target_normal": a["target_normal"], "recipe": recipes, "split": a["split"],
        "method": np.full(n, method), "seed": np.full(n, seed, dtype=np.int64), "epoch": np.full(n, epoch, dtype=np.int32),
        "checkpoint_content_sha256": np.full(n, checkpoint_content_sha256), "q_normal": np.asarray(q, dtype=np.float64),
        "p_corrupt": 1 - np.asarray(q, dtype=np.float64)}


def train_model(dataset, method, seed, output, *, execute_full=False, environment_lock=None,
                device="cpu", audit_attention=False, resume_checkpoint=None):
    ge.require(method in METHODS and seed in SEEDS, "method/seed must be preregistered")
    # Full data cannot reach even model construction without explicit future gates.
    if dataset.data_kind != "synthetic_fixture":
        ge.require(execute_full, "full graph training is protected; this milestone permits smoke fixtures only")
        ge.require(environment_lock is not None, "full run requires validated environment lock")
    else:
        ge.require(len(dataset.arrays["target_normal"]) <= 512,
                   "smoke training refuses oversized or relabeled full datasets")
    configure_determinism(seed)
    if dataset.data_kind != "synthetic_fixture":
        verify_environment_lock(environment_lock, device)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    history, stopper, selected_path, selected_hash = [], EarlyStopping(), None, None
    model = paired_models(seed)[method].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # Initializer consumption cannot change paired training dropout RNG state.
    configure_determinism(seed)
    start_epoch = 1
    if resume_checkpoint is not None:
        model, optimizer, saved = load_checkpoint(resume_checkpoint, dataset, device, continuation=True)
        ge.require(saved["metadata"]["method"] == method and saved["metadata"]["seed"] == seed, "continuation method/seed")
        stopper = EarlyStopping(**saved["early_stopping_state"])
        history = copy.deepcopy(saved["history"]); start_epoch = saved["metadata"]["epoch"] + 1
        selected_path, selected_hash = str(Path(resume_checkpoint).resolve()), saved["checkpoint_content_sha256"]
        ge.require(stopper.bad_epochs < TRAINING_SPEC["patience"], "cannot continue a completed patience stop")
    finite_gradients, updates = True, 0
    try:
        for epoch in range(start_epoch, 101):
            model.train(); loss_sum, count = 0.0, 0
            row_order = []
            for rows in epoch_batches(dataset, seed, epoch - 1):
                ge.require(np.all(dataset.arrays["split"][rows] == 0), "validation/CAL optimization rejected")
                x, e, target = dataset.batch(rows, device)
                optimizer.zero_grad(set_to_none=True)
                q = model(x, e)
                loss = F.binary_cross_entropy(q, target)
                ge.require(torch.isfinite(loss).item(), "nonfinite training loss")
                loss.backward()
                finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
                finite_gradients &= finite
                ge.require(finite, "nonfinite gradient; no fallback or clipping")
                optimizer.step(); updates += 1
                loss_sum += loss.item() * len(rows); count += len(rows); row_order.extend(rows.tolist())
            vi = dataset.validation_indices
            q_val = evaluate(model, dataset, vi, device)
            val, per_scenario = scenario_macro_bce(q_val, dataset.arrays["target_normal"][vi], dataset.scenarios[vi], dataset.splits["GRAPH_VALIDATION"])
            save, stop = stopper.observe(val, epoch)
            history.append({"epoch": epoch, "train_BCE": loss_sum / count, "validation_scenario_macro_BCE": val,
                            "validation_per_scenario_BCE": per_scenario,
                            "train_row_order": row_order if dataset.data_kind == "synthetic_fixture" else None,
                            "train_row_order_sha256": hashlib.sha256(np.asarray(row_order, dtype="<i8").tobytes()).hexdigest(),
                            "batch_sizes": [len(b) for b in epoch_batches(dataset, seed, epoch - 1)],
                            "patience_bad_epochs": stopper.bad_epochs, "patience_tracked_best": stopper.tracked_best})
            if save:
                path = output / f"checkpoint_epoch_{epoch:03d}.pt"
                selected_hash = save_checkpoint(path, model, optimizer, method, seed, epoch, stopper, dataset, history)
                selected_path = str(path.resolve())
            if stop:
                break
        ge.require(selected_path is not None, "no checkpoint selected")
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        changed = any(not torch.equal(initial_state[k], final_state[k]) for k in initial_state)
        selected, _, payload = load_checkpoint(selected_path, dataset, device)
        q = evaluate(selected, dataset, device=device)
        predictions = prediction_arrays(dataset, q, method, seed, stopper.best_epoch, selected_hash)
        with (output / "predictions.npz").open("xb") as f:
            np.savez_compressed(f, **predictions)
        metrics = dataset_metric_report(dataset, q)
        ge.save_json(output / "metrics.json", metrics)
        ge.save_json(output / "history.json", history)
        if audit_attention and method != "nograph":
            q_audit, attention = evaluate(selected, dataset, dataset.validation_indices, device, audit_attention=True)
            ge.require(np.array_equal(q_audit, q[dataset.validation_indices]), "attention audit changed predictions")
            with (output / "validation_attention.npz").open("xb") as f:
                np.savez_compressed(f, row_key=dataset.arrays["row_key"][dataset.validation_indices],
                                    attention=attention, attention_semantics=np.array("pre_dropout,incoming_receiver_by_sender"))
        summary = {"status": "complete", "method": method, "seed": seed, "data_kind": dataset.data_kind,
            "interpretation": "engineering smoke evidence only" if dataset.data_kind == "synthetic_fixture" else "internal graph development",
            "epochs_executed": len(history), "selected_epoch": stopper.best_epoch,
            "best_validation_scenario_macro_BCE": stopper.best_loss, "selected_checkpoint": selected_path,
            "checkpoint_content_sha256": selected_hash, "finite_gradients": finite_gradients,
            "optimizer_updates": updates, "parameters_updated": changed,
            "initial_model_content_sha256": recursive_content_hash(initial_state),
            "training_data_content_sha256": payload["metadata"]["training_data_content_sha256"],
            "graph_artifact_scientific_sha256": ARTIFACT_SHA256, "protocol_sha256": ge.PROTOCOL_SHA256,
            "trainer_identity": trainer_identity(), "model_configuration": model_configuration(method),
            "training_specification": TRAINING_SPEC, "environment": environment_record(),
            "prediction_content_sha256": recursive_content_hash(predictions)}
        ge.save_json(output / "run_summary.json", summary)
        return summary
    except Exception as exc:
        ge.save_json(output / "failure.json", {"method": method, "seed": seed, "error": repr(exc),
                     "history": history, "retain_run": True, "replace_seed": False})
        raise

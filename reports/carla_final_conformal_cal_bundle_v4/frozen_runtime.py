"""Inference-only loading of byte-bound frozen sources and selected states."""
import importlib
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def forbid(*args, **kwargs):
    raise RuntimeError("FROZEN_INFERENCE_ONLY: fitting/training/tuning forbidden")


def bootstrap():
    # Namespace loading avoids the generic orchestration and all trainer imports.
    for name in ("cognix", "cognix.core", "cognix.adapters", "cognix.adapters.carla", "cognix.graph"):
        if name in sys.modules:
            raise RuntimeError("Run in a fresh process: cognix namespace already loaded")
        module = types.ModuleType(name)
        module.__path__ = [str(HERE / "frozen_sources" / name.replace(".", "/"))]
        sys.modules[name] = module
    normality = importlib.import_module("cognix.adapters.carla.normality")
    agents = importlib.import_module("cognix.adapters.carla.real_agents")
    graph = importlib.import_module("cognix.graph.epistemic_gat")
    for module in (normality, agents, graph):
        for obj in vars(module).values():
            if isinstance(obj, type) and obj.__module__ == module.__name__:
                for name in ("fit", "fit_calibrator"):
                    if hasattr(obj, name):
                        setattr(obj, name, forbid)
    return agents


def recursive_content_hash(value):
    # Exact frozen scientific checkpoint hashing algorithm, no trainer import.
    import hashlib
    import numpy as np
    import torch
    def canonical(v):
        return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    h = hashlib.sha256()
    def walk(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            h.update(canonical({"tensor": str(v.dtype), "shape": list(a.shape)})); h.update(a.tobytes())
        elif isinstance(v, np.ndarray):
            h.update(canonical({"array": str(v.dtype), "shape": list(v.shape)})); h.update(v.tobytes())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                walk(k); walk(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(canonical({"sequence": type(v).__name__, "length": len(v)}))
            for item in v:
                walk(item)
        else:
            h.update(canonical({"type": type(v).__name__, "value": v}))
    walk(value)
    return h.hexdigest()


def load(device="cpu", audit_original=False):
    import numpy as np
    import torch
    bootstrap()
    from cognix.adapters.carla.graph_training_models import SharedNodeMLP, BatchedGAT
    from cognix.adapters.carla.graph_fit_export import restore_agents
    registry = json.loads((HERE / "model_registry.json").read_text())
    models = {}
    for row in registry["scorers"]:
        path = HERE / row["bundle_state"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = SharedNodeMLP() if row["method"] == "nograph" else BatchedGAT(row["method"] == "epistemic_gat")
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        # Disable train(True) throughout the model tree after constructor/restore.
        for child in model.modules():
            original_train = child.train
            def train(mode=True, original=original_train):
                if mode:
                    return forbid()
                return original(False)
            child.train = train
        models[row["scorer_id"]] = model
    with np.load(HERE / "upstream/fitted_oneclass_parameters.npz", allow_pickle=False) as z:
        parameters = dict(z)
    handoff = json.loads((HERE / "upstream/manifest.json").read_text())
    agents = restore_agents(parameters, handoff)
    return models, agents

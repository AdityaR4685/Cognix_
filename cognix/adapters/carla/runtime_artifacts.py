"""Read-only restoration of the identified frozen demonstration artifacts.

No fit, checkpoint selection, label access, threshold import or calibration calls.
The collective conformal stage is intentionally unavailable for this fusion chain.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cognix.core.interfaces import GraphRefinement
from cognix.core.types import GraphResult
from cognix.runtime.replay import SourceError

ORDER = ("Camera", "IMU", "Seg")
DIMS = (18, 10, 29)
COLUMNS = ("prob_normal", "epistemic", "aleatoric")
CALIBRATION_REASON = "No collective conformal artifact is bound to frozen upstream + node refinement + runtime epistemic-weighted fusion"
BINDINGS_SHA = "d772113b690a082819634b3beb410eff04d6bc51088ad6e548af20d027761b54"
HEALTH_SHA = "8cbb80b10464bd38eb6ef567857d071c11313a8bda937c1ded4e27f20bfca320"


def require(condition, message):
    if not condition:
        raise SourceError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def authenticated(path, expected):
    require(Path(path).is_file(), f"Required artifact missing: {path}")
    require(sha256(path) == expected, f"Artifact hash mismatch: {path}")


def bound_definitions(path, names, namespace):
    """Execute only named, authenticated definitions; never bundle entry points."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    require({n.name for n in nodes} == set(names), "Missing frozen definitions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def source_bindings(repo: Path):
    report = repo / "reports/carla_train_full_v1"
    path = report / "graph_test_label_blind_execution_bundle_v1/phase_A_input_bindings.json"
    authenticated(path, BINDINGS_SHA)
    bindings = json.loads(path.read_text())
    for source in bindings["sources"]:
        authenticated(repo / source["path"], source["sha256"])
    return report, bindings


def load_sanitizer(repo: Path):
    report, _ = source_bindings(repo)
    namespace = bound_definitions(report / "gate1_execution_bundle_v8/segmentation_oov.py",
                                 {"sanitize"}, {"np": np, "require": require})
    return namespace["sanitize"]


def verify_unit(path, expected_seal):
    authenticated(path / "SHA256SUMS", expected_seal)
    for line in (path / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(None, 1)
        target = (path / name).resolve()
        require(target.is_relative_to(path.resolve()), "Artifact manifest escapes unit")
        authenticated(target, expected)


def validate_checkpoint_tensor_schema(model: nn.Module, model_state: dict, checkpoint_path=None) -> nn.Module:
    """Validate and load state_dict into model with strict schema enforcement.

    Fails closed with SourceError if tensor names or dimensions mismatch,
    or if nonfinite values are detected.
    """
    path_label = str(checkpoint_path) if checkpoint_path is not None else "<checkpoint>"
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise SourceError(f"Checkpoint tensor schema mismatch: {path_label}: {exc}") from exc
    require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()), "Nonfinite graph state")
    return model


class FrozenNodeRefinement(GraphRefinement):
    """Expose the sealed node logits to the ordinary generic pipeline."""
    def __init__(self, model, strategy):
        self.model, self.strategy = model.eval(), strategy

    def is_trained(self):
        return True

    def forward(self, node_features, adjacency, epistemic_uncertainties, agent_order):
        require(tuple(agent_order) == ORDER, "Frozen graph agent order mismatch")
        require(np.asarray(node_features).shape == (3, 3), "Frozen graph requires three feature columns")
        require(np.isfinite(node_features).all(), "Nonfinite graph inputs")
        expected_adjacency = np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32)
        require(np.array_equal(adjacency, expected_adjacency), "Frozen graph adjacency mismatch")
        nodes = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0)
        original_e = torch.tensor([[epistemic_uncertainties[k] for k in ORDER]], dtype=torch.float64)
        attention = []
        with torch.no_grad():
            if self.strategy == "NoGraph":
                h = self.model.last(self.model.dropout(F.elu(self.model.first(nodes))))
            else:
                prior = self.model.prior(original_e)
                h = nodes
                for i, layer in enumerate(self.model.layers):
                    h, alpha = self.model.layer_forward(layer, h, prior, i == 1)
                    attention.append(alpha[0].cpu().numpy())
        return GraphResult(node_outputs=h[0].cpu().numpy(), attention=attention,
                           metadata={"strategy": self.strategy, "method": self.strategy,
                                     "communication": self.strategy != "NoGraph"})


def load_runtime_artifacts(repo: Path, strategy="NoGraph", seed=101, require_calibration=False):
    require(strategy in ("NoGraph", "StandardGAT", "EpistemicGAT"), "Unknown graph strategy")
    require(seed in (101, 202, 303, 404, 505), "Seed is not a frozen demonstration configuration")
    if require_calibration:
        raise SourceError(CALIBRATION_REASON)
    report, bindings = source_bindings(repo)
    verify_unit(report / "graph_test_checkpoint_freeze_bundle_v1", bindings["checkpoint_freeze_seal"])
    health_path = report / "gate2_train_health_v3/FINAL.json"
    authenticated(health_path, HEALTH_SHA)
    health = json.loads(health_path.read_text())
    from .normality import MahalanobisNormality, BootstrapNormalityEnsemble, EnsemblePredictiveCalibrator
    from .real_agents import RealCameraAgent, RealIMUAgent, RealSegAgent
    agents = []
    for name, dim, cls in zip(ORDER, DIMS, (RealCameraAgent, RealIMUAgent, RealSegAgent)):
        fit = report / "gate2_train_health_v3/units" / ("fit-" + name)
        audit_root = fit.parent / ("audit-" + name)
        for path in (fit, audit_root):
            verify_unit(path, health["committed_unit_seals"][path.name])
        audit = json.loads((audit_root / "audit.json").read_text())
        require(audit["status"] == "VALID" and audit["optimization_info"]["finite_optimum"] is True,
                f"Upstream calibration is not valid: {name}")
        ensemble = BootstrapNormalityEnsemble(n_members=5, seed=42)
        for i in range(5):
            with np.load(fit / f"member_{i}.npz", allow_pickle=False) as z:
                require(set(z.files) == {"mean", "precision", "d2_scale", "shrinkage"}, "Member schema mismatch")
                mean, precision = z["mean"].copy(), z["precision"].copy()
                scale, shrinkage = float(z["d2_scale"]), float(z["shrinkage"])
            require(mean.shape == (dim,) and precision.shape == (dim, dim), "Member dimension mismatch")
            require(mean.dtype == precision.dtype == np.float64 and np.isfinite(mean).all() and
                    np.isfinite(precision).all() and np.isfinite(scale) and scale > 0 and shrinkage == 0.1,
                    "Invalid frozen member state")
            member = MahalanobisNormality(shrinkage=shrinkage)
            member._mean, member._prec, member._d2_scale = mean, precision, scale
            ensemble._members.append(member)
        with np.load(audit_root / "audit_arrays.npz", allow_pickle=False) as z:
            params = z["calibrator_parameters_a_b_score_mean_std"].copy()
        require(params.shape == (4,) and np.isfinite(params).all() and params[3] > 0 and
                np.array_equal(params, np.array(audit["calibrator_parameters"])), "Upstream calibrator mismatch")
        calibrator = EnsemblePredictiveCalibrator()
        calibrator.a, calibrator.b, calibrator._s_mean, calibrator._s_std = map(float, params)
        calibrator._fitted, calibrator._validity = True, audit["optimization_info"].copy()
        for field in ("nll_initial", "nll_final", "n_iter", "converged"):
            setattr(calibrator, "_" + field, audit["optimization_info"][field])
        calibrator.require_valid(diagnostic=False)
        agent = cls()
        agent._ensemble, agent._calibrator = ensemble, calibrator
        agents.append(agent)

    selected = next(c for c in bindings["checkpoints"] if c["method"] == strategy and c["seed"] == seed)
    checkpoint_path = report / selected["path"]
    authenticated(checkpoint_path, selected["sha256"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(state["method"] == strategy and state["seed"] == seed and state["epoch"] == selected["epoch"],
            "Checkpoint identity mismatch")
    freeze = json.loads((report / "graph_test_checkpoint_freeze_bundle_v1/freeze_manifest.json").read_text())
    require(state["graph_seal"] == freeze["graph_data_seal"] and
            state["scientific_content_sha256"] == freeze["scientific_content_sha256"], "Checkpoint lineage mismatch")
    definitions = bound_definitions(report / "kaggle_graph_training_bundle_v1/kaggle_models.py",
                                   {"GATLayer", "NoGraph", "BatchedGAT"},
                                   {"torch": torch, "nn": nn, "F": F, "require": require,
                                    "ADJACENCY": [[0, 1, 1], [1, 0, 1], [1, 1, 0]]})
    model = definitions["NoGraph"]() if strategy == "NoGraph" else definitions["BatchedGAT"](strategy == "EpistemicGAT")
    validate_checkpoint_tensor_schema(model, state["model_state"], checkpoint_path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {"artifact_id": selected["run_id"], "checkpoint_sha256": selected["sha256"],
                  "checkpoint_epoch": selected["epoch"], "seed": seed, "graph_strategy": strategy,
                  "artifact_bindings_sha256": BINDINGS_SHA, "upstream_health_sha256": HEALTH_SHA,
                  "source_hashes": {s["path"]: s["sha256"] for s in bindings["sources"]},
                  "configuration_role": "demonstration configuration, no TEST performance selection",
                  "node_feature_order": list(COLUMNS), "agent_order": list(ORDER),
                  "fusion": "runtime epistemic-weighted fusion; different from benchmark mean pooling",
                  "calibration_unavailable_reason": CALIBRATION_REASON, "full_calibrated_chain": False,
                  "uq_method": "bootstrap one-class ensemble entropy decomposition",
                  "probability_label": "Constructed-task normality probability",
                  "prediction_semantics": "P(target=normal) under TRAIN-derived clean-vs-pseudo task",
                  "decision_semantics": "Framework demonstration policy; not a physical driving command"}
    return agents, FrozenNodeRefinement(model, strategy), provenance

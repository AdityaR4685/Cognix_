"""Interface independence, label isolation and real-format replay, without fitting."""
import ast
from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from cognix import DecisionEngine
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.core.types import PredictionResult, UncertaintyResult, GraphResult
from cognix.runtime.replay import ObservationEnvelope, ReplayController, SourceError
from cognix.runtime.bridge import RuntimeBridge
from cognix.adapters.carla.replay_source import ArchiveConfig, CarlAnomalyReplaySource, SyntheticReplaySource
from cognix.adapters.carla.runtime_artifacts import load_runtime_artifacts, load_sanitizer, validate_checkpoint_tensor_schema, ORDER
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from dashboard.payload import build_payload

ROOT = Path(__file__).resolve().parents[2]


class ToySource:
    """An industrial diagnostic fixture, independent of driving/sensor layouts."""
    info = {"mode": "DATASET", "domain": "toy_diagnostics", "source_id": "ToyReplaySource"}
    def __init__(self, names):
        self.names = names
    def sequences(self):
        return ["case-17", "case-18"]
    def sample_count(self, sequence_id):
        return 3
    def read(self, sequence_id, index):
        return ObservationEnvelope("ToyReplaySource", "toy_diagnostics", sequence_id, str(index),
                                   {name: np.array([0.7 + index / 100]) for name in self.names},
                                   metadata={"display_only_label": "must never enter inference"}, provenance=self.info)


class ToyAgent:
    def __init__(self, name):
        self.agent_id = name
        self.calls = []
    def metadata(self):
        return {"agent_id": self.agent_id}
    def predict(self, inputs):
        self.calls.append(dict(inputs))
        assert "display_only_label" not in inputs and "ground_truth" not in inputs
        return PredictionResult(float(inputs[self.agent_id][0]))
    def estimate_uncertainty(self, inputs):
        return UncertaintyResult(float(inputs[self.agent_id][0]), 0.02, 0.1, 0.12)


class IndependentRefinement:
    def is_trained(self):
        return True
    def forward(self, node_features, adjacency, epistemic_uncertainties, agent_order):
        q = node_features[:, :1]
        return GraphResult(np.log(q / (1 - q)), [], {"method": "independent_node_refinement"})


def toy_cycle(names):
    source = ToySource(names)
    controller = ReplayController(source)
    controller.select("case-17")
    agents = [ToyAgent(n) for n in names]
    engine = DecisionEngine(graph=IndependentRefinement(), belief=EpistemicWeightedFusion(),
                            attribution=EpistemicShapley(), mode="research")
    bridge = RuntimeBridge(engine, agents, {"graph_strategy": "IndependentRefinement"})
    return bridge.run(controller.step()), agents


@pytest.mark.parametrize("names", [("AgentA", "AgentB", "AgentC"), tuple(f"diagnostic {i}" for i in range(6)),
                                  ("μ-check", "name <x>", "a/b")])
def test_generic_source_reaches_engine_and_dynamic_payload(names):
    cycle, agents = toy_cycle(names)
    payload = build_payload(cycle)
    assert payload["active_agents"] == list(names)
    assert payload["mode"] == "DATASET" and payload["domain"] == "toy_diagnostics"
    assert payload["decision"] == cycle.result.decision.value
    assert payload["module_status"] == cycle.result.metadata["module_status"]
    assert payload["graph_executed"] and not payload["graph_communication_executed"]
    assert payload["communication_statistics"] is None
    assert not payload["calibration_available"] and payload["conformal_set"] is None
    assert len(payload["shapley"]) == len(names)
    assert all(len(agent.calls) == 1 for agent in agents)
    json.dumps(payload, allow_nan=False)


def test_envelope_is_immutable_and_has_only_generic_fields():
    original = np.array([1.0])
    observation = ObservationEnvelope("s", "d", "case", "sample", {"AgentA": original}, metadata={"a": [1]})
    original[0] = 9
    assert observation.agent_inputs["AgentA"][0] == 1
    with pytest.raises(FrozenInstanceError):
        observation.sample_id = "changed"
    with pytest.raises(TypeError):
        observation.agent_inputs["x"] = 1
    with pytest.raises(ValueError):
        observation.agent_inputs["AgentA"].flags.writeable = True
    assert not {"rgb", "imu", "segmentation", "carla"} & set(observation.__dataclass_fields__)


def test_post_inference_label_cannot_change_result_or_trigger_agents():
    cycle, agents = toy_cycle(("AgentA", "AgentB", "AgentC"))
    original = cycle.result.to_dict()
    def annotation():
        assert all(len(agent.calls) == 1 for agent in agents)
        return "normal"
    a = build_payload(cycle, annotation=annotation)
    b = build_payload(cycle, annotation="anomaly")
    for key in ("decision", "confidence", "shapley", "agents", "module_status", "conformal_set"):
        assert a[key] == b[key]
    assert original == cycle.result.to_dict()
    assert a["ground_truth"]["value"] != b["ground_truth"]["value"]
    assert a["provenance"]["label_visible_to_inference"] is False
    assert all(set(call) == {"AgentA", "AgentB", "AgentC"} for agent in agents for call in agent.calls)


def test_controller_navigation_and_rate():
    c = ReplayController(ToySource(("AgentA",)))
    with pytest.raises(SourceError, match="Select"):
        c.step()
    with pytest.raises(SourceError, match="Unknown"):
        c.select("missing")
    c.select("case-17")
    assert c.step().sample_id == "0"
    assert c.step().sample_id == "1"
    assert c.step(-1).sample_id == "0"
    c.playing = True
    c.step()
    c.step()
    assert c.playing is False
    c.restart()
    assert c.index == -1
    assert c.step().sample_id == "0"
    c.set_rate(2)
    for rate in (0, 11, float("nan")):
        with pytest.raises(SourceError):
            c.set_rate(rate)


def test_schema_mismatch_fails_before_agents():
    agent = ToyAgent("AgentA")
    bridge = RuntimeBridge(DecisionEngine(), [agent])
    with pytest.raises(SourceError, match="schema/order"):
        bridge.run(ObservationEnvelope("s", "d", "c", "1", {"wrong": np.array([0.5])}))
    assert not agent.calls


@pytest.fixture
def tiny_archive(tmp_path):
    scenario = "test/normal/Town01/scenario-1"
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        def add(name, data):
            member = tarfile.TarInfo(scenario + "/" + name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        for tick in range(3):
            for folder, suffix in (("rgb-front", "jpg"), ("segmentation-front", "png")):
                out = io.BytesIO()
                array = np.full((16, 16, 3), 128 if folder == "rgb-front" else 30, dtype=np.uint8)
                Image.fromarray(array).save(out, format="JPEG" if suffix == "jpg" else "PNG")
                add(f"{folder}/{tick:06d}.{suffix}", out.getvalue())
        # Poison payloads must never be opened/decoded as part of replay.
        add("anomaly-front/000001.png", b"forbidden")
        add("anomaly-observation.feather", b"forbidden")
        stream = io.BytesIO()
        pd.DataFrame({"acceleration_x": [0., 1., 2.], "acceleration_y": [0., 0., 0.],
                      "acceleration_z": [0., 0., 0.]}).to_feather(stream)
        add("imu.feather", stream.getvalue())
    return path, scenario


@pytest.mark.research_assets
def test_real_format_archive_is_bounded_and_label_blind(tiny_archive, monkeypatch):
    path, scenario = tiny_archive
    opened = []
    original = tarfile.TarFile.extractfile
    def guarded(tar, member):
        assert "anomaly" not in member.name.split("/")[-2:] and "anomaly-front" not in member.name
        assert "anomaly-observation" not in member.name
        opened.append(member.name)
        return original(tar, member)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", guarded)
    source = CarlAnomalyReplaySource(ArchiveConfig(path, max_samples=2), {scenario: 2}, load_sanitizer(ROOT))
    observation = source.read(scenario, 0)
    assert tuple(observation.agent_inputs) == ORDER
    assert [v.shape for v in observation.agent_inputs.values()] == [(18,), (10,), (29,)]
    assert observation.agent_inputs["Seg"][22] == 1  # Reused frozen OOV transform.
    assert len(opened) == 5
    assert source.read(scenario, 1).metadata["cache_hit"] is True
    assert observation.timestamp is None
    assert observation.provenance["label_payloads_opened"] is False


def test_missing_archive_is_explicit(tmp_path):
    with pytest.raises(SourceError, match="archive not configured/found"):
        CarlAnomalyReplaySource.configured(ROOT, archive=tmp_path / "missing.tar.gz")


def test_missing_required_sensor_fails_closed(tiny_archive, monkeypatch):
    path, scenario = tiny_archive
    # Asking for another missing tick never zero-fills inputs or enters a model.
    # A synthetic no-op sanitizer is injected: the IMU causal-window check fires
    # before any segmentation member is decoded, so the sanitizer is never called.
    # This isolates the fail-closed contract without requiring the local research bundle.
    def _synthetic_sanitizer(img):
        arr = np.asarray(img)[..., 0].copy()
        arr[arr > 28] = 22
        return arr, {}
    source = CarlAnomalyReplaySource(ArchiveConfig(path, max_samples=3), {scenario: 3}, _synthetic_sanitizer)
    with pytest.raises(SourceError, match="causal replay windows"):
        source.read(scenario, 0)
    assert source._cache == []


def test_synthetic_source_still_has_six_agents():
    source = SyntheticReplaySource(CarlAnomalyDataset(n_frames_per_anomaly=1, seed=42))
    observation = source.read("NORMAL", 0)
    assert len(observation.agent_inputs) == 6
    assert observation.provenance["mode"] == "SYNTHETIC"
    assert "label" not in observation.agent_inputs


@pytest.mark.research_assets
@pytest.mark.parametrize("method", ["NoGraph", "StandardGAT", "EpistemicGAT"])
def test_frozen_graph_restores_exact_node_semantics(method):
    agents, graph, provenance = load_runtime_artifacts(ROOT, method, 101)
    assert tuple(a.agent_id for a in agents) == ORDER
    assert provenance["node_feature_order"] == ["prob_normal", "epistemic", "aleatoric"]
    nodes = np.array([[.31, .018, .12], [.6, .002, .3], [.9, .14, .23]], dtype=np.float32)
    epi = {k: float(v) for k, v in zip(ORDER, [.018, .002, .14])}
    adjacency = np.ones((3,3), dtype=np.float32) - np.eye(3, dtype=np.float32)
    result = graph.forward(nodes, adjacency, epi, ORDER)
    with torch.no_grad():
        expected = graph.model(torch.tensor(nodes)[None], torch.tensor([list(epi.values())], dtype=torch.float64))
    actual = torch.sigmoid(torch.tensor(result.node_outputs[:, 0])).mean()
    assert torch.equal(actual, expected.squeeze())
    assert bool(result.attention) == (method != "NoGraph")
    assert not any(p.requires_grad for p in graph.model.parameters())
    with pytest.raises(SourceError, match="order"):
        graph.forward(nodes, adjacency, epi, ORDER[::-1])
    with pytest.raises(SourceError, match="columns"):
        graph.forward(nodes[:, :2], adjacency, epi, ORDER)


def test_full_calibration_requirement_and_unfrozen_config_fail_closed():
    for config, message in [({"require_calibration": True}, "conformal"),
                            ({"seed": 999}, "Seed"), ({"strategy": "best"}, "strategy")]:
        with pytest.raises(SourceError, match=message):
            load_runtime_artifacts(ROOT, **config)


def test_checkpoint_schema_validation_valid():
    """Portable unit test: compatible tensor names and shapes load cleanly."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    state = model.state_dict()
    modified_state = {k: v + 0.5 for k, v in state.items()}
    loaded_model = validate_checkpoint_tensor_schema(model, modified_state, "test_checkpoint.pt")
    assert loaded_model is model
    for k, v in model.state_dict().items():
        assert torch.allclose(v, modified_state[k])


def test_checkpoint_schema_validation_dimension_mismatch():
    """Portable unit test: mismatched tensor dimension fails closed with SourceError."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    state = model.state_dict()
    corrupted_state = {k: v.clone() for k, v in state.items()}
    corrupted_state["0.weight"] = torch.zeros((12, 4))
    with pytest.raises(SourceError, match="tensor schema mismatch"):
        validate_checkpoint_tensor_schema(model, corrupted_state, "bad_dim_checkpoint.pt")


def test_checkpoint_schema_validation_missing_key():
    """Portable unit test: missing tensor key fails closed with SourceError."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    state = model.state_dict()
    missing_state = {k: v.clone() for k, v in state.items() if k != "0.bias"}
    with pytest.raises(SourceError, match="tensor schema mismatch"):
        validate_checkpoint_tensor_schema(model, missing_state, "missing_key_checkpoint.pt")


def test_checkpoint_schema_validation_unexpected_key():
    """Portable unit test: unexpected tensor key fails closed with SourceError."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    state = model.state_dict()
    extra_state = {k: v.clone() for k, v in state.items()}
    extra_state["unexpected.weight"] = torch.zeros((4, 4))
    with pytest.raises(SourceError, match="tensor schema mismatch"):
        validate_checkpoint_tensor_schema(model, extra_state, "unexpected_key_checkpoint.pt")


def test_checkpoint_schema_validation_nonfinite_fails_closed():
    """Portable unit test: nonfinite tensor values fail closed with SourceError."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    state = model.state_dict()
    nan_state = {k: v.clone() for k, v in state.items()}
    nan_state["0.weight"][0, 0] = float("nan")
    with pytest.raises(SourceError, match="Nonfinite graph state"):
        validate_checkpoint_tensor_schema(model, nan_state, "nan_checkpoint.pt")


@pytest.mark.research_assets
def test_wrong_checkpoint_tensor_dimensions_fail_closed(monkeypatch):
    """Full integration test: verifies load_runtime_artifacts fails closed on bad state.

    This test exercises the full production loader path including source bindings,
    cryptographic seals, and historical training bundle verification. For portable
    unit verification of the checkpoint schema validation contract on clean CI runners,
    see test_checkpoint_schema_validation_*.
    """
    original = torch.load
    def incompatible(*args, **kwargs):
        state = original(*args, **kwargs)
        state["model_state"]["first.weight"] = torch.zeros((12, 4))
        return state
    monkeypatch.setattr(torch, "load", incompatible)
    with pytest.raises(SourceError, match="tensor schema mismatch"):
        load_runtime_artifacts(ROOT, "NoGraph", 101)


def test_core_has_no_domain_import_dependencies():
    for folder in ("engine", "core", "belief", "uncertainty", "graph", "calibration", "attribution", "runtime"):
        for path in (ROOT / "cognix" / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                assert not any(n == "carla" or n.startswith(("carla.", "cognix.adapters.carla", "simulation.carla")) for n in names), path


def test_mode_api_failure_never_exposes_synthetic_as_dataset():
    from dashboard import app as module
    from fastapi import HTTPException
    try:
        with pytest.raises(HTTPException) as exc:
            module.set_mode(module.ModeRequest(mode="DATASET", archive="missing.tar.gz"))
        assert exc.value.status_code == 422
        state = module.source_status()
        assert state["mode"] == "DATASET" and state["source"] is None and state["error"]
        assert module.history == []
        with pytest.raises(HTTPException):
            module.api_latest()
        with pytest.raises(HTTPException):
            module.set_mode(module.ModeRequest(mode="LIVE"))
    finally:
        module.set_mode(module.ModeRequest(mode="SYNTHETIC"))


def test_api_navigation_rate_and_missing_scenario():
    from dashboard import app as module
    from fastapi import HTTPException
    module.pause()
    with pytest.raises(HTTPException):
        module.set_scenario(module.ScenarioRequest(scenario="missing"))
    module.set_scenario(module.ScenarioRequest(scenario="NORMAL"))
    first = module.step(module.StepRequest(delta=1))
    second = module.step(module.StepRequest(delta=1))
    previous = module.step(module.StepRequest(delta=-1))
    assert (first["tick"], second["tick"], previous["tick"]) == (1,2,1)
    assert module.rate(module.RateRequest(rate=2))["source"]["rate"] == 2
    assert module.play()["source"]["playing"]
    assert not module.pause()["source"]["playing"]
    assert module.restart()["source"]["index"] == -1

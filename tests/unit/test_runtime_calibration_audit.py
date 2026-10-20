"""Numeric core regression and conformal contract checks; no model/calibrator fitting."""
import ast
import json
from pathlib import Path
import subprocess
import random

import numpy as np
import pytest

from cognix import DecisionEngine
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor
from cognix.core.types import PredictionResult, UncertaintyResult, GraphResult
from cognix.engine.pipeline import CognixPipeline
from cognix.runtime.bridge import RuntimeBridge
from cognix.runtime.replay import ObservationEnvelope, SourceError
from cognix.adapters.carla.runtime_artifacts import load_runtime_artifacts
from dashboard.payload import build_payload

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("diagnostic A", "μ-monitor", "Agent <C>")


class Agent:
    def __init__(self, name, metadata_error=False, uncertainty_error=False):
        self.agent_id = name
        self.metadata_error, self.uncertainty_error = metadata_error, uncertainty_error

    def predict(self, inputs):
        assert set(inputs) == set(NAMES)
        return PredictionResult(float(inputs[self.agent_id][0]))

    def estimate_uncertainty(self, inputs):
        if self.uncertainty_error:
            raise ValueError("fixture uncertainty failed")
        index = NAMES.index(self.agent_id)
        return UncertaintyResult(float(inputs[self.agent_id][0]), .01 * (index + 1), .1, .1 + .01 * (index + 1))

    def metadata(self):
        if self.metadata_error:
            raise ValueError("optional metadata failed")
        return {"uq_method": "reference_uncertainty"}


class Graph:
    def __init__(self, attention=True, metadata=None, failure=False):
        self.attention, self.metadata, self.failure = attention, metadata, failure

    def forward(self, nodes, adjacency, epistemic, order):
        if self.failure:
            raise ValueError("fixture graph failed")
        q = nodes[:, :1]
        matrix = adjacency / adjacency.sum(axis=1, keepdims=True)
        return GraphResult(np.log(q / (1 - q)), [matrix] if self.attention else [], self.metadata)


class RecordingConformal(ConformalPredictor):
    def __init__(self):
        super().__init__()
        # Reference state only. No fitting or reading any research calibration score.
        self.cal_scores = np.linspace(.15, .95, 39, dtype=np.float64)
        self.n_cal = len(self.cal_scores)
        self.received = None

    def predict(self, outputs, alpha=.05):
        self.received = outputs.copy()
        return super().predict(outputs, alpha)


def inputs():
    return {name: np.array([.71 + i * .04]) for i, name in enumerate(NAMES)}


def pipeline(cls=CognixPipeline, mode="research", **overrides):
    return cls(mode=mode, graph=overrides.get("graph", Graph(metadata={"method": "reference_graph"})),
               belief_fuser=overrides.get("belief", EpistemicWeightedFusion()),
               calibrator=overrides.get("calibrator", RecordingConformal()),
               attribution=overrides.get("attribution", EpistemicShapley()))


def run(p, agents=None):
    # Compare the same Monte Carlo Shapley permutations without changing runtime RNG policy.
    prior = random.getstate()
    random.seed(20261011)
    try:
        return p.run(agents or [Agent(n) for n in NAMES], inputs(), {}, {n: 1. for n in NAMES})
    finally:
        random.setstate(prior)


def scientific_outputs(result):
    fields = ("confidence", "calibrated_confidence", "epistemic_uncertainty", "aleatoric_uncertainty",
              "total_uncertainty", "risk_level", "decision", "escalation_required", "abstained",
              "requested_information", "agent_predictions", "agent_trust_weights",
              "agent_contributions", "calibration_metrics")
    return {key: getattr(result, key) for key in fields}


@pytest.fixture(scope="module")
def base_pipeline():
    source = subprocess.check_output(["git", "show", "HEAD:cognix/engine/pipeline.py"], cwd=ROOT, text=True)
    namespace = {"__name__": "cognix.engine._base_pipeline_audit"}
    exec(compile(source, "base_pipeline_audit", "exec"), namespace)
    return namespace["CognixPipeline"]


@pytest.mark.parametrize("mode", ["research", "production"])
def test_valid_non_domain_outputs_equal_base_pipeline(base_pipeline, mode):
    old, new = run(pipeline(base_pipeline, mode)), run(pipeline(mode=mode))
    assert scientific_outputs(old) == scientific_outputs(new)
    assert all(v["executed"] for v in new.metadata["module_status"].values())
    assert new.metadata["module_status"]["uq"]["method"] == "reference_uncertainty"
    assert new.metadata["module_status"]["gnn"]["method"] == "reference_graph"


@pytest.mark.parametrize("mode", ["research", "production"])
def test_optional_agent_metadata_error_preserves_successful_uq(base_pipeline, mode):
    agents = [Agent(n, metadata_error=True) for n in NAMES]
    old, new = run(pipeline(base_pipeline, mode), agents), run(pipeline(mode=mode), agents)
    assert scientific_outputs(old) == scientific_outputs(new)
    assert new.metadata["module_status"]["uq"]["failure_reason"] is None


@pytest.mark.parametrize("metadata", [None, [], {"method": None}, {"method": ""}])
def test_optional_graph_metadata_cannot_invalidate_numeric_stage(metadata):
    result = run(pipeline(graph=Graph(metadata=metadata)))
    status = result.metadata["module_status"]["gnn"]
    assert status["executed"] and status["failure_reason"] is None
    assert status["method"] == "epistemic_gat"


def test_node_refinement_without_attention_reports_no_communication():
    result = run(pipeline(graph=Graph(attention=False)))
    assert result.metadata["module_status"]["gnn"]["executed"] is True
    assert result.communication_statistics is None


@pytest.mark.parametrize("stage", ["graph", "belief", "calibrator", "attribution", "uncertainty"])
def test_configured_failure_still_fails_closed_in_research(base_pipeline, stage):
    class Broken:
        cal_scores = np.array([.3])
        def fuse(self, **kwargs):
            raise ValueError("fixture fusion failed")
        def predict(self, *args, **kwargs):
            raise ValueError("fixture calibration failed")
        def compute(self, *args, **kwargs):
            raise ValueError("fixture attribution failed")
    overrides = {stage: Graph(failure=True) if stage == "graph" else Broken()} if stage != "uncertainty" else {}
    agents = [Agent(n, uncertainty_error=stage == "uncertainty") for n in NAMES]
    for cls in (base_pipeline, CognixPipeline):
        with pytest.raises(RuntimeError, match="RESEARCH MODE"):
            run(pipeline(cls, **overrides), agents)


@pytest.mark.parametrize("stage", ["graph", "belief", "calibrator", "attribution", "uncertainty"])
def test_production_fallback_outputs_equal_base(base_pipeline, stage):
    class Broken:
        cal_scores = np.array([.3])
        def fuse(self, **kwargs):
            raise ValueError("fixture fusion failed")
        def predict(self, *args, **kwargs):
            raise ValueError("fixture calibration failed")
        def compute(self, *args, **kwargs):
            raise ValueError("fixture attribution failed")
    overrides = {stage: Graph(failure=True) if stage == "graph" else Broken()} if stage != "uncertainty" else {}
    agents = [Agent(n, uncertainty_error=stage == "uncertainty") for n in NAMES]
    old = run(pipeline(base_pipeline, "production", **overrides), agents)
    new = run(pipeline(mode="production", **overrides), agents)
    assert scientific_outputs(old) == scientific_outputs(new)
    for name, status in old.metadata["module_status"].items():
        assert new.metadata["module_status"][name]["executed"] == status["executed"]
        assert new.metadata["module_status"][name]["failure_reason"] == status["failure_reason"]


def test_offline_score_construction_matches_actual_runtime_calibrator_input():
    cp = RecordingConformal()
    result = run(pipeline(calibrator=cp))
    nodes = np.array([[inputs()[n][0], .01 * (i + 1), .1] for i, n in enumerate(NAMES)], dtype=np.float32)
    logits = Graph().forward(nodes, np.ones((3, 3)) - np.eye(3), {}, NAMES).node_outputs
    refined = np.clip(1 / (1 + np.exp(-logits[:, 0])), 1e-7, 1 - 1e-7)
    expected = EpistemicWeightedFusion().fuse(dict(zip(NAMES, map(float, refined))),
                                            {n: .01 * (i + 1) for i, n in enumerate(NAMES)}, {n: 1. for n in NAMES}).probability
    expected = float(np.clip(expected, 0, 1))
    np.testing.assert_array_equal(cp.received, np.array([[1 - expected, expected]], dtype=np.float64))
    assert result.calibrated_confidence == result.confidence == expected
    assert result.metadata["module_status"]["calibration"]["executed"] is True
    assert np.isfinite(cp.cal_scores).all() and cp.n_cal == 39
    rank = int(np.ceil((cp.n_cal + 1) * .95))
    assert result.calibration_metrics["quantile"] == cp.cal_scores[rank - 1]


def test_post_inference_labels_cannot_change_conformal_or_other_outputs():
    engine = DecisionEngine(graph=Graph(), belief=EpistemicWeightedFusion(), calibrator=RecordingConformal(),
                            attribution=EpistemicShapley(), mode="research")
    observation = ObservationEnvelope("reference", "industrial_diagnostic", "case-1", "1", inputs())
    cycle = RuntimeBridge(engine, [Agent(n) for n in NAMES]).run(observation)
    before = cycle.result.to_dict()
    first = build_payload(cycle, annotation="normal")
    second = build_payload(cycle, annotation="anomaly")
    assert cycle.result.to_dict() == before
    for key in ("confidence", "agents", "conformal_set", "calibration_available", "risk_level", "decision", "shapley", "module_status"):
        assert first[key] == second[key]
    assert first["calibration_available"] is True and first["conformal_set"] is not None
    assert first["provenance"]["label_visible_to_inference"] is False


def test_existing_conformal_algorithm_can_return_empty_sets_without_fabrication():
    cp = ConformalPredictor()
    cp.cal_scores, cp.n_cal = np.full(39, .01), 39
    assert cp.predict(np.array([[.5, .5]]))[0].prediction_set == []


@pytest.mark.parametrize("strategy", ["NoGraph", "StandardGAT", "EpistemicGAT"])
def test_unavailable_real_collective_calibration_fails_closed_for_every_strategy(strategy):
    with pytest.raises(SourceError, match="No collective conformal artifact"):
        load_runtime_artifacts(ROOT, strategy, 101, require_calibration=True)


@pytest.mark.parametrize("name", ["Camera", "IMU", "Seg"])
def test_frozen_cal_partition_was_consumed_by_upstream_probability_mapping(name):
    report = ROOT / "reports/carla_train_full_v1"
    partition = json.loads((report / "partition_freeze_v1/partition.json").read_text())
    audit = json.loads((report / f"gate2_train_health_v3/units/audit-{name}/audit.json").read_text())
    assert audit["calibrator_runtime_class"].startswith("EnsemblePredictiveCalibrator")
    assert set(audit["CAL_parent_scenarios"]) == set(partition["CAL_NORMAL"])
    assert audit["sample_counts"]["CAL_clean"] == 25 * 2999
    assert set(audit["FIT_binding"]["FIT_NORMAL"]) | set(audit["CAL_parent_scenarios"]) == set(partition["sorted_canonical_ids"])


def test_scientific_helper_methods_are_ast_identical_to_base():
    old = ast.parse(subprocess.check_output(["git", "show", "HEAD:cognix/engine/pipeline.py"], cwd=ROOT, text=True))
    new = ast.parse((ROOT / "cognix/engine/pipeline.py").read_text())
    def methods(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CognixPipeline")
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
    a, b = methods(old), methods(new)
    for name in ("_compute_trust_weights", "_aggregate_confidence", "_aggregate_uncertainty",
                 "_run_belief_fusion", "_run_attribution", "_assess_risk", "_make_decision"):
        assert a[name] == b[name], name

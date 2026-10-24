"""
Domain-neutral pipeline contract and boundary tests for the COGNIX framework.

Validates the generic COGNIX pipeline orchestration contracts:
- Normal end-to-end orchestration with domain-neutral agents
- Input and output boundary contracts (types, empty inputs, non-finite values)
- Failure propagation and fault isolation (research vs. production mode)
- Repeatability and state isolation
- Serialization and result integrity
- Cross-domain boundary test (independent of CARLA or autonomous driving)
- DecisionEngine configuration and shorthand module resolution
- Bare integer and floating-point scalar confidence semantics
- Non-finite prediction and confidence validation (NaN, +inf, -inf)
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from cognix.core.interfaces import AgentInterface, BeliefFuser, UncertaintyEstimator
from cognix.core.types import PredictionResult, UncertaintyResult
from cognix.decision.escalation import EscalationEngine
from cognix.engine.decision_engine import DecisionEngine
from cognix.engine.pipeline import CognixPipeline, PRODUCTION_MODE, RESEARCH_MODE
from cognix.engine.result import DecisionOutcome, DecisionResult, RiskLevel
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor


# ── Minimal Fake Agents for Domain-Neutral Testing ─────────────────────────────

class DomainNeutralAgent(AgentInterface):
    """Minimal domain-neutral agent implementing canonical AgentInterface."""

    def __init__(self, agent_id: str, pred_val: float, epistemic: float = 0.1, aleatoric: float = 0.05):
        self._id = agent_id
        self._pred_val = pred_val
        self._epistemic = epistemic
        self._aleatoric = aleatoric

    @property
    def agent_id(self) -> str:
        return self._id

    def predict(self, observation: any) -> PredictionResult:
        return PredictionResult(value=self._pred_val, confidence=self._pred_val)

    def estimate_uncertainty(self, observation: any) -> UncertaintyResult:
        return UncertaintyResult(
            prediction=self._pred_val,
            epistemic=self._epistemic,
            aleatoric=self._aleatoric,
            total=self._epistemic + self._aleatoric,
        )

    def metadata(self) -> dict:
        return {"agent_id": self._id, "domain": "generic_sensor"}


class DictUQAgent:
    """Agent that returns uncertainty as a standard dictionary."""

    def __init__(self, agent_id: str, pred_val: float, epistemic: float = 0.15):
        self.agent_id = agent_id
        self.pred_val = pred_val
        self.epistemic = epistemic

    def predict(self, observation: any) -> float:
        return self.pred_val

    def estimate_uncertainty(self, observation: any) -> dict:
        return {
            "epistemic": self.epistemic,
            "aleatoric": 0.05,
            "total": self.epistemic + 0.05,
        }


class ValueOnlyPredictionAgent:
    """Agent that returns PredictionResult with value only (confidence=None)."""

    def __init__(self, agent_id: str, val: float):
        self.agent_id = agent_id
        self.val = val

    def predict(self, observation: any) -> PredictionResult:
        return PredictionResult(value=self.val, confidence=None)

    def estimate_uncertainty(self, observation: any) -> UncertaintyResult:
        return UncertaintyResult(
            prediction=float(self.val),
            epistemic=0.1,
            aleatoric=0.05,
            total=0.15,
        )


class FailingAgent:
    """Agent designed to fail in predict or estimate_uncertainty on demand."""

    def __init__(self, agent_id: str, fail_in_predict: bool = False, fail_in_uq: bool = False):
        self.agent_id = agent_id
        self.fail_in_predict = fail_in_predict
        self.fail_in_uq = fail_in_uq

    def predict(self, observation: any):
        if self.fail_in_predict:
            raise RuntimeError(f"Simulated predict failure for {self.agent_id}")
        return 0.75

    def estimate_uncertainty(self, observation: any):
        if self.fail_in_uq:
            raise RuntimeError(f"Simulated UQ failure for {self.agent_id}")
        return UncertaintyResult(prediction=0.75, epistemic=0.1, aleatoric=0.05, total=0.15)


class BareIntegerAgent:
    """Agent returning bare integer predictions (e.g. class labels 0 or 1) with no UQ."""

    def __init__(self, agent_id: str, label: int):
        self.agent_id = agent_id
        self.label = label

    def predict(self, observation: any) -> int:
        return self.label


class NonFiniteAgent:
    """Agent designed to return non-finite predictions or confidences."""

    def __init__(self, agent_id: str, mode: str = "nan_bare"):
        self.agent_id = agent_id
        self.mode = mode

    def predict(self, observation: any):
        if self.mode == "nan_bare":
            return float("nan")
        elif self.mode == "pos_inf_bare":
            return float("inf")
        elif self.mode == "neg_inf_bare":
            return float("-inf")
        elif self.mode == "nan_conf_in_result":
            return PredictionResult(value=0.8, confidence=float("nan"))
        elif self.mode == "pos_inf_conf_in_result":
            return PredictionResult(value=0.8, confidence=float("inf"))
        elif self.mode == "nan_value_in_result":
            return PredictionResult(value=float("nan"), confidence=0.8)
        elif self.mode == "nan_in_dict":
            return {"confidence": float("nan"), "value": 0.8}
        raise ValueError(f"Unknown test mode: {self.mode}")


# ── Section A: Normal End-to-End Orchestration ────────────────────────────────

def test_generic_pipeline_e2e_normal_orchestration():
    """Verify normal end-to-end execution of generic pipeline with domain-neutral agents."""
    agents = [
        DomainNeutralAgent("sensor_alpha", pred_val=0.85, epistemic=0.08, aleatoric=0.04),
        DomainNeutralAgent("sensor_beta", pred_val=0.80, epistemic=0.12, aleatoric=0.06),
    ]

    pipeline = CognixPipeline(
        belief_fuser=EpistemicWeightedFusion(),
        mode=PRODUCTION_MODE,
    )

    result = pipeline.run(
        agents=agents,
        input_data={"feature_a": 1.2, "feature_b": -0.5},
        context={"session_id": "test_e2e_session_001"},
    )

    assert isinstance(result, DecisionResult)
    assert result.decision in (DecisionOutcome.ACT, DecisionOutcome.WAIT, DecisionOutcome.REQUEST_INFORMATION, DecisionOutcome.ESCALATE)
    assert 0.0 <= result.confidence <= 1.0
    assert result.epistemic_uncertainty is not None
    assert result.epistemic_uncertainty >= 0.0
    assert result.session_id == "test_e2e_session_001"
    assert "module_status" in result.metadata
    assert result.metadata["module_status"]["uq"]["executed"] is True
    assert result.metadata["module_status"]["belief"]["executed"] is True
    assert len(result.agent_predictions) == 2
    assert "sensor_alpha" in result.agent_predictions
    assert "sensor_beta" in result.agent_predictions


# ── Section B: Boundary Contracts (Types, Shapes, Values) ──────────────────────

def test_extract_confidence_scalar_semantics():
    """
    Verify _extract_confidence distinguishes bare integers (baseline fallback 0.5)
    from bare floats and explicit confidence values.
    """
    # 1. Bare Python integers retain the baseline fallback of 0.5 (not coerced to 0.0 or 1.0)
    assert CognixPipeline._extract_confidence(0) == 0.5
    assert CognixPipeline._extract_confidence(1) == 0.5
    assert CognixPipeline._extract_confidence(2) == 0.5

    # 2. NumPy integer scalars retain the baseline fallback of 0.5
    assert CognixPipeline._extract_confidence(np.int64(0)) == 0.5
    assert CognixPipeline._extract_confidence(np.int64(1)) == 0.5
    assert CognixPipeline._extract_confidence(np.int32(2)) == 0.5

    # 3. Bare Python float confidence is preserved and clipped to [0, 1]
    assert CognixPipeline._extract_confidence(0.88) == pytest.approx(0.88, abs=1e-5)

    # 4. NumPy floating-point scalars are supported consistently with Python floats
    assert CognixPipeline._extract_confidence(np.float32(0.88)) == pytest.approx(0.88, abs=1e-4)
    assert CognixPipeline._extract_confidence(np.float64(0.88)) == pytest.approx(0.88, abs=1e-5)

    # 5. Finite out-of-range scalars are clipped to [0, 1]
    assert CognixPipeline._extract_confidence(1.5) == 1.0
    assert CognixPipeline._extract_confidence(-0.5) == 0.0

    # 6. Explicitly supplied confidence in PredictionResult is respected according to contract
    p_with_conf = PredictionResult(value="class_1", confidence=0.85)
    assert CognixPipeline._extract_confidence(p_with_conf) == pytest.approx(0.85, abs=1e-4)

    # Explicit numeric confidence in PredictionResult (e.g. 1 or 0) is accepted
    assert CognixPipeline._extract_confidence(PredictionResult(value="obs", confidence=1)) == 1.0
    assert CognixPipeline._extract_confidence(PredictionResult(value="obs", confidence=0)) == 0.0

    # 7. PredictionResult(value=..., confidence=None) preserves uncalibrated fallback 0.5
    assert CognixPipeline._extract_confidence(PredictionResult(value=0.92, confidence=None)) == 0.5
    assert CognixPipeline._extract_confidence(PredictionResult(value=1, confidence=None)) == 0.5
    assert CognixPipeline._extract_confidence(PredictionResult(value=0, confidence=None)) == 0.5

    # 8. Documented dict confidence keys
    assert CognixPipeline._extract_confidence({"confidence": 0.75}) == pytest.approx(0.75, abs=1e-4)
    assert CognixPipeline._extract_confidence({"prob": 0.82}) == pytest.approx(0.82, abs=1e-4)
    assert CognixPipeline._extract_confidence({"confidence": 1}) == 1.0
    assert CognixPipeline._extract_confidence({"confidence": 0}) == 0.0


def test_extract_confidence_rejects_non_finite_values():
    """Verify _extract_confidence raises ValueError on NaN, +inf, and -inf."""
    # Bare non-finite floats
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence(float("nan"))
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence(np.nan)
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence(float("inf"))
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence(float("-inf"))

    # Non-finite in PredictionResult.confidence
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence(PredictionResult(value=0.8, confidence=float("nan")))
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence(PredictionResult(value=0.8, confidence=float("inf")))
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence(PredictionResult(value=0.8, confidence=float("-inf")))

    # Non-finite in dictionary confidence
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence({"confidence": float("nan")})
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_confidence({"confidence": float("inf")})


def test_extract_prediction_probability_boundary_and_finiteness():
    """Verify _extract_prediction_probability accepts valid values and rejects non-finite data."""
    # Valid extractions
    assert CognixPipeline._extract_prediction_probability(0) == 0.0
    assert CognixPipeline._extract_prediction_probability(1) == 1.0
    assert CognixPipeline._extract_prediction_probability(0.72) == pytest.approx(0.72, abs=1e-4)
    assert CognixPipeline._extract_prediction_probability(np.float32(0.72)) == pytest.approx(0.72, abs=1e-4)
    assert CognixPipeline._extract_prediction_probability(PredictionResult(value=0.65)) == pytest.approx(0.65, abs=1e-4)

    # Rejection of non-finite values
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_prediction_probability(float("nan"))
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_prediction_probability(float("inf"))
    with pytest.raises(ValueError, match="(?i)non-finite"):
        CognixPipeline._extract_prediction_probability(PredictionResult(value=float("nan")))


def test_dict_based_uncertainty_extraction():
    """Verify agents returning uncertainty dictionaries are correctly supported."""
    agent = DictUQAgent("dict_agent", pred_val=0.78, epistemic=0.14)
    pipeline = CognixPipeline(belief_fuser=EpistemicWeightedFusion(), mode=RESEARCH_MODE)

    result = pipeline.run(agents=[agent], input_data=[1, 2, 3], context={})
    assert result.agent_predictions["dict_agent"] == pytest.approx(0.78, abs=1e-4)
    assert result.epistemic_uncertainty == pytest.approx(0.14, abs=1e-4)


def test_empty_agent_list_handling():
    """Verify empty agent list handling in production vs research mode."""
    # Production: returns safe emergency ABSTAIN with consistent metadata
    prod_pipeline = CognixPipeline(mode=PRODUCTION_MODE)
    res_prod = prod_pipeline.run(agents=[], input_data={}, context={})
    assert res_prod.decision == DecisionOutcome.ABSTAIN
    assert res_prod.abstained is True
    assert res_prod.escalation_required is True
    assert "module_status" in res_prod.metadata
    assert res_prod.metadata["mode"] == PRODUCTION_MODE

    # Research: must fail loudly
    res_pipeline = CognixPipeline(mode=RESEARCH_MODE)
    with pytest.raises(RuntimeError, match="(?i)no.*prediction"):
        res_pipeline.run(agents=[], input_data={}, context={})


def test_non_finite_uncertainty_rejection_in_research_mode():
    """Verify non-finite uncertainty values (NaN, inf) are rejected in research mode."""
    class NaNUncertaintyAgent:
        agent_id = "nan_agent"
        def predict(self, x): return 0.8
        def estimate_uncertainty(self, x):
            return {"epistemic": float("nan"), "aleatoric": 0.05, "total": float("nan")}

    pipeline = CognixPipeline(mode=RESEARCH_MODE)
    with pytest.raises((ValueError, RuntimeError), match="(?i)non-finite|nan"):
        pipeline.run(agents=[NaNUncertaintyAgent()], input_data={}, context={})


# ── Section C: Bare Integer Predictions & Class Label Semantics ────────────────

def test_integer_class_labels_do_not_create_artificial_certainty():
    """
    Prove that bare integer class labels 0 and 1 retain the baseline fallback (0.5),
    and do NOT create artificial zero uncertainty or near-infinite trust weights in production.
    """
    agent_0 = BareIntegerAgent("agent_label_0", label=0)
    agent_1 = BareIntegerAgent("agent_label_1", label=1)

    pipeline = CognixPipeline(belief_fuser=EpistemicWeightedFusion(), mode=PRODUCTION_MODE)

    # 1. Agent returning label 0 does NOT produce epistemic uncertainty of 0.5 (which would derive from conf=0.0)
    # Baseline fallback: conf=0.5 -> epistemic = max(0, 1-0.5)*0.5 = 0.25
    res_0 = pipeline.run(agents=[agent_0], input_data=[1.0], context={})
    assert res_0.epistemic_uncertainty == pytest.approx(0.25, abs=1e-4)

    # 2. Agent returning label 1 does NOT produce epistemic uncertainty of 0.0 (which would derive from conf=1.0)
    # Baseline fallback: conf=0.5 -> epistemic = max(0, 1-0.5)*0.5 = 0.25
    res_1 = pipeline.run(agents=[agent_1], input_data=[1.0], context={})
    assert res_1.epistemic_uncertainty == pytest.approx(0.25, abs=1e-4)

    # 3. Ensemble of agent_0 and agent_1: trust weights must be equal, not skewed by 10^8
    res_ensemble = pipeline.run(agents=[agent_0, agent_1], input_data=[1.0], context={})
    assert res_ensemble.agent_trust_weights["agent_label_0"] == pytest.approx(0.5, abs=1e-4)
    assert res_ensemble.agent_trust_weights["agent_label_1"] == pytest.approx(0.5, abs=1e-4)


# ── Section D: Non-Finite Input Handling and Safety Gating ─────────────────────

def test_non_finite_predictions_fail_loudly_in_research_mode():
    """Verify research mode rejects all non-finite predictions/confidences loudly."""
    pipeline = CognixPipeline(mode=RESEARCH_MODE)

    for test_mode in ["nan_bare", "pos_inf_bare", "neg_inf_bare", "nan_conf_in_result", "pos_inf_conf_in_result", "nan_value_in_result"]:
        agent = NonFiniteAgent(f"bad_agent_{test_mode}", mode=test_mode)
        with pytest.raises((ValueError, RuntimeError), match="(?i)non-finite|nan|inf"):
            pipeline.run(agents=[agent], input_data=[1.0], context={})


def test_non_finite_predictions_isolated_in_production_mode():
    """
    Verify production mode isolates non-finite agent output:
    - If a healthy agent is present, production proceeds with the healthy agent.
    - If all agents are invalid, production returns emergency ABSTAIN requiring escalation.
    - Invalid input NEVER produces a confident ACT outcome.
    """
    healthy = DomainNeutralAgent("healthy_sensor", pred_val=0.82, epistemic=0.08)
    corrupted = NonFiniteAgent("corrupted_sensor", mode="nan_bare")

    pipeline = CognixPipeline(belief_fuser=EpistemicWeightedFusion(), mode=PRODUCTION_MODE)

    # 1. Mixed ensemble: corrupted agent is isolated, healthy agent proceeds safely
    res_mixed = pipeline.run(agents=[healthy, corrupted], input_data=[1.0], context={})
    assert len(res_mixed.agent_predictions) == 1
    assert "healthy_sensor" in res_mixed.agent_predictions
    assert "corrupted_sensor" not in res_mixed.agent_predictions
    assert res_mixed.confidence == pytest.approx(0.82, abs=1e-4)
    assert res_mixed.decision in (DecisionOutcome.ACT, DecisionOutcome.WAIT, DecisionOutcome.REQUEST_INFORMATION)

    # 2. All agents corrupted: must return emergency ABSTAIN requiring escalation
    all_corrupted = [
        NonFiniteAgent("c1", mode="nan_bare"),
        NonFiniteAgent("c2", mode="pos_inf_bare"),
    ]
    res_emergency = pipeline.run(agents=all_corrupted, input_data=[1.0], context={})
    assert res_emergency.decision == DecisionOutcome.ABSTAIN
    assert res_emergency.escalation_required is True
    assert res_emergency.abstained is True
    assert res_emergency.risk_level == RiskLevel.HIGH
    assert res_emergency.decision != DecisionOutcome.ACT


# ── Section E: Failure Propagation and Fault Isolation ─────────────────────────

def test_predict_failure_isolation_in_production_mode():
    """Verify one failing agent does not crash production pipeline if other agents succeed."""
    agents = [
        DomainNeutralAgent("healthy_agent", pred_val=0.82, epistemic=0.1),
        FailingAgent("broken_agent", fail_in_predict=True),
    ]
    pipeline = CognixPipeline(belief_fuser=EpistemicWeightedFusion(), mode=PRODUCTION_MODE)
    result = pipeline.run(agents=agents, input_data=[0.5], context={})

    assert len(result.agent_predictions) == 1
    assert "healthy_agent" in result.agent_predictions
    assert "broken_agent" not in result.agent_predictions
    assert result.decision is not None


def test_predict_failure_loud_propagation_in_research_mode():
    """Verify agent predict failure raises RuntimeError in research mode."""
    agents = [
        DomainNeutralAgent("healthy_agent", pred_val=0.82, epistemic=0.1),
        FailingAgent("broken_agent", fail_in_predict=True),
    ]
    pipeline = CognixPipeline(belief_fuser=EpistemicWeightedFusion(), mode=RESEARCH_MODE)
    with pytest.raises(RuntimeError, match="(?i)predict failed"):
        pipeline.run(agents=agents, input_data=[0.5], context={})


def test_belief_fusion_failure_isolation_in_production():
    """Verify broken belief fuser records failure in module_status and uses fallback in production."""
    class BrokenFuser(BeliefFuser):
        def fuse(self, predictions, uncertainties, reliabilities):
            raise ValueError("Intentional fuser breakdown")

    agents = [DomainNeutralAgent("a1", 0.8), DomainNeutralAgent("a2", 0.6)]
    pipeline = CognixPipeline(belief_fuser=BrokenFuser(), mode=PRODUCTION_MODE)

    result = pipeline.run(agents=agents, input_data=[1], context={})
    status = result.metadata["module_status"]["belief"]
    assert status["executed"] is False
    assert "Intentional fuser breakdown" in status["failure_reason"]
    # Downstream pipeline completed safely with fallback
    assert result.decision is not None


def test_belief_fusion_failure_propagation_in_research():
    """Verify broken belief fuser raises loudly in research mode."""
    class BrokenFuser(BeliefFuser):
        def fuse(self, predictions, uncertainties, reliabilities):
            raise ValueError("Intentional fuser breakdown")

    agents = [DomainNeutralAgent("a1", 0.8), DomainNeutralAgent("a2", 0.6)]
    pipeline = CognixPipeline(belief_fuser=BrokenFuser(), mode=RESEARCH_MODE)

    with pytest.raises(RuntimeError, match="Intentional fuser breakdown"):
        pipeline.run(agents=agents, input_data=[1], context={})


# ── Section F: Repeatability and State Isolation ──────────────────────────────

def test_pipeline_determinism_and_state_isolation():
    """Verify repeated pipeline calls with equivalent inputs yield deterministic results."""
    agents = [
        DomainNeutralAgent("a1", pred_val=0.75, epistemic=0.10, aleatoric=0.05),
        DomainNeutralAgent("a2", pred_val=0.85, epistemic=0.08, aleatoric=0.04),
    ]
    engine = DecisionEngine(belief=EpistemicWeightedFusion(), mode=PRODUCTION_MODE)

    res1 = engine.decide(agents=agents, input_data=[1.0, 2.0], context={"run": 1})
    res2 = engine.decide(agents=agents, input_data=[1.0, 2.0], context={"run": 2})

    assert res1.decision == res2.decision
    assert res1.confidence == pytest.approx(res2.confidence, abs=1e-6)
    assert res1.epistemic_uncertainty == pytest.approx(res2.epistemic_uncertainty, abs=1e-6)
    assert res1.agent_trust_weights == pytest.approx(res2.agent_trust_weights, abs=1e-6)


# ── Section G: Serialization and Result Integrity ──────────────────────────────

def test_decision_result_to_dict_and_from_dict_roundtrip():
    """Verify DecisionResult serializes to dict/JSON and deserializes correctly."""
    agents = [DomainNeutralAgent("a1", pred_val=0.85, epistemic=0.08)]
    pipeline = CognixPipeline(belief_fuser=EpistemicWeightedFusion(), mode=PRODUCTION_MODE)
    result = pipeline.run(agents=agents, input_data=[1.0], context={"session_id": "serialize_01"})

    # 1. Serialize to dict
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["decision"] == result.decision.value
    assert d["risk_level"] == result.risk_level.value

    # 2. JSON roundtrip
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)

    # 3. Deserialize back to DecisionResult
    reconstructed = DecisionResult.from_dict(loaded)
    assert isinstance(reconstructed.decision, DecisionOutcome)
    assert isinstance(reconstructed.risk_level, RiskLevel)
    assert reconstructed.decision == result.decision
    assert reconstructed.risk_level == result.risk_level
    assert reconstructed.confidence == pytest.approx(result.confidence, abs=1e-6)
    assert reconstructed.is_safe_decision == result.is_safe_decision
    assert "Decision:" in reconstructed.summary()


def test_ascii_safe_explanation_formatting():
    """Verify reasoning steps and explanation do not crash on ASCII or cp1252 encodings."""
    agents = [DomainNeutralAgent("a1", pred_val=0.85, epistemic=0.08)]
    # Create a pipeline where a module failure occurs in production to test warning text
    class FailingFuser(BeliefFuser):
        def fuse(self, predictions, uncertainties, reliabilities):
            raise RuntimeError("Fuser error")

    pipeline = CognixPipeline(belief_fuser=FailingFuser(), mode=PRODUCTION_MODE)
    result = pipeline.run(agents=agents, input_data=[1.0], context={})

    for step in result.reasoning_steps:
        # Must be encodable as ascii without UnicodeEncodeError
        step.encode("ascii", errors="strict")


# ── Section H: Cross-Domain Boundary Test ──────────────────────────────────────

def test_cross_domain_medical_triage_boundary():
    """
    Test generic pipeline on an entirely different domain (medical diagnosis triage),
    confirming it operates correctly without any autonomous driving or CARLA dependencies.
    """
    # Three diagnostic clinical models evaluating probability of pathology
    clinical_agents = [
        DomainNeutralAgent("radiology_cnn", pred_val=0.88, epistemic=0.05, aleatoric=0.03),
        DomainNeutralAgent("pathology_mlp", pred_val=0.92, epistemic=0.04, aleatoric=0.02),
        DomainNeutralAgent("clinical_ehr", pred_val=0.79, epistemic=0.15, aleatoric=0.08),
    ]

    engine = DecisionEngine(
        belief=EpistemicWeightedFusion(),
        mode=PRODUCTION_MODE,
    )

    patient_vitals = {"systolic_bp": 135, "heart_rate": 78, "wbc_count": 9.5}
    result = engine.decide(
        agents=clinical_agents,
        input_data=patient_vitals,
        context={"patient_id": "PT-9042"},
    )

    assert result.decision in (DecisionOutcome.ACT, DecisionOutcome.REQUEST_INFORMATION, DecisionOutcome.ESCALATE)
    assert result.confidence > 0.80
    assert result.epistemic_uncertainty < 0.10
    # Radiology and Pathology have lower epistemic uncertainty, so should have higher trust weights
    assert result.agent_trust_weights["pathology_mlp"] > result.agent_trust_weights["clinical_ehr"]
    assert result.agent_trust_weights["radiology_cnn"] > result.agent_trust_weights["clinical_ehr"]


# ── Section I: DecisionEngine String Shorthand Resolution ──────────────────────

def test_decision_engine_shorthand_module_resolution():
    """
    Verify DecisionEngine correctly resolves documented string shorthands:
      uncertainty='mc_dropout', belief='epistemic_weighted', calibrator='conformal', escalation=True
    as promised by its docstring and public usage example.
    """
    engine = DecisionEngine(
        uncertainty="mc_dropout",
        belief="epistemic_weighted",
        calibrator="conformal",
        escalation=True,
        mode=PRODUCTION_MODE,
    )

    assert isinstance(engine.uncertainty, UncertaintyEstimator)
    assert isinstance(engine.belief, BeliefFuser)
    assert isinstance(engine.calibrator, ConformalPredictor)
    assert isinstance(engine.escalation_module, EscalationEngine)

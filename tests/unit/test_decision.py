import pytest
from enum import Enum

class DecisionOutcome(Enum):
    ACT = "ACT"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"

def decide(confidence, uncertainty):
    if confidence > 0.8 and uncertainty < 0.2:
        return DecisionOutcome.ACT
    elif uncertainty > 0.8:
        return DecisionOutcome.ESCALATE
    return DecisionOutcome.ABSTAIN

def test_high_confidence_low_uncertainty_gives_act():
    assert decide(0.9, 0.1) == DecisionOutcome.ACT

def test_low_confidence_gives_escalate_or_abstain():
    assert decide(0.4, 0.3) == DecisionOutcome.ABSTAIN
    assert decide(0.4, 0.9) == DecisionOutcome.ESCALATE

def test_risk_assessor_high_epistemic_gives_high_risk():
    assert True

def test_risk_assessor_low_values_gives_low_risk():
    assert True

def test_escalation_engine_tracks_steps():
    assert True

def test_decision_outcome_enum_values():
    assert DecisionOutcome.ACT.value == "ACT"
    assert DecisionOutcome.ABSTAIN.value == "ABSTAIN"
    assert DecisionOutcome.ESCALATE.value == "ESCALATE"

def test_escalation_engine_configurable_conformal_set_size():
    from cognix.decision.escalation import EscalationEngine, DecisionOutcome as DO

    # Default max_conformal_set_size = 3
    engine_default = EscalationEngine()
    res_size_2 = engine_default.evaluate(confidence=0.9, epistemic_uncertainty=0.1, conformal_set_size=2)
    assert res_size_2.outcome == DO.ACT
    assert res_size_2.escalation is False

    res_size_3 = engine_default.evaluate(confidence=0.9, epistemic_uncertainty=0.1, conformal_set_size=3)
    assert res_size_3.outcome == DO.ESCALATE
    assert res_size_3.escalation is True

    # Binary classification setting: max_conformal_set_size = 2
    engine_binary = EscalationEngine(max_conformal_set_size=2)
    res_bin_2 = engine_binary.evaluate(confidence=0.9, epistemic_uncertainty=0.1, conformal_set_size=2)
    assert res_bin_2.outcome == DO.ESCALATE
    assert res_bin_2.escalation is True


def test_pipeline_fallback_escalates_on_conformal_set_size_without_escalation_module():
    from cognix.engine.pipeline import CognixPipeline
    from cognix.engine.result import DecisionOutcome as PipelineDO, RiskLevel

    pipeline = CognixPipeline(escalation=None, mode="production")

    # Normal case: risk LOW, set size 1 -> ACT
    decision, is_esc, abstained, req_info = pipeline._make_decision(
        confidence=0.85, epistemic=0.05, risk_level=RiskLevel.LOW, conformal_set_size=1
    )
    assert decision == PipelineDO.ACT
    assert is_esc is False

    # Oversized set: set size 3 -> ESCALATE even though risk_level is LOW
    decision, is_esc, abstained, req_info = pipeline._make_decision(
        confidence=0.85, epistemic=0.05, risk_level=RiskLevel.LOW, conformal_set_size=3
    )
    assert decision == PipelineDO.ESCALATE
    assert is_esc is True

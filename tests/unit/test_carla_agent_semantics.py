"""
Unit tests for CARLA agent prediction semantics and dashboard calibration labels.

Verifies:
1.  CARLA agents expose P(ACT/safe) as PredictionResult.value — NOT P(hazard).
2.  hazard_probability is retained in metadata.
3.  P(ACT) + P(hazard) ≈ 1.0 for every agent.
4.  All returned probabilities are in [0, 1].
5.  Pipeline receives ACT probability (> 0.5 for low-hazard observations).
6.  Low-hazard NORMAL observations do NOT trigger HIGH RISK / ESCALATE.
7.  Dashboard payload calibration fields are truthful:
      - calibrated_confidence is None when no calibrator injected.
      - calibration_available is False.
      - collective_probability is the raw fused confidence.
8.  No fake conformal prediction set is generated without a calibrator.
"""
import pytest
import numpy as np

from cognix.adapters.carla.agents import (
    CameraAgent, DepthAgent, LiDARAgent,
    GNSSAgent, IMUAgent, SegAgent,
)
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix import DecisionEngine, CognixConfig
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.engine.result import DecisionResult


# ── helpers ──────────────────────────────────────────────────────────────────

def _normal_inputs():
    """Generate one NORMAL frame and build the agent input dict."""
    dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=1)
    frames = dataset.generate_frames("NORMAL")
    frame = frames[0]
    return {
        "Camera": frame.rgb,
        "Depth":  frame.depth,
        "LiDAR":  frame.lidar,
        "GNSS":   frame.gnss,
        "IMU":    frame.imu,
        "Seg":    frame.segmentation,
    }


ALL_AGENT_CLASSES = [CameraAgent, DepthAgent, LiDARAgent, GNSSAgent, IMUAgent, SegAgent]


# ── Test 1: PredictionResult.value is P(ACT/safe), i.e. > 0.5 for NORMAL ───

@pytest.mark.parametrize("AgentCls", ALL_AGENT_CLASSES)
def test_predict_value_is_p_act_not_p_hazard(AgentCls):
    """For a NORMAL observation the hazard score is low (~0.05–0.15),
    so the inverted P(ACT) must be significantly above 0.5."""
    agent = AgentCls()
    inputs = _normal_inputs()
    result = agent.predict(inputs)
    # P(ACT/safe) should be in [0, 1]
    assert 0.0 <= result.value <= 1.0, f"value={result.value} out of [0,1]"
    # For NORMAL driving the ACT probability must be > 0.5
    assert result.value > 0.5, (
        f"{AgentCls.__name__}: expected P(ACT) > 0.5 for NORMAL, got {result.value:.4f}. "
        f"If value < 0.5 the semantic inversion is wrong."
    )


# ── Test 2: hazard_probability is preserved in metadata ──────────────────────

@pytest.mark.parametrize("AgentCls", ALL_AGENT_CLASSES)
def test_hazard_probability_in_metadata(AgentCls):
    agent = AgentCls()
    inputs = _normal_inputs()
    result = agent.predict(inputs)
    assert "hazard_probability" in result.metadata, (
        f"{AgentCls.__name__}: metadata missing 'hazard_probability'"
    )
    hp = result.metadata["hazard_probability"]
    assert 0.0 <= hp <= 1.0, f"hazard_probability={hp} out of [0,1]"


# ── Test 3: P(ACT) + P(hazard) ≈ 1.0 ────────────────────────────────────────

@pytest.mark.parametrize("AgentCls", ALL_AGENT_CLASSES)
def test_act_plus_hazard_equals_one(AgentCls):
    agent = AgentCls()
    inputs = _normal_inputs()
    result = agent.predict(inputs)
    hp = result.metadata["hazard_probability"]
    total = result.value + hp
    assert abs(total - 1.0) < 1e-9, (
        f"{AgentCls.__name__}: P(ACT)={result.value:.6f} + P(hazard)={hp:.6f} = {total:.6f} ≠ 1.0"
    )


# ── Test 4: prediction_semantics annotation is present ───────────────────────

@pytest.mark.parametrize("AgentCls", ALL_AGENT_CLASSES)
def test_prediction_semantics_annotation(AgentCls):
    agent = AgentCls()
    inputs = _normal_inputs()
    result = agent.predict(inputs)
    assert result.metadata.get("prediction_semantics") == "P(ACT/safe)", (
        f"{AgentCls.__name__}: missing or wrong prediction_semantics annotation"
    )


# ── Test 5: confidence field is also P(ACT/safe) ─────────────────────────────

@pytest.mark.parametrize("AgentCls", ALL_AGENT_CLASSES)
def test_confidence_equals_value(AgentCls):
    """After the fix confidence == value (both equal P(ACT/safe))."""
    agent = AgentCls()
    inputs = _normal_inputs()
    result = agent.predict(inputs)
    assert abs(result.confidence - result.value) < 1e-9, (
        f"{AgentCls.__name__}: confidence={result.confidence} ≠ value={result.value}"
    )


# ── Test 6: low-hazard NORMAL → ACT, not ESCALATE ────────────────────────────

def test_normal_scenario_produces_act_not_escalate():
    """After the semantic fix a NORMAL frame must resolve to ACT / LOW RISK."""
    config = CognixConfig()
    engine = DecisionEngine(config=config, attribution=EpistemicShapley())
    agents = [cls() for cls in ALL_AGENT_CLASSES]
    inputs = _normal_inputs()
    reliabilities = {a.agent_id: 1.0 for a in agents}

    result: DecisionResult = engine.decide(agents, inputs, agent_reliabilities=reliabilities)

    assert result.decision.name == "ACT", (
        f"NORMAL scenario produced {result.decision.name} "
        f"(confidence={result.confidence:.4f}, risk={result.risk_level.name}). "
        "The semantic inversion fix may not have taken effect."
    )
    from cognix.engine.result import RiskLevel
    assert result.risk_level == RiskLevel.LOW, (
        f"NORMAL scenario risk={result.risk_level.name}, expected LOW."
    )


# ── Test 7: fused confidence > 0.7 for NORMAL ───────────────────────────────

def test_normal_fused_confidence_above_threshold():
    """The fused P(ACT) for NORMAL must exceed the HIGH_CONF_THRESH=0.7."""
    config = CognixConfig()
    engine = DecisionEngine(config=config, attribution=EpistemicShapley())
    agents = [cls() for cls in ALL_AGENT_CLASSES]
    inputs = _normal_inputs()
    reliabilities = {a.agent_id: 1.0 for a in agents}
    result = engine.decide(agents, inputs, agent_reliabilities=reliabilities)
    assert result.confidence > 0.7, (
        f"NORMAL fused P(ACT)={result.confidence:.4f} is below 0.7. "
        "This suggests the hazard→ACT inversion is still incomplete."
    )


# ── Test 8: dashboard payload has truthful calibration fields ────────────────

def test_dashboard_calibration_fields_without_calibrator():
    """When no ConformalPredictor is injected:
    - calibrated_confidence should be None
    - calibration_available should be False
    - collective_probability should equal result.confidence
    No fake conformal set should be generated.
    """
    config = CognixConfig()
    engine = DecisionEngine(config=config, attribution=EpistemicShapley())
    agents = [cls() for cls in ALL_AGENT_CLASSES]
    inputs = _normal_inputs()
    reliabilities = {a.agent_id: 1.0 for a in agents}
    result = engine.decide(agents, inputs, agent_reliabilities=reliabilities)

    # These match what dashboard/app.py now computes
    calibrated_confidence = result.calibrated_confidence
    calibration_available = result.calibrated_confidence is not None
    collective_probability = result.confidence

    assert calibrated_confidence is None, (
        f"Expected calibrated_confidence=None (no calibrator), got {calibrated_confidence}"
    )
    assert calibration_available is False, (
        "Expected calibration_available=False when no ConformalPredictor injected."
    )
    assert 0.0 <= collective_probability <= 1.0, (
        f"collective_probability={collective_probability} out of [0,1]"
    )


# ── Test 9: no fake conformal set without calibrator ─────────────────────────

def test_no_fake_conformal_set_without_calibrator():
    """Dashboard app.py must emit conformal_set=None when no calibrator exists.
    Simulate the app.py logic directly."""
    config = CognixConfig()
    engine = DecisionEngine(config=config, attribution=EpistemicShapley())
    agents = [cls() for cls in ALL_AGENT_CLASSES]
    inputs = _normal_inputs()
    reliabilities = {a.agent_id: 1.0 for a in agents}
    result = engine.decide(agents, inputs, agent_reliabilities=reliabilities)

    # Replicate the exact dashboard logic
    calibration_metrics = result.calibration_metrics or {}
    real_prediction_set = calibration_metrics.get("prediction_set", None)
    conformal_set = real_prediction_set if real_prediction_set is not None else None

    assert conformal_set is None, (
        f"Expected conformal_set=None without calibrator, got {conformal_set}"
    )

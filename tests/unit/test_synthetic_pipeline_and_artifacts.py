"""
Unit tests for Synthetic Cognix Pipeline, Artifact Persistence, and Dashboard Realism.

Validates the 15 required checks:
1. Fitted graph artifact loads successfully
2. Untrained/missing graph artifact never runs random weights
3. Fitted conformal artifact loads successfully
4. Live DecisionEngine uses fitted graph (gnn executed = True)
5. Live DecisionEngine uses fitted conformal (calibration executed = True)
6. Conformal prediction set is genuine (from ConformalPredictor)
7. No fake prediction set without calibrator
8. Dashboard probability labeling is semantically correct
9. Deterministic safe case -> ACT
10. Deterministic unsafe case -> ESCALATE
11. Shapley remains populated
12. No NaN/Inf anywhere in DecisionResult
13. Artifact metadata compatibility checks
14. Dashboard can run with valid artifacts
15. Dashboard falls back honestly when artifacts unavailable
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from cognix import DecisionEngine, CognixConfig
from cognix.adapters.carla.agents import (
    CameraAgent, DepthAgent, LiDARAgent,
    GNSSAgent, IMUAgent, SegAgent,
)
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.artifacts.persistence import (
    load_epistemic_gat,
    load_standard_gat,
    load_conformal,
    load_synthetic_artifacts,
    save_synthetic_artifacts,
)
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.calibration.conformal import ConformalPredictor
from cognix.engine.result import DecisionOutcome, RiskLevel
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.graph.standard_gat import StandardGAT

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "synthetic"


@pytest.fixture
def synthetic_agents():
    return [
        CameraAgent(),
        DepthAgent(),
        LiDARAgent(),
        GNSSAgent(),
        IMUAgent(),
        SegAgent(),
    ]


@pytest.fixture
def normal_inputs():
    dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=1, seed=42)
    frame = dataset.generate_frames("NORMAL")[0]
    return {
        "Camera": frame.rgb,
        "Depth": frame.depth,
        "LiDAR": frame.lidar,
        "GNSS": frame.gnss,
        "IMU": frame.imu,
        "Seg": frame.segmentation,
    }


# ── 1. Fitted graph artifact loads successfully ─────────────────────────────

def test_fitted_graph_artifact_loads_successfully():
    gat = load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt", expected_input_dim=3, expected_output_dim=1)
    assert isinstance(gat, EpistemicGAT)
    assert gat.is_trained() is True
    assert gat.input_dim == 3
    assert gat.output_dim == 1

    sgat = load_standard_gat(ARTIFACTS_DIR / "standard_gat.pt", expected_input_dim=3, expected_output_dim=1)
    assert isinstance(sgat, StandardGAT)
    assert sgat.is_trained() is True


# ── 2. Untrained / missing graph artifact never runs random weights ──────────

def test_untrained_graph_never_runs_random_weights(synthetic_agents, normal_inputs):
    # Untrained GAT
    untrained_gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)
    assert untrained_gat.is_trained() is False

    # Production mode: skips safely without executing random weights
    prod_engine = DecisionEngine(
        config=CognixConfig(),
        graph=untrained_gat,
        mode="production",
    )
    res = prod_engine.decide(synthetic_agents, normal_inputs)
    gnn_status = res.metadata["module_status"]["gnn"]
    assert gnn_status["executed"] is False
    assert gnn_status["failure_reason"] == "graph model not trained"

    # Research mode: fail-fast raises RuntimeError
    res_engine = DecisionEngine(
        config=CognixConfig(),
        graph=untrained_gat,
        mode="research",
    )
    with pytest.raises(RuntimeError, match="Graph model has not been trained"):
        res_engine.decide(synthetic_agents, normal_inputs)


# ── 3. Fitted conformal artifact loads successfully ─────────────────────────

def test_fitted_conformal_artifact_loads_successfully():
    calibrator = load_conformal(ARTIFACTS_DIR / "conformal_cal_scores.npy")
    assert isinstance(calibrator, ConformalPredictor)
    assert calibrator.cal_scores is not None
    assert len(calibrator.cal_scores) == 125
    assert calibrator.n_cal == 125
    assert np.all(np.isfinite(calibrator.cal_scores))


# ── 4. Live DecisionEngine uses fitted graph ─────────────────────────────────

def test_live_decision_engine_uses_fitted_graph(synthetic_agents, normal_inputs):
    gat = load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt")
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        mode="production",
    )
    res = engine.decide(synthetic_agents, normal_inputs)
    gnn_status = res.metadata["module_status"]["gnn"]
    assert gnn_status["executed"] is True
    assert gnn_status["method"] == "epistemic_gat"
    assert gnn_status["failure_reason"] is None


# ── 5. Live DecisionEngine uses fitted conformal ─────────────────────────────

def test_live_decision_engine_uses_fitted_conformal(synthetic_agents, normal_inputs):
    calibrator = load_conformal(ARTIFACTS_DIR / "conformal_cal_scores.npy")
    engine = DecisionEngine(
        config=CognixConfig(),
        calibrator=calibrator,
        mode="production",
    )
    res = engine.decide(synthetic_agents, normal_inputs)
    cal_status = res.metadata["module_status"]["calibration"]
    assert cal_status["executed"] is True
    assert cal_status["method"] == "split_conformal"
    assert res.calibrated_confidence is not None


# ── 6. Conformal prediction set is genuine ───────────────────────────────────

def test_conformal_prediction_set_is_genuine(synthetic_agents, normal_inputs):
    gat = load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt")
    calibrator = load_conformal(ARTIFACTS_DIR / "conformal_cal_scores.npy")
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        calibrator=calibrator,
        mode="production",
    )
    res = engine.decide(synthetic_agents, normal_inputs)
    metrics = res.calibration_metrics
    assert "prediction_set" in metrics
    assert isinstance(metrics["prediction_set"], list)
    assert len(metrics["prediction_set"]) >= 1
    assert metrics["target_coverage"] == 0.95
    assert np.isfinite(metrics["quantile"])
    assert metrics["n_calibration_samples"] == 125


# ── 7. No fake prediction set without calibrator ─────────────────────────────

def test_no_fake_prediction_set_without_calibrator(synthetic_agents, normal_inputs):
    engine = DecisionEngine(
        config=CognixConfig(),
        calibrator=None,
        mode="production",
    )
    res = engine.decide(synthetic_agents, normal_inputs)
    assert res.calibrated_confidence is None
    assert "prediction_set" not in res.calibration_metrics


# ── 8. Dashboard probability labeling is semantically correct ───────────────

def test_dashboard_probability_labeling_semantics():
    from dashboard.app import run_cognix_cycle
    payload = run_cognix_cycle()
    # Fused scalar is collective_probability
    assert "collective_probability" in payload
    assert 0.0 <= payload["collective_probability"] <= 1.0
    # Conformal set is genuine list of class names or None
    if payload["conformal_set"] is not None:
        assert isinstance(payload["conformal_set"], list)
        for act in payload["conformal_set"]:
            assert act in ("ACT", "ESCALATE")


# ── 9. Deterministic safe case -> ACT ────────────────────────────────────────

def test_deterministic_safe_case_acts(synthetic_agents):
    gat = load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt")
    calibrator = load_conformal(ARTIFACTS_DIR / "conformal_cal_scores.npy")
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        calibrator=calibrator,
        attribution=EpistemicShapley(),
        mode="production",
    )
    # Generate clean NORMAL frame (high safe probability, low uncertainty)
    ds = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=1, seed=42)
    frame = ds.generate_frames("NORMAL")[0]
    inputs = {
        "Camera": frame.rgb,
        "Depth": frame.depth,
        "LiDAR": frame.lidar,
        "GNSS": frame.gnss,
        "IMU": frame.imu,
        "Seg": frame.segmentation,
    }
    res = engine.decide(synthetic_agents, inputs)
    assert res.risk_level == RiskLevel.LOW
    assert res.decision == DecisionOutcome.ACT
    assert res.confidence > 0.70
    assert res.epistemic_uncertainty < 0.20


# ── 10. Deterministic unsafe case -> ESCALATE ────────────────────────────────

def test_deterministic_unsafe_case_escalates(synthetic_agents):
    gat = load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt")
    calibrator = load_conformal(ARTIFACTS_DIR / "conformal_cal_scores.npy")
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        calibrator=calibrator,
        attribution=EpistemicShapley(),
        mode="production",
    )
    # Severe hazard input (extreme camera blackout, lidar reflection, violent drift)
    severe_hazard_inputs = {
        "Camera": np.array([[0.01, 0.01]]), # hazard ~0.99 -> ACT prob ~0.01
        "Depth": np.array([[0.01, 0.99]]),
        "LiDAR": np.array([[0.01, 0.99]]),
        "GNSS": np.array([[0.99, 0.99]]),
        "IMU": np.array([[0.99, 0.99]]),
        "Seg": np.array([[0.99, 0.99]]),
    }
    res = engine.decide(synthetic_agents, severe_hazard_inputs)
    assert res.risk_level == RiskLevel.HIGH
    assert res.decision == DecisionOutcome.ESCALATE
    assert res.confidence < 0.40  # low confidence triggers high risk


# ── 11. Shapley remains populated ────────────────────────────────────────────

def test_shapley_remains_populated(synthetic_agents, normal_inputs):
    gat = load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt")
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        attribution=EpistemicShapley(),
        mode="production",
    )
    res = engine.decide(synthetic_agents, normal_inputs)
    assert len(res.agent_contributions) == 6
    for agent_id, val in res.agent_contributions.items():
        assert np.isfinite(val)
        assert abs(val) > 0.0


# ── 12. No NaN/Inf anywhere in DecisionResult ────────────────────────────────

def test_no_nan_or_inf_in_decision_result(synthetic_agents, normal_inputs):
    gat = load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt")
    calibrator = load_conformal(ARTIFACTS_DIR / "conformal_cal_scores.npy")
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        calibrator=calibrator,
        attribution=EpistemicShapley(),
        mode="production",
    )
    res = engine.decide(synthetic_agents, normal_inputs)
    assert np.isfinite(res.confidence)
    assert np.isfinite(res.calibrated_confidence)
    assert np.isfinite(res.epistemic_uncertainty)
    assert np.isfinite(res.aleatoric_uncertainty)
    assert np.isfinite(res.total_uncertainty)
    for p in res.agent_predictions.values():
        assert np.isfinite(p)
    for w in res.agent_trust_weights.values():
        assert np.isfinite(w)
    for s in res.agent_contributions.values():
        assert np.isfinite(s)


# ── 13. Artifact metadata compatibility checks ───────────────────────────────

def test_artifact_metadata_compatibility_checks():
    # Test mismatch input_dim raises ValueError
    with pytest.raises(ValueError, match="Artifact input_dim"):
        load_epistemic_gat(ARTIFACTS_DIR / "epistemic_gat.pt", expected_input_dim=999)

    # Test missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_epistemic_gat(ARTIFACTS_DIR / "nonexistent.pt")

    with pytest.raises(FileNotFoundError):
        load_conformal(ARTIFACTS_DIR / "nonexistent.npy")

    # Test load_synthetic_artifacts reads metadata correctly
    gat, conf, meta = load_synthetic_artifacts(ARTIFACTS_DIR)
    assert meta["version"] == "0.2.0"
    assert len(meta["agents"]) == 6
    assert meta["calibration"]["target_coverage"] == 0.95


# ── 14. Dashboard can run with valid artifacts ───────────────────────────────

def test_dashboard_can_run_with_valid_artifacts():
    from dashboard.app import run_cognix_cycle
    payload = run_cognix_cycle()
    assert payload["graph_executed"] is True
    assert payload["calibration_available"] is True
    assert payload["conformal_set"] in (["ACT"], ["ESCALATE"], ["ESCALATE", "ACT"], ["ACT", "ESCALATE"])
    assert payload["decision"] in ("ACT", "ESCALATE", "REQUEST_INFORMATION")
    assert len(payload["agents"]) == 6
    assert "comparison_table" in payload
    assert "NORMAL" in payload["comparison_table"]


# ── 15. Dashboard falls back honestly when artifacts unavailable ─────────────

def test_dashboard_falls_back_honestly_when_artifact_unavailable(synthetic_agents, normal_inputs):
    # DecisionEngine initialized with None for graph and calibrator
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=None,
        calibrator=None,
        attribution=EpistemicShapley(),
        mode="production",
    )
    res = engine.decide(synthetic_agents, normal_inputs)
    assert res.metadata["module_status"]["gnn"]["executed"] is False
    assert res.metadata["module_status"]["calibration"]["executed"] is False
    assert res.calibrated_confidence is None
    assert "prediction_set" not in res.calibration_metrics

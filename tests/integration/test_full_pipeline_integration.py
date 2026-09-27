"""
COGNIX End-to-End Pipeline Integration Test Suite.

Validates the full M1 -> M4 -> M2 -> M3 -> M5 decision cycle using real Cognix modules:
  - MC-Dropout Agents (HeterogeneousAgent)
  - Trained Graph Refinement (StandardGAT and EpistemicGAT)
  - Belief Fusion (EpistemicWeightedFusion)
  - Calibrated Conformal Prediction (ConformalPredictor fitted strictly on CAL)
  - Epistemic Shapley Attribution (EpistemicShapley)
  - Risk Assessment & Decision (DecisionEngine)

Enforces strict split separation:
  TRAIN (seed 42) != CALIBRATION (seed 1042) != TEST (seed 2042)
"""

import numpy as np
import pytest
import torch

from cognix.config.schema import CognixConfig
from cognix.core.types import PredictionResult, UncertaintyResult
from cognix.engine.decision_engine import DecisionEngine
from cognix.engine.result import DecisionOutcome, RiskLevel
from cognix.evaluation.run import HeterogeneousAgent
from cognix.evaluation.scenarios import DataGenerator
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.graph.standard_gat import StandardGAT
from cognix.graph.no_graph import NoGraph
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor
from cognix.attribution.epistemic_shapley import EpistemicShapley


@pytest.fixture(scope="module")
def pipeline_data_and_agents():
    """
    Module fixture creating deterministic synthetic data splits and trained agents.
    Strict split separation:
      TRAIN: seed 42 (n=50)
      CAL:   seed 1042 (n=50)
      TEST:  seed 2042 (n=20)
    """
    # 1. Deterministic data generation
    num_agents = 3
    gen_train = DataGenerator(n_samples=50, seed=42)
    gen_cal = DataGenerator(n_samples=50, seed=1042)
    gen_test = DataGenerator(n_samples=20, seed=2042)

    X_train, y_train = gen_train.generate_base(num_features=num_agents)
    X_cal, y_cal = gen_cal.generate_base(num_features=num_agents)
    X_test, y_test = gen_test.generate_base(num_features=num_agents)

    # 2. Instantiate and fit agents on TRAIN only
    torch.manual_seed(42)
    np.random.seed(42)
    agents = [HeterogeneousAgent(f"Agent_{i}", i) for i in range(num_agents)]
    for agent in agents:
        agent.T = 15
        agent.fit(X_train, y_train)

    # 3. Compute agent reliabilities from CAL split
    agent_reliabilities = {}
    for agent in agents:
        cal_preds = [round(agent.predict(X_cal[i]).value) for i in range(len(X_cal))]
        agent_reliabilities[agent.agent_id] = float(np.mean(np.array(cal_preds) == y_cal))

    return {
        "agents": agents,
        "X_train": X_train,
        "y_train": y_train,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
        "agent_reliabilities": agent_reliabilities,
    }


def _calibrate_conformal(agents, gat, fuser, X_cal, y_cal, agent_reliabilities):
    """
    Run CAL split through upstream path (agents -> UQ -> fitted GAT -> fusion)
    with conformal disabled, collecting calibration probabilities to fit ConformalPredictor.
    """
    cal_engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        belief=fuser,
        calibrator=None,
        mode="production",
    )
    cal_probs = []
    for x_c in X_cal:
        res = cal_engine.decide(
            agents=agents,
            input_data=x_c,
            context={"split": "calibration"},
            agent_reliabilities=agent_reliabilities,
        )
        p = float(np.clip(res.confidence, 1e-7, 1.0 - 1e-7))
        cal_probs.append([1.0 - p, p])

    calibrator = ConformalPredictor()
    calibrator.fit(np.array(cal_probs, dtype=np.float32), y_cal.astype(int))
    return calibrator


def test_agent_canonical_contracts(pipeline_data_and_agents):
    """Phase 4: Verify agents output canonical PredictionResult and UncertaintyResult."""
    data = pipeline_data_and_agents
    agents = data["agents"]
    x_sample = data["X_train"][0]

    for agent in agents:
        pred = agent.predict(x_sample)
        assert isinstance(pred, PredictionResult), f"{agent.agent_id} did not return PredictionResult"
        assert 0.0 <= pred.value <= 1.0
        assert 0.0 <= pred.confidence <= 1.0

        unc = agent.estimate_uncertainty(x_sample)
        assert isinstance(unc, UncertaintyResult), f"{agent.agent_id} did not return UncertaintyResult"
        assert 0.0 <= unc.prediction <= 1.0
        assert unc.epistemic >= 0.0
        assert unc.aleatoric >= 0.0
        assert unc.total >= 0.0
        assert np.isclose(unc.total, unc.epistemic + unc.aleatoric, atol=1e-5)


@pytest.mark.parametrize("graph_type", ["StandardGAT", "EpistemicGAT"])
def test_full_pipeline_e2e_research_mode(pipeline_data_and_agents, graph_type):
    """
    Phases 5-8: Full DecisionEngine E2E integration test in RESEARCH MODE.
    Tests both StandardGAT and EpistemicGAT.
    """
    data = pipeline_data_and_agents
    agents = data["agents"]
    X_train, y_train = data["X_train"], data["y_train"]
    X_cal, y_cal = data["X_cal"], data["y_cal"]
    X_test = data["X_test"]
    agent_reliabilities = data["agent_reliabilities"]

    # 1. Fit GAT strictly on TRAIN data
    if graph_type == "StandardGAT":
        gat = StandardGAT(input_dim=3, hidden_dim=8, output_dim=1)
        gat.fit(agents, X_train, y_train, epochs=20, lr=1e-3, seed=42)
    else:
        gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1, use_epistemic_prior=True)
        gat.fit(agents, X_train, y_train, epochs=20, lr=1e-3, seed=42)

    assert gat.is_trained() is True, f"{graph_type} should report is_trained() == True after fit"

    # 2. Upstream belief fuser
    fuser = EpistemicWeightedFusion(eps=1e-8)

    # 3. Fit ConformalPredictor strictly on CAL data
    calibrator = _calibrate_conformal(agents, gat, fuser, X_cal, y_cal, agent_reliabilities)
    assert calibrator.cal_scores is not None
    assert calibrator.n_cal == len(X_cal)

    # 4. Construct complete DecisionEngine in RESEARCH MODE
    engine = DecisionEngine(
        config=CognixConfig(),
        graph=gat,
        belief=fuser,
        calibrator=calibrator,
        attribution=EpistemicShapley(),
        mode="research",
    )

    # 5. Run test samples through DecisionEngine
    for idx in range(min(5, len(X_test))):
        x_test = X_test[idx]
        result = engine.decide(
            agents=agents,
            input_data=x_test,
            context={"session_id": f"e2e_{graph_type}_{idx}"},
            agent_reliabilities=agent_reliabilities,
        )

        # 1. Agent predictions
        assert len(result.agent_predictions) == len(agents)
        for a in agents:
            assert a.agent_id in result.agent_predictions
            assert 0.0 <= result.agent_predictions[a.agent_id] <= 1.0

        # 2 & 3. Finite epistemic and aleatoric uncertainties
        assert result.epistemic_uncertainty is not None
        assert np.isfinite(result.epistemic_uncertainty) and result.epistemic_uncertainty >= 0.0
        assert result.aleatoric_uncertainty is not None
        assert np.isfinite(result.aleatoric_uncertainty) and result.aleatoric_uncertainty >= 0.0
        assert np.isfinite(result.total_uncertainty) and result.total_uncertainty >= 0.0

        # 4. Graph stage executed
        module_status = result.metadata["module_status"]
        assert module_status["gnn"]["executed"] is True
        assert module_status["gnn"]["method"] == "epistemic_gat"
        assert module_status["gnn"]["failure_reason"] is None

        # 5. Belief fusion executed
        assert module_status["belief"]["executed"] is True
        assert "epistemic" in module_status["belief"]["method"]
        assert module_status["belief"]["failure_reason"] is None

        # 6. Conformal calibration executed
        assert module_status["calibration"]["executed"] is True
        assert module_status["calibration"]["method"] == "split_conformal"
        assert module_status["calibration"]["failure_reason"] is None

        # 7. Genuine prediction set
        assert result.calibration_metrics is not None
        pred_set = result.calibration_metrics.get("prediction_set")
        assert isinstance(pred_set, list)
        assert len(pred_set) > 0
        assert all(isinstance(cls_idx, (int, np.integer)) for cls_idx in pred_set)

        # 8. Genuine calibrated/conformal metadata
        assert result.calibration_metrics.get("method") == "split_conformal"
        assert result.calibration_metrics.get("n_calibration_samples") == len(X_cal)
        assert np.isfinite(result.calibration_metrics.get("quantile"))
        assert result.calibrated_confidence is not None
        assert np.isfinite(result.calibrated_confidence)

        # 9. Valid risk level
        assert isinstance(result.risk_level, RiskLevel)

        # 10. Valid final decision
        assert isinstance(result.decision, DecisionOutcome)

        # 11 & 12. Non-empty, finite Shapley contributions
        assert len(result.agent_contributions) == len(agents)
        for a in agents:
            assert a.agent_id in result.agent_contributions
            phi = result.agent_contributions[a.agent_id]
            assert np.isfinite(phi), f"Shapley value for {a.agent_id} is non-finite: {phi}"

        # 13. Attribution executed
        assert module_status["attribution"]["executed"] is True
        assert module_status["attribution"]["method"] == "epistemic_shapley"
        assert module_status["attribution"]["failure_reason"] is None

        # 14. Real stage latency measurements
        expected_latency_keys = {
            "collect_predictions", "estimate_uncertainty", "compute_trust",
            "graph_refinement", "belief_fusion", "aggregation", "calibration",
            "risk_assessment", "decision", "attribution", "explanation",
        }
        assert expected_latency_keys.issubset(result.latency_ms.keys())
        for stage, lat in result.latency_ms.items():
            assert np.isfinite(lat) and lat >= 0.0, f"Stage {stage} latency non-finite or negative: {lat}"
        assert np.isfinite(result.total_latency_ms) and result.total_latency_ms > 0.0

        # 15. No NaN / Inf in any primary output
        assert np.isfinite(result.confidence) and 0.0 <= result.confidence <= 1.0


# ── Phase 9: Focused Failure-Mode Tests ──────────────────────────────────────────

def test_failure_mode_a_research_untrained_gat_raises(pipeline_data_and_agents):
    """Failure Mode A: RESEARCH MODE with untrained learned GAT must raise RuntimeError."""
    data = pipeline_data_and_agents
    agents = data["agents"]
    x_test = data["X_test"][0]

    untrained_gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)
    assert untrained_gat.is_trained() is False

    engine = DecisionEngine(
        config=CognixConfig(),
        graph=untrained_gat,
        belief=EpistemicWeightedFusion(),
        mode="research",
    )
    with pytest.raises(RuntimeError, match="not been trained"):
        engine.decide(agents=agents, input_data=x_test, context={})

    # Same check for StandardGAT
    untrained_standard = StandardGAT(input_dim=3, hidden_dim=8, output_dim=1)
    assert untrained_standard.is_trained() is False

    engine_std = DecisionEngine(
        config=CognixConfig(),
        graph=untrained_standard,
        belief=EpistemicWeightedFusion(),
        mode="research",
    )
    with pytest.raises(RuntimeError, match="not been trained"):
        engine_std.decide(agents=agents, input_data=x_test, context={})


def test_failure_mode_b_production_untrained_gat_skips_safely(pipeline_data_and_agents):
    """
    Failure Mode B: PRODUCTION MODE with untrained learned GAT must:
      - skip graph refinement (no random forward pass)
      - log a warning
      - set executed=False with failure_reason="graph model not trained"
      - retain original agent predictions
      - return a valid downstream DecisionResult
    """
    data = pipeline_data_and_agents
    agents = data["agents"]
    x_test = data["X_test"][0]

    untrained_gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)
    assert untrained_gat.is_trained() is False

    engine = DecisionEngine(
        config=CognixConfig(),
        graph=untrained_gat,
        belief=EpistemicWeightedFusion(),
        mode="production",
    )

    result = engine.decide(agents=agents, input_data=x_test, context={})

    # Graph stage must be recorded as NOT executed due to untrained model
    gnn_status = result.metadata["module_status"]["gnn"]
    assert gnn_status["executed"] is False
    assert gnn_status["failure_reason"] == "graph model not trained"

    # Original agent predictions retained without GAT transformation
    for a in agents:
        raw_pred = a.predict(x_test).value
        pred_val = result.agent_predictions[a.agent_id]
        assert isinstance(pred_val, float), f"Expected float, got {type(pred_val).__name__}"
        assert np.isclose(pred_val, raw_pred, atol=1e-5)

    # Downstream decision continues safely
    assert isinstance(result.decision, DecisionOutcome)
    assert np.isfinite(result.confidence)


def test_failure_mode_c_research_unfitted_conformal_raises(pipeline_data_and_agents):
    """Failure Mode C: RESEARCH MODE with unfitted ConformalPredictor must raise RuntimeError."""
    data = pipeline_data_and_agents
    agents = data["agents"]
    x_test = data["X_test"][0]

    unfitted_cal = ConformalPredictor()
    assert unfitted_cal.cal_scores is None

    engine = DecisionEngine(
        config=CognixConfig(),
        calibrator=unfitted_cal,
        belief=EpistemicWeightedFusion(),
        mode="research",
    )
    with pytest.raises(RuntimeError, match="has not been calibrated"):
        engine.decide(agents=agents, input_data=x_test, context={})


def test_failure_mode_d_production_unfitted_conformal_graceful_fallback(pipeline_data_and_agents):
    """
    Failure Mode D: PRODUCTION MODE with unfitted ConformalPredictor must:
      - not report calibration as executed
      - not generate a fake conformal prediction set
      - set failure_reason clearly
      - allow valid fallback decision to proceed
    """
    data = pipeline_data_and_agents
    agents = data["agents"]
    x_test = data["X_test"][0]

    unfitted_cal = ConformalPredictor()
    assert unfitted_cal.cal_scores is None

    engine = DecisionEngine(
        config=CognixConfig(),
        calibrator=unfitted_cal,
        belief=EpistemicWeightedFusion(),
        mode="production",
    )

    result = engine.decide(agents=agents, input_data=x_test, context={})

    cal_status = result.metadata["module_status"]["calibration"]
    assert cal_status["executed"] is False
    assert cal_status["failure_reason"] is not None
    assert "has not been calibrated" in cal_status["failure_reason"]

    # No fake conformal prediction set or calibrated confidence
    assert result.calibration_metrics.get("prediction_set") is None
    assert result.calibrated_confidence is None

    # Valid fallback decision proceeds
    assert isinstance(result.decision, DecisionOutcome)
    assert np.isfinite(result.confidence)


def test_stateless_nograph_allowed_in_both_modes(pipeline_data_and_agents):
    """Verify stateless NoGraph is NOT rejected as untrained in either mode."""
    data = pipeline_data_and_agents
    agents = data["agents"]
    x_test = data["X_test"][0]

    # Research mode
    engine_res = DecisionEngine(
        config=CognixConfig(),
        graph=NoGraph(),
        belief=EpistemicWeightedFusion(),
        mode="research",
    )
    res_res = engine_res.decide(agents=agents, input_data=x_test, context={})
    assert res_res.metadata["module_status"]["gnn"]["executed"] is True

    # Production mode
    engine_prod = DecisionEngine(
        config=CognixConfig(),
        graph=NoGraph(),
        belief=EpistemicWeightedFusion(),
        mode="production",
    )
    res_prod = engine_prod.decide(agents=agents, input_data=x_test, context={})
    assert res_prod.metadata["module_status"]["gnn"]["executed"] is True


# ── Phase 4: Type Consistency & Dashboard Compatibility Tests ───────────────────

def test_agent_predictions_type_consistency_across_all_graph_configurations(pipeline_data_and_agents):
    """
    Phase 4: Verify DecisionResult.agent_predictions is strictly Dict[str, float]
    across all graph modes and fallbacks:
      1. graph=None -> every value is float
      2. NoGraph() -> every value is float
      3. fitted StandardGAT -> every value is float
      4. fitted EpistemicGAT -> every value is float
      5. production + untrained learned GAT fallback -> every value is float
      6. all probabilities are finite and inside [0, 1]
    """
    data = pipeline_data_and_agents
    agents = data["agents"]
    X_train, y_train = data["X_train"], data["y_train"]
    x_test = data["X_test"][0]

    # Fit StandardGAT & EpistemicGAT strictly on train split
    std_gat = StandardGAT(input_dim=3, hidden_dim=8, output_dim=1)
    std_gat.fit(agents, X_train, y_train, epochs=15, seed=42)

    epi_gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1, use_epistemic_prior=True)
    epi_gat.fit(agents, X_train, y_train, epochs=15, seed=42)

    untrained_gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)

    configurations = [
        ("graph_none", None, "production"),
        ("no_graph", NoGraph(), "production"),
        ("fitted_standard_gat", std_gat, "production"),
        ("fitted_epistemic_gat", epi_gat, "production"),
        ("untrained_gat_production_fallback", untrained_gat, "production"),
        ("graph_none_research", None, "research"),
        ("no_graph_research", NoGraph(), "research"),
        ("fitted_standard_gat_research", std_gat, "research"),
        ("fitted_epistemic_gat_research", epi_gat, "research"),
    ]

    for name, graph_inst, mode in configurations:
        engine = DecisionEngine(
            config=CognixConfig(),
            graph=graph_inst,
            belief=EpistemicWeightedFusion(),
            mode=mode,
        )
        res = engine.decide(agents=agents, input_data=x_test, context={"mode": name})

        assert isinstance(res.agent_predictions, dict), f"{name}: agent_predictions must be dict"
        assert len(res.agent_predictions) == len(agents), f"{name}: missing agents in agent_predictions"

        for a in agents:
            val = res.agent_predictions.get(a.agent_id)
            assert isinstance(val, float), (
                f"{name}: agent_predictions[{a.agent_id}] is {type(val).__name__}, expected float"
            )
            assert np.isfinite(val), f"{name}: agent_predictions[{a.agent_id}] is not finite: {val}"
            assert 0.0 <= val <= 1.0, f"{name}: agent_predictions[{a.agent_id}] = {val} not in [0, 1]"


def test_dashboard_consumes_normalized_agent_predictions(pipeline_data_and_agents):
    """
    Verify dashboard/app.py logic continues to consume normalized agent_predictions correctly.
    """
    data = pipeline_data_and_agents
    agents = data["agents"]
    x_test = data["X_test"][0]

    engine = DecisionEngine(
        config=CognixConfig(),
        attribution=EpistemicShapley(),
        mode="production",
    )
    result = engine.decide(agents=agents, input_data=x_test, context={})

    # Emulate dashboard payload extraction logic from dashboard/app.py
    for a in agents:
        pred_obj = result.agent_predictions.get(a.agent_id, 0.5)
        if hasattr(pred_obj, 'value'):
            pred = float(pred_obj.value)
        elif hasattr(pred_obj, 'prediction'):
            pred = float(pred_obj.prediction)
        else:
            pred = float(pred_obj)
        assert isinstance(pred, float)
        assert 0.0 <= pred <= 1.0


# ── Phase 5: Documented Non-Blocking Discrepancies ──────────────────────────────
# 1. Benchmark Execution Path:
#    The universal benchmark (experiments/universal_evaluation/run_universal_benchmark.py)
#    evaluates collective classification/ECE/coverage metrics via a standalone _run_single_sample
#    loop that bypasses DecisionEngine and EpistemicShapley. This integration test suite
#    validates the full DecisionEngine path (including risk assessment, decision outcome, and Shapley).
# 2. EpistemicWeightedFusion Epsilon:
#    The universal benchmark explicitly initializes EpistemicWeightedFusion(eps=1e-8), whereas the
#    default constructor in cognix/belief/fusion.py defaults to eps=1e-6. Both values are numerically
#    sound and preserve the canonical inverse-epistemic trust weighting mechanism.


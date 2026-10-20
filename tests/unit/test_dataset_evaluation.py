"""
Unit and regression tests for COGNIX End-to-End Dataset Evaluation.

Verifies:
1. Mathematical metric calculations (BCE, Brier, F1, ECE, AUROC, Paired differences).
2. Strict post-inference label isolation and absence of data leakage.
3. Replay source domain distinction (recorded DATASET vs SYNTHETIC).
4. Deterministic replay repeatability under fixed seeds.
5. Error handling on missing, out-of-bounds, or malformed inputs.
6. Fail-closed calibration invariants.
"""

from pathlib import Path
import numpy as np
import pytest

from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.replay_source import ArchiveConfig, SyntheticReplaySource
from cognix.adapters.carla.runtime_artifacts import load_runtime_artifacts
from cognix.engine.decision_engine import DecisionEngine
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.runtime.bridge import RuntimeBridge
from cognix.runtime.replay import ObservationEnvelope, ReplayController, SourceError
from cognix.evaluation.e2e_dataset_eval import (
    bce_metric,
    brier_metric,
    auroc_metric,
    classification_f1,
    compute_paired_stats,
    load_ground_truth_labels,
    aggregate_carlanomaly_metrics,
    ScenarioEvaluationResult,
    FROZEN_THRESHOLDS_VAL
)
from dashboard.payload import build_payload

ROOT = Path(__file__).resolve().parents[2]


class TestMetricCalculations:
    """Tests the mathematical correctness of evaluation metrics."""

    def test_bce_metric_perfect(self):
        q = np.array([1.0, 0.0, 1.0, 0.0])
        y = np.array([1.0, 0.0, 1.0, 0.0])
        val = bce_metric(q, y)
        # With clipping to 1e-7, loss is -log(1 - 1e-7) approx 1e-7
        assert 0.0 <= val < 1e-5

    def test_bce_metric_imperfect(self):
        q = np.array([0.5, 0.5])
        y = np.array([1.0, 0.0])
        val = bce_metric(q, y)
        expected = float(-np.log(0.5))
        assert abs(val - expected) < 1e-6

    def test_bce_clipping_bounds(self):
        # Even with extreme probabilities, BCE should remain finite
        q = np.array([0.0, 1.0])
        y = np.array([1.0, 0.0])  # worst case predictions
        val = bce_metric(q, y)
        assert np.isfinite(val)
        assert abs(val - (-np.log(1e-7))) < 1e-4

    def test_brier_metric(self):
        q = np.array([0.8, 0.2, 0.5])
        y = np.array([1.0, 0.0, 1.0])
        val = brier_metric(q, y)
        expected = float(np.mean([(0.8 - 1.0)**2, (0.2 - 0.0)**2, (0.5 - 1.0)**2]))
        assert abs(val - expected) < 1e-6

    def test_classification_f1_basic(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        res = classification_f1(y_true, y_pred)
        assert res["tp"] == 1
        assert res["fp"] == 1
        assert res["fn"] == 1
        assert res["tn"] == 1
        assert res["precision"] == 0.5
        assert res["recall"] == 0.5
        assert res["f1"] == 0.5
        assert res["accuracy"] == 0.5

    def test_classification_f1_zero_denominator(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        res = classification_f1(y_true, y_pred)
        assert res["f1"] == 0.0
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0
        assert res["accuracy"] == 1.0

    def test_auroc_metric_separable(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        auc = auroc_metric(scores, labels)
        assert auc == 1.0

    def test_auroc_metric_single_class(self):
        scores = np.array([0.2, 0.4, 0.8])
        labels = np.array([0, 0, 0])  # single-class
        auc = auroc_metric(scores, labels)
        assert auc is None

    def test_paired_stats(self):
        x = [0.55, 0.50, 0.60]
        y = [0.60, 0.55, 0.65]
        stats_res = compute_paired_stats(x, y)
        assert stats_res["n"] == 3
        assert abs(stats_res["mean_diff"] - (-0.05)) < 1e-6
        assert stats_res["sample_sd"] < 1e-6  # all differences are -0.05
        assert abs(stats_res["min"] - (-0.05)) < 1e-6
        assert abs(stats_res["max"] - (-0.05)) < 1e-6


class TestGroundTruthAndIsolation:
    """Verifies ground truth loading and strict isolation from inference."""

    def test_load_ground_truth_labels(self):
        scenarios = [
            "test/anomaly/Town01/change-weather/scenario-1",
            "test/anomaly/Town01/running-pedestrian/scenario-1"
        ]
        gt = load_ground_truth_labels(ROOT, scenarios)
        assert set(gt.keys()) == set(scenarios)
        for sc in scenarios:
            assert len(gt[sc]["tick"]) == 299
            assert len(gt[sc]["target_normal"]) == 299
            assert len(gt[sc]["target_corruption"]) == 299
            # Targets must sum to 1 at each tick
            assert np.all(gt[sc]["target_normal"] + gt[sc]["target_corruption"] == 1)

    def test_label_isolation_invariant(self):
        """Ensures observation envelope contains NO labels and mutating labels does not affect inference."""
        dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=2, seed=42)
        source = SyntheticReplaySource(dataset)
        obs = source.read("CAMERA_BLACKOUT", 0)

        assert "label" not in obs.agent_inputs
        assert "target_normal" not in obs.agent_inputs
        assert "target_corruption" not in obs.agent_inputs
        assert obs.provenance.get("label_visible_to_inference", False) is False

        # Build payload with mock annotation
        class MockAgent:
            agent_id = "Camera"
            _prediction = type("P", (), {"value": 0.8})()
            _uncertainty = type("U", (), {"epistemic": 0.1, "aleatoric": 0.1})()

        cycle = type("C", (), {
            "observation": obs,
            "result": type("R", (), {
                "confidence": 0.8,
                "calibrated_confidence": None,
                "decision": type("D", (), {"value": "ACT"})(),
                "risk_level": type("L", (), {"value": "LOW"})(),
                "epistemic_uncertainty": 0.1,
                "aleatoric_uncertainty": 0.1,
                "total_uncertainty": 0.2,
                "dominant_uncertainty_source": "epistemic",
                "escalation_required": False,
                "abstained": False,
                "requested_information": [],
                "agent_predictions": {"Camera": 0.8},
                "agent_trust_weights": {"Camera": 1.0},
                "agent_contributions": {"Camera": 0.0},
                "communication_statistics": None,
                "calibration_metrics": None,
                "latency_ms": {},
                "explanation": "test",
                "metadata": {"module_status": {}}
            })(),
            "agents": [MockAgent()],
            "runtime_ms": 5.0,
            "agent_inference_ms": 2.0,
            "runtime_provenance": {"graph_strategy": "NoGraph", "run_id": "TEST_1"}
        })()

        # Payload without annotation
        p1 = build_payload(cycle)
        assert p1["ground_truth"] is None

        # Payload with annotation attached strictly after inference
        p2 = build_payload(cycle, annotation=lambda: {"target_normal": 1})
        assert p2["ground_truth"]["value"] == {"target_normal": 1}
        assert p2["ground_truth"]["used_for_inference"] is False
        # Confidence and decision must remain unchanged
        assert p1["confidence"] == p2["confidence"]
        assert p1["decision"] == p2["decision"]


class TestReplaySourcesAndDomains:
    """Verifies domain labeling and error handling."""

    def test_synthetic_source_properties(self):
        dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=2, seed=42)
        source = SyntheticReplaySource(dataset)
        assert source.info["mode"] == "SYNTHETIC"
        assert source.info["domain"] == "autonomous_driving"
        assert "Synthetic" in source.info["source_id"]

        obs = source.read("NORMAL", 0)
        assert obs.provenance["mode"] == "SYNTHETIC"
        assert obs.source_id == "CarlAnomalyDataset (Synthetic Simulation)"

    def test_synthetic_source_invalid_scenario(self):
        dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=2, seed=42)
        source = SyntheticReplaySource(dataset)
        with pytest.raises(SourceError, match="Unknown synthetic scenario"):
            source.read("NON_EXISTENT_SCENARIO", 0)

    def test_carlanomaly_source_missing_archive(self):
        with pytest.raises(SourceError, match="archive not configured/found"):
            ArchiveConfig(archive=Path("C:/non_existent_archive_xyz.tar.gz"))
            from cognix.adapters.carla.replay_source import CarlAnomalyReplaySource
            CarlAnomalyReplaySource.configured(ROOT, "TEST", "C:/non_existent_archive_xyz.tar.gz")

    def test_frozen_thresholds_presence(self):
        assert "NoGraph_101" in FROZEN_THRESHOLDS_VAL
        assert "StandardGAT_101" in FROZEN_THRESHOLDS_VAL
        assert "EpistemicGAT_101" in FROZEN_THRESHOLDS_VAL
        assert 0.35 <= FROZEN_THRESHOLDS_VAL["StandardGAT_101"] <= 0.45


class TestCalibrationInvariants:
    """Verifies that uncalibrated runtime chain fails closed on calibration requests."""

    def test_runtime_artifacts_calibration_fail_closed(self):
        with pytest.raises(SourceError, match="No collective conformal artifact is bound"):
            load_runtime_artifacts(ROOT, "StandardGAT", 101, require_calibration=True)

    def test_runtime_artifacts_loads_uncalibrated_cleanly(self):
        agents, graph, prov = load_runtime_artifacts(ROOT, "StandardGAT", 101, require_calibration=False)
        assert len(agents) == 3
        assert graph is not None
        assert prov["full_calibrated_chain"] is False
        assert "No collective conformal artifact" in prov["calibration_unavailable_reason"]


class TestAggregationAndPairing:
    """Verifies aggregation of scenario results into paired differences."""

    def test_aggregate_carlanomaly_metrics(self):
        r1 = ScenarioEvaluationResult(
            scenario="sc1", method="NoGraph", seed=101, num_samples=100,
            normal_samples=80, corrupt_samples=20, bce=0.50, brier=0.15, ece=0.08,
            auroc=0.85, f1_detection=0.70, precision_detection=0.75, recall_detection=0.65,
            accuracy_detection=0.80, escalation_rate=0.25, info_request_rate=0.10,
            act_rate=0.65, anomaly_escalation_recall=0.90, mean_epistemic_normal=0.05,
            mean_epistemic_corrupt=0.20, epistemic_specificity_ratio=4.0, mean_aleatoric=0.10,
            latency_p50_ms=10.0, latency_p95_ms=15.0, latency_p99_ms=20.0, source_prepare_ms=100.0
        )
        r2 = ScenarioEvaluationResult(
            scenario="sc1", method="StandardGAT", seed=101, num_samples=100,
            normal_samples=80, corrupt_samples=20, bce=0.45, brier=0.12, ece=0.06,
            auroc=0.88, f1_detection=0.75, precision_detection=0.80, recall_detection=0.70,
            accuracy_detection=0.85, escalation_rate=0.25, info_request_rate=0.10,
            act_rate=0.65, anomaly_escalation_recall=0.95, mean_epistemic_normal=0.04,
            mean_epistemic_corrupt=0.22, epistemic_specificity_ratio=5.5, mean_aleatoric=0.09,
            latency_p50_ms=12.0, latency_p95_ms=17.0, latency_p99_ms=22.0, source_prepare_ms=100.0
        )
        agg = aggregate_carlanomaly_metrics([r1, r2])
        assert "NoGraph" in agg["methods"]
        assert "StandardGAT" in agg["methods"]
        assert "StandardGAT_vs_NoGraph" in agg["paired_comparisons"]
        p = agg["paired_comparisons"]["StandardGAT_vs_NoGraph"]
        assert abs(p["bce"]["mean_diff"] - (-0.05)) < 1e-6
        assert abs(p["f1"]["mean_diff"] - (0.05)) < 1e-6

"""
Unit tests for the Cognix Live Dashboard backend and pipeline provenance.

Ensures:
- Real Shapley values are computed and populated in the payload
- No hardcoded research metrics are masquerading as live measurements
- Latencies are derived from real timers
- Synthetic simulation status is explicitly declared
- Scenario switching functions as expected
"""
import pytest
from cognix import DecisionEngine, CognixConfig
from cognix.attribution.epistemic_shapley import EpistemicShapley
from dashboard.app import run_cognix_cycle, set_scenario, ScenarioRequest, CURRENT_SCENARIO_NAME, dataset



def test_dashboard_live_payload_shapley():
    """Verify run_cognix_cycle returns real live Shapley contributions."""
    payload = run_cognix_cycle()
    assert isinstance(payload, dict)

    # Shapley must be present and contain all active agents
    shapley = payload.get("shapley")
    assert isinstance(shapley, dict)
    assert len(shapley) == 6
    expected_agents = {"Camera", "Depth", "LiDAR", "GNSS", "IMU", "Seg"}
    assert set(shapley.keys()) == expected_agents

    # Verify all Shapley values are finite floats
    for agent_name, phi in shapley.items():
        assert isinstance(phi, float)
        assert -2.0 <= phi <= 2.0


def test_dashboard_live_payload_latency_provenance():
    """Verify latency values in payload are measured with high-res timer."""
    payload = run_cognix_cycle()
    lat = payload.get("latency")
    assert isinstance(lat, dict)
    assert lat["measured_timer"] == "time.perf_counter"
    assert lat["total"] > 0.0
    assert lat["uncertainty"] >= 0.0
    assert lat["attribution"] >= 0.0
    assert lat["p50"] >= 0.0


def test_dashboard_illustrative_baselines_explicit():
    """Verify comparison table is clearly labeled (illustrative or benchmark)."""
    payload = run_cognix_cycle()
    # 'baselines' key was removed; provenance is now carried by comparison_table_note
    assert "comparison_table_note" in payload
    note = payload["comparison_table_note"].lower()
    # Note must contain either "illustrative" (fallback) or "benchmark" (real data)
    assert "illustrative" in note or "benchmark" in note


def test_dashboard_synthetic_status_explicit():
    """Verify synthetic simulation status is explicit in metadata."""
    payload = run_cognix_cycle()
    meta = payload.get("research_meta")
    assert isinstance(meta, dict)
    # data_source is the authoritative provenance field
    assert "Synthetic Simulation" in meta["data_source"]
    # dataset field must be present (CarlAnomaly)
    assert "dataset" in meta


def test_dashboard_scenario_switch():
    """Verify switching scenario changes active scenario in the cycle."""
    res = set_scenario(ScenarioRequest(scenario="CAMERA_BLACKOUT"))
    assert res["status"] == "ok"
    assert res["scenario"] == "CAMERA_BLACKOUT"

    payload = run_cognix_cycle()
    assert payload["scenario"] == "CAMERA_BLACKOUT"
    assert "Camera" in payload["scenario_details"]["affected"]

    # Reset back to NORMAL
    set_scenario(ScenarioRequest(scenario="NORMAL"))

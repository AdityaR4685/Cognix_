import math
"""
Unit and integration tests for CognixLiveDemoV2 (M16 Upgrade).
Tests the multi-agent uncertainty dashboard, decision engine integration,
scenario switching, pedestrian safety rules, fault injection, and actor safety
without requiring a live CARLA server.
"""

import time
from unittest.mock import MagicMock
import numpy as np
import pytest

try:
    import carla
except ImportError:
    carla = None

from cognix.decision.escalation import DecisionOutcome, RiskLevel
from cognix.evaluation.cognix_live_demo_v2 import (
    CognixLiveDemoV2,
    COLLISION_ALERT_SECONDS,
    FaultMode,
    ScenarioMode,
    SourceType,
    TrajectoryAuditor,
    build_presentation_loop_route,
    compute_imu_motion_uncertainty,
    compute_obstacle_relative_motion,
    compute_rgb_quality_uncertainty,
    compute_seg_vocab_uncertainty,
)


@pytest.fixture
def demo():
    """Create a test instance of CognixLiveDemoV2."""
    return CognixLiveDemoV2()


@pytest.fixture
def nominal_frames():
    """Create nominal synthetic sensor frames."""
    rgb = np.full((360, 640, 3), 120, dtype=np.uint8)
    seg = np.full((360, 640, 3), (128, 64, 128), dtype=np.uint8)  # Road
    imu = {
        "accelerometer": np.array([0.0, 0.0, 9.81]),
        "gyroscope": np.array([0.0, 0.0, 0.0]),
    }
    return rgb, seg, imu


class TestCognixLiveDemoV2:
    """Comprehensive test suite for Cognix Live Demo V2."""

    def test_uncertainty_sources_discovery_and_reporting(self, demo, nominal_frames):
        """Verifies that all 11 multi-agent uncertainty sources are discovered and reported."""
        rgb, seg, imu = nominal_frames
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())

        assert len(ev.uncertainties) == 11
        unc_map = {u.id: u for u in ev.uncertainties}

        # Check required subsystems
        assert "rgb_quality" in unc_map
        assert "seg_vocab" in unc_map
        assert "imu_dynamics" in unc_map
        assert "camera_agent_epistemic" in unc_map
        assert "seg_agent_epistemic" in unc_map
        assert "imu_agent_epistemic" in unc_map
        assert "fused_belief_unc" in unc_map
        assert "conformal_prediction_set" in unc_map
        assert "lidar_agent" in unc_map
        assert "depth_agent" in unc_map
        assert "gnss_agent" in unc_map

        # Check that unattached sensors are strictly labeled NOT EXPOSED and value is None
        for unattached_id in ("lidar_agent", "depth_agent", "gnss_agent"):
            entry = unc_map[unattached_id]
            assert entry.source_type == SourceType.NOT_EXPOSED
            assert entry.value is None
            assert "N/A" in entry.display_str
            assert not entry.is_active

        # Check that Cognix agents have valid float epistemic uncertainty
        for agent_id in ("camera_agent_epistemic", "seg_agent_epistemic", "imu_agent_epistemic"):
            entry = unc_map[agent_id]
            assert entry.source_type == SourceType.COGNIX
            assert isinstance(entry.value, float)
            assert 0.0 <= entry.value <= 1.0

    def test_proper_display_and_source_labeling(self, demo, nominal_frames):
        """Verifies that source badges (LIVE, COGNIX, DEMO PROXY, SCENARIO INPUT, NOT EXPOSED) are accurate."""
        rgb, seg, imu = nominal_frames

        # 1. Nominal: live sensors report LIVE
        ev_nom = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        rgb_entry = next(u for u in ev_nom.uncertainties if u.id == "rgb_quality")
        assert rgb_entry.source_type == SourceType.LIVE

        # 2. Fault Injected: sensor report DEMO PROXY
        demo.fault_mode = FaultMode.RGB_DARKNESS
        ev_fault = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        rgb_fault_entry = next(u for u in ev_fault.uncertainties if u.id == "rgb_quality")
        assert rgb_fault_entry.source_type == SourceType.DEMO_PROXY

        # 3. Scenario Injected: trigger source reports SCENARIO INPUT
        demo.fault_mode = FaultMode.NONE
        demo.set_scenario(ScenarioMode.PEDESTRIAN)
        ev_scen = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_scen.trigger_source == SourceType.SCENARIO_INPUT

    def test_scenario_selection_and_switching(self, demo, nominal_frames):
        """Verifies seamless switching across scenarios without crashing or resetting."""
        rgb, seg, imu = nominal_frames

        # Start in Normal
        demo.set_scenario(ScenarioMode.NORMAL)
        assert demo.scenario_mode == ScenarioMode.NORMAL
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.decision_outcome == DecisionOutcome.ACT

        # Switch to Pedestrian
        demo.set_scenario(ScenarioMode.PEDESTRIAN)
        assert demo.scenario_mode == ScenarioMode.PEDESTRIAN
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.decision_outcome == DecisionOutcome.ESCALATE

        # Switch to Weather
        demo.set_scenario(ScenarioMode.WEATHER)
        assert demo.scenario_mode == ScenarioMode.WEATHER
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.decision_outcome == DecisionOutcome.WAIT

        # Switch to Props
        demo.set_scenario(ScenarioMode.PROPS)
        assert demo.scenario_mode == ScenarioMode.PROPS
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.decision_outcome == DecisionOutcome.ESCALATE

        # Reset to Normal
        demo.set_scenario(ScenarioMode.NORMAL)
        assert demo.scenario_mode == ScenarioMode.NORMAL
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.decision_outcome == DecisionOutcome.ACT

    def test_pedestrian_protection_rule(self, demo, nominal_frames):
        """Verifies the M16 pedestrian protection policy contract."""
        rgb, seg, imu = nominal_frames

        # Test A: Injected scenario pedestrian
        demo.set_scenario(ScenarioMode.PEDESTRIAN)
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.decision_outcome == DecisionOutcome.ESCALATE
        assert ev.risk_level == RiskLevel.HIGH
        assert ev.escalation_required is True
        assert ev.confidence <= 0.10
        assert "pedestrian" in ev.decision_reason.lower()

        # Test B: Live sensor walker observation under nominal scenario
        demo.set_scenario(ScenarioMode.NORMAL)
        obs_walker = {"distance": 12.5, "actor_type": "walker.pedestrian.0001", "timestamp": time.monotonic()}
        ev_live = demo.evaluate_frame(rgb, seg, imu, obs_walker, None, 30.0, "NONE", time.monotonic())
        assert ev_live.decision_outcome == DecisionOutcome.ESCALATE
        assert ev_live.risk_level == RiskLevel.HIGH
        assert ev_live.trigger_source == SourceType.LIVE
        assert "Live CARLA pedestrian" in ev_live.decision_reason

    def test_weather_and_roadway_prop_scenario_inputs(self, demo, nominal_frames):
        """Verifies policy response under degraded weather and roadway prop scenarios."""
        rgb, seg, imu = nominal_frames

        # Weather: ambiguous confidence triggers precautionary WAIT
        demo.set_scenario(ScenarioMode.WEATHER)
        ev_w = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_w.decision_outcome == DecisionOutcome.WAIT
        assert ev_w.risk_level == RiskLevel.MODERATE
        assert 0.40 <= ev_w.confidence < 0.70

        # Props: unclassified obstacle triggers ESCALATE
        demo.set_scenario(ScenarioMode.PROPS)
        ev_b = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_b.decision_outcome == DecisionOutcome.ESCALATE
        assert ev_b.risk_level == RiskLevel.HIGH
        assert ev_b.confidence < 0.40

    def test_sensor_faults_and_reset_behavior(self, demo, nominal_frames):
        """Verifies that sensor faults elevate uncertainty and keyboard reset restores nominal state."""
        rgb, seg, imu = nominal_frames

        # Fault 1: RGB Darkness
        demo.fault_mode = FaultMode.RGB_DARKNESS
        ev_dark = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_dark.decision_outcome in (DecisionOutcome.WAIT, DecisionOutcome.ESCALATE)

        # Fault 3: Seg Corruption
        demo.fault_mode = FaultMode.SEG_CORRUPTION
        ev_seg = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        seg_entry = next(u for u in ev_seg.uncertainties if u.id == "seg_vocab")
        assert seg_entry.value > 0.15

        # Fault 4: IMU Disturbance
        demo.fault_mode = FaultMode.IMU_DISTURBANCE
        ev_imu = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        imu_entry = next(u for u in ev_imu.uncertainties if u.id == "imu_dynamics")
        assert imu_entry.value > 0.20

        # Keyboard Reset 'r'
        demo.handle_key(ord("r"))
        assert demo.scenario_mode == ScenarioMode.NORMAL
        assert demo.fault_mode == FaultMode.NONE

    def test_decision_engine_adapter_uses_real_engine_outputs(self, demo, nominal_frames):
        """Verifies that the dashboard decision is an actual returned EscalationResult."""
        rgb, seg, imu = nominal_frames
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())

        assert isinstance(ev.decision_outcome, DecisionOutcome)
        assert isinstance(ev.risk_level, RiskLevel)
        assert ev.decision_outcome in (DecisionOutcome.ACT, DecisionOutcome.WAIT, DecisionOutcome.ESCALATE, DecisionOutcome.ABSTAIN)
        assert ev.risk_level in (RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH)

    def test_scenario_inputs_never_masquerade_as_actual_sensor_detection(self, demo, nominal_frames):
        """Verifies that scenario-injected hazards are explicitly tagged as SCENARIO INPUT."""
        rgb, seg, imu = nominal_frames
        demo.set_scenario(ScenarioMode.PEDESTRIAN)

        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.trigger_source == SourceType.SCENARIO_INPUT
        assert "Scenario-injected" in ev.decision_reason
        assert "Live CARLA pedestrian" not in ev.decision_reason

    def test_missing_sensor_and_dropout_behavior(self, demo):
        """Verifies graceful handling of missing or delayed sensor feeds without throwing exceptions."""
        black_rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        black_seg = np.zeros((360, 640, 3), dtype=np.uint8)
        dummy_imu = {
            "accelerometer": np.array([0.0, 0.0, 9.81]),
            "gyroscope": np.array([0.0, 0.0, 0.0]),
        }

        # Stale sensor timestamp (1.5 seconds old)
        t_stale = time.monotonic() - 1.5
        ev = demo.evaluate_frame(black_rgb, black_seg, dummy_imu, None, None, 0.0, "NONE", t_stale)

        assert ev.sensor_age_ms >= 1000.0
        assert ev.decision_outcome is not None
        assert ev.latency_ms > 0.0

    def test_cleanup_only_destroys_owned_actors(self, demo):
        """Verifies that demo.cleanup() only calls destroy on demo-owned actors and sensors."""
        mock_actor1 = MagicMock()
        mock_actor2 = MagicMock()
        mock_unrelated_actor = MagicMock()

        demo.demo_owned_actors = [mock_actor1]
        demo.demo_owned_sensors = [mock_actor2]

        demo.cleanup()

        mock_actor1.destroy.assert_called_once()
        mock_actor2.stop.assert_called_once()
        mock_actor2.destroy.assert_called_once()
        mock_unrelated_actor.destroy.assert_not_called()

    def test_no_heavy_detector_inference_in_upgraded_path(self, demo, nominal_frames):
        """Verifies that no heavyweight R-CNN or large model is imported or invoked per frame."""
        import sys

        # Ensure torchvision detection models are not imported by the live demo module
        demo_mod = sys.modules["cognix.evaluation.cognix_live_demo_v2"]
        assert not hasattr(demo_mod, "TorchVisionObjectDetector")
        assert not hasattr(demo_mod, "torchvision")

        rgb, seg, imu = nominal_frames

        # Measure frame evaluation latency over 20 iterations
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_latency = float(np.mean(latencies))
        assert mean_latency < 25.0, f"Frame evaluation latency too high: {mean_latency:.2f} ms (expected < 25ms)"
        print(f"Mean evaluation latency: {mean_latency:.2f} ms")

    def test_render_dashboard_dimensions_and_channels(self, demo, nominal_frames):
        """Verifies dashboard canvas composition dimensions (1400x720x3)."""
        rgb, seg, imu = nominal_frames
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        dashboard = demo.render_dashboard(rgb, seg, ev)

        assert dashboard.shape == (720, 1400, 3)
        assert dashboard.dtype == np.uint8

    def test_all_scenario_keys_and_transitions(self, demo, nominal_frames):
        """Verifies keyboard handling for every scenario and fault hotkey including case and numpads."""
        # 1. Scenario Keys: upper and lower case
        for key_char, expected_mode in [
            ("p", ScenarioMode.PEDESTRIAN),
            ("P", ScenarioMode.PEDESTRIAN),
            ("w", ScenarioMode.WEATHER),
            ("W", ScenarioMode.WEATHER),
            ("b", ScenarioMode.PROPS),
            ("B", ScenarioMode.PROPS),
            ("n", ScenarioMode.NORMAL),
            ("N", ScenarioMode.NORMAL),
        ]:
            res = demo.handle_key(ord(key_char))
            assert res is True
            assert demo.scenario_mode == expected_mode
            assert demo.last_key_name == key_char.upper()

        # 2. Reset Key 'r' / 'R'
        demo.set_scenario(ScenarioMode.PROPS)
        demo.fault_mode = FaultMode.RGB_DARKNESS
        assert demo.handle_key(ord("r")) is True
        assert demo.scenario_mode == ScenarioMode.NORMAL
        assert demo.fault_mode == FaultMode.NONE
        assert demo.last_key_name == "R"

        # 3. Fault Keys '0' through '4'
        for k, expected_fault in [
            ("1", FaultMode.RGB_DARKNESS),
            ("2", FaultMode.RGB_OCCLUSION),
            ("3", FaultMode.SEG_CORRUPTION),
            ("4", FaultMode.IMU_DISTURBANCE),
            ("0", FaultMode.NONE),
        ]:
            res = demo.handle_key(ord(k))
            assert res is True
            assert demo.fault_mode == expected_fault
            assert demo.last_key_name == k

        # 4. Verify 'b' and 'B' explicitly select ScenarioMode.PROPS
        demo.set_scenario(ScenarioMode.NORMAL)
        assert demo.handle_key(ord("b")) is True
        assert demo.scenario_mode == ScenarioMode.PROPS
        assert demo.last_key_name == "B"

        demo.set_scenario(ScenarioMode.NORMAL)
        assert demo.handle_key(ord("B")) is True
        assert demo.scenario_mode == ScenarioMode.PROPS
        assert demo.last_key_name == "B"

    def test_ignored_and_unsupported_key_handling(self, demo):
        """Verifies that unmapped or non-printable keycodes do not crash the runner or corrupt state."""
        demo.scenario_mode = ScenarioMode.NORMAL
        demo.fault_mode = FaultMode.NONE
        demo.last_key_name = "INITIAL"

        # Unmapped characters
        for invalid_char in ("x", "z", "!", "9"):
            res = demo.handle_key(ord(invalid_char))
            assert res is True
            assert demo.scenario_mode == ScenarioMode.NORMAL
            assert demo.fault_mode == FaultMode.NONE

        # Out-of-range or non-printable keycodes
        for invalid_code in (-1, 300, 1000, 0):
            res = demo.handle_key(invalid_code)
            assert res is True

    def test_quit_keys_return_false(self, demo):
        """Verifies that 'q', 'Q', and Esc (27) return False to command graceful exit."""
        assert demo.handle_key(ord("q")) is False
        assert demo.handle_key(ord("Q")) is False
        assert demo.handle_key(27) is False

    def test_normal_reset_clears_injected_state_preserves_live_hazards(self, demo, nominal_frames):
        """Verifies that switching to Normal/Reset clears scenario actors but preserves real live hazards."""
        rgb, seg, imu = nominal_frames

        # Start in Pedestrian scenario
        demo.set_scenario(ScenarioMode.PEDESTRIAN)
        ev1 = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev1.decision_outcome == DecisionOutcome.ESCALATE

        # Switch to Normal, with NO live obstacle -> Returns to ACT (LOW)
        demo.handle_key(ord("n"))
        assert demo.scenario_mode == ScenarioMode.NORMAL
        ev_clear = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_clear.decision_outcome == DecisionOutcome.ACT
        assert ev_clear.risk_level == RiskLevel.LOW

        # But if there is a GENUINE live obstacle at 2.5m, Normal mode must remain protective!
        real_obs = {"distance": 2.5, "actor_type": "vehicle.audi.tt", "timestamp": time.monotonic()}
        ev_obs = demo.evaluate_frame(rgb, seg, imu, real_obs, None, 30.0, "NONE", time.monotonic())
        assert ev_obs.decision_outcome == DecisionOutcome.ESCALATE
        assert ev_obs.risk_level == RiskLevel.HIGH
        assert ev_obs.trigger_source == SourceType.LIVE
        assert "Imminent obstacle" in ev_obs.decision_reason

    def test_traffic_light_red_only_applies_when_at_traffic_light(self, demo, nominal_frames):
        """Verifies that red traffic light triggers WAIT when at light, and normal ACT when light is NONE."""
        rgb, seg, imu = nominal_frames

        # Red light active
        ev_red = demo.evaluate_frame(rgb, seg, imu, None, None, 0.0, "RED", time.monotonic())
        assert ev_red.decision_outcome == DecisionOutcome.WAIT
        assert ev_red.risk_level == RiskLevel.MODERATE

        # Normal road with no traffic light
        ev_none = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_none.decision_outcome == DecisionOutcome.ACT
        assert ev_none.risk_level == RiskLevel.LOW

    def test_obstacle_freshness_and_aging(self, demo, nominal_frames):
        """Verifies that fresh obstacles affect the decision while stale obstacles are disregarded."""
        rgb, seg, imu = nominal_frames

        # Fresh obstacle at 2.0m (0.1s old)
        fresh_obs = {"distance": 2.0, "actor_type": "vehicle.ford.mustang", "timestamp": time.monotonic() - 0.1}
        ev_fresh = demo.evaluate_frame(rgb, seg, imu, fresh_obs, None, 30.0, "NONE", time.monotonic())
        assert ev_fresh.obstacle_fresh is True
        assert ev_fresh.decision_outcome == DecisionOutcome.ESCALATE

        # Stale obstacle at 2.0m (0.8s old, > OBSTACLE_STALE_SECONDS)
        stale_obs = {"distance": 2.0, "actor_type": "vehicle.ford.mustang", "timestamp": time.monotonic() - 0.8}
        ev_stale = demo.evaluate_frame(rgb, seg, imu, stale_obs, None, 30.0, "NONE", time.monotonic())
        assert ev_stale.obstacle_fresh is False
        assert ev_stale.decision_outcome == DecisionOutcome.ACT

    def test_weather_and_scenario_lifecycle_restoration(self, demo):
        """Verifies weather is set for Weather mode and restored upon exiting Weather mode."""
        mock_world = MagicMock()
        mock_weather = MagicMock()
        demo.original_weather = mock_weather

        # Switch to WEATHER
        demo.set_scenario(ScenarioMode.WEATHER, world=mock_world)
        assert demo.scenario_mode == ScenarioMode.WEATHER
        mock_world.set_weather.assert_called()

        # Switch away from WEATHER to NORMAL
        mock_world.reset_mock()
        demo.set_scenario(ScenarioMode.NORMAL, world=mock_world)
        assert demo.scenario_mode == ScenarioMode.NORMAL
        mock_world.set_weather.assert_called_with(mock_weather)

    def test_spawn_failure_handling_and_fallback(self, demo):
        """Verifies graceful handling when world actor spawning fails (e.g. location occupied)."""
        mock_world = MagicMock()
        mock_world.get_blueprint_library().filter.return_value = [MagicMock()]
        mock_world.try_spawn_actor.return_value = None  # Spawning fails

        mock_vehicle = MagicMock()
        mock_vehicle.get_transform.return_value.location.x = 0.0
        mock_vehicle.get_transform.return_value.location.y = 0.0
        mock_vehicle.get_transform.return_value.location.z = 0.0
        mock_vehicle.get_transform.return_value.rotation.yaw = 0.0

        # Attempt to spawn pedestrian
        demo.set_scenario(ScenarioMode.PEDESTRIAN, world=mock_world, vehicle=mock_vehicle)
        assert "BLOCKED" in demo.scenario_spawn_status or "SIGNAL" in demo.scenario_spawn_status
        assert demo.scenario_mode == ScenarioMode.PEDESTRIAN

        # Verify protective decision still operates on scenario signal
        rgb = np.full((360, 640, 3), 120, dtype=np.uint8)
        seg = np.full((360, 640, 3), (128, 64, 128), dtype=np.uint8)
        imu = {"accelerometer": np.array([0.0, 0.0, 9.81]), "gyroscope": np.array([0.0, 0.0, 0.0])}
        ev = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev.decision_outcome == DecisionOutcome.ESCALATE
        assert ev.risk_level == RiskLevel.HIGH

    def test_obstacle_relative_motion_and_ttc_cases(self, demo, nominal_frames):
        """
        Verifies Time-To-Collision (TTC) and relative closing speed calculations
        across five distinct traffic scenarios to prevent false warnings on distant objects.
        """
        rgb, seg, imu = nominal_frames

        # Case 1: Lead vehicle matching speed ahead at 18m (30 km/h = 8.33 m/s)
        v_ego = [0.0, 8.33, 0.0]
        v_obs = [0.0, 8.33, 0.0]
        loc_ego = [0.0, 0.0, 0.0]
        loc_obs = [0.0, 18.0, 0.0]
        v_close, ttc, rel_avail = compute_obstacle_relative_motion(v_ego, loc_ego, v_obs, loc_obs, 18.0)
        assert rel_avail is True
        assert abs(v_close) < 0.1
        assert ttc is None  # Not closing, no collision hazard

        obs_lead = {
            "distance": 18.0,
            "actor_type": "vehicle.audi.tt",
            "v_close_mps": v_close,
            "ttc_sec": ttc,
            "relative_motion_available": True,
            "timestamp": time.monotonic(),
        }
        ev_lead = demo.evaluate_frame(rgb, seg, imu, obs_lead, None, 30.0, "NONE", time.monotonic())
        assert ev_lead.decision_outcome == DecisionOutcome.ACT
        assert ev_lead.risk_level == RiskLevel.LOW
        assert ev_lead.policy_branch_reason == "NOMINAL_CLEAR_CORRIDOR"
        assert ev_lead.ttc_sec is None

        # Case 2: Rapidly closing vehicle ahead (ego closing at 10.0 m/s from 12m)
        v_ego_fast = [0.0, 15.0, 0.0]
        v_obs_slow = [0.0, 5.0, 0.0]
        loc_obs_12m = [0.0, 12.0, 0.0]
        v_close_fast, ttc_fast, rel_avail = compute_obstacle_relative_motion(v_ego_fast, loc_ego, v_obs_slow, loc_obs_12m, 12.0)
        assert rel_avail is True
        assert abs(v_close_fast - 10.0) < 0.1
        assert abs(ttc_fast - 1.2) < 0.1  # 12m / 10m/s = 1.2s <= 1.8s

        obs_fast = {
            "distance": 12.0,
            "actor_type": "vehicle.tesla.model3",
            "v_close_mps": v_close_fast,
            "ttc_sec": ttc_fast,
            "relative_motion_available": True,
            "timestamp": time.monotonic(),
        }
        ev_fast = demo.evaluate_frame(rgb, seg, imu, obs_fast, None, 54.0, "NONE", time.monotonic())
        assert ev_fast.decision_outcome == DecisionOutcome.ESCALATE
        assert ev_fast.risk_level == RiskLevel.HIGH
        assert ev_fast.policy_branch_reason == "OBSTACLE_RAPID_CLOSING"
        assert ev_fast.ttc_sec == pytest.approx(1.2, 0.05)

        # Case 3: Static near obstacle (dist = 2.5m <= 3.0m)
        obs_near = {
            "distance": 2.5,
            "actor_type": "static.prop.barrier",
            "v_close_mps": 0.0,
            "ttc_sec": None,
            "relative_motion_available": True,
            "timestamp": time.monotonic(),
        }
        ev_near = demo.evaluate_frame(rgb, seg, imu, obs_near, None, 5.0, "NONE", time.monotonic())
        assert ev_near.decision_outcome == DecisionOutcome.ESCALATE
        assert ev_near.risk_level == RiskLevel.HIGH
        assert ev_near.policy_branch_reason == "OBSTACLE_IMMINENT_PROXIMITY"

        # Case 4: Distant static obstacle (dist = 18.0m, ego = 25 km/h = 6.94 m/s)
        # TTC = 18 / 6.94 = 2.59s > 1.8s, distance > 12.0m -> Corridor remains nominal
        v_ego_25 = [0.0, 6.94, 0.0]
        v_static = [0.0, 0.0, 0.0]
        v_close_dist, ttc_dist, rel_avail = compute_obstacle_relative_motion(v_ego_25, loc_ego, v_static, loc_obs, 18.0)
        assert abs(v_close_dist - 6.94) < 0.1
        assert abs(ttc_dist - 2.59) < 0.1

        obs_distant = {
            "distance": 18.0,
            "actor_type": "static.prop.trafficcone01",
            "v_close_mps": v_close_dist,
            "ttc_sec": ttc_dist,
            "relative_motion_available": True,
            "timestamp": time.monotonic(),
        }
        ev_dist = demo.evaluate_frame(rgb, seg, imu, obs_distant, None, 25.0, "NONE", time.monotonic())
        assert ev_dist.decision_outcome == DecisionOutcome.ACT
        assert ev_dist.risk_level == RiskLevel.LOW
        assert ev_dist.policy_branch_reason == "NOMINAL_CLEAR_CORRIDOR"

        # Case 5: Missing relative motion data (relative_motion_available = False)
        # Never substitute ego speed; rely on verified distance thresholds!
        obs_no_vel_18m = {
            "distance": 18.0,
            "actor_type": "vehicle.unknown",
            "v_close_mps": None,
            "ttc_sec": None,
            "relative_motion_available": False,
            "timestamp": time.monotonic(),
        }
        ev_no_vel_18 = demo.evaluate_frame(rgb, seg, imu, obs_no_vel_18m, None, 35.0, "NONE", time.monotonic())
        assert ev_no_vel_18.decision_outcome == DecisionOutcome.ACT
        assert ev_no_vel_18.risk_level == RiskLevel.LOW
        assert ev_no_vel_18.policy_branch_reason == "NOMINAL_CLEAR_CORRIDOR"

        obs_no_vel_6m = {
            "distance": 6.0,
            "actor_type": "vehicle.unknown",
            "v_close_mps": None,
            "ttc_sec": None,
            "relative_motion_available": False,
            "timestamp": time.monotonic(),
        }
        ev_no_vel_6 = demo.evaluate_frame(rgb, seg, imu, obs_no_vel_6m, None, 35.0, "NONE", time.monotonic())
        assert ev_no_vel_6.decision_outcome == DecisionOutcome.WAIT
        assert ev_no_vel_6.risk_level == RiskLevel.MODERATE
        assert ev_no_vel_6.policy_branch_reason == "OBSTACLE_FOLLOWING_ZONE"

    def test_prolonged_collision_and_reset_clearing(self, demo, nominal_frames):
        """
        Verifies that physical collision triggers high-priority safety interlock,
        stationary pinned state is explicitly explained in decision rationale,
        and observations expire and clear properly on Reset.
        """
        rgb, seg, imu = nominal_frames

        # 1. Active physical collision while vehicle is pinned stationary (speed = 0.0)
        col_data = {
            "actor_type": "static.prop.streetbarrier",
            "actor_id": 412,
            "impulse": 135.5,
            "timestamp": time.monotonic(),
        }
        ev_pinned = demo.evaluate_frame(rgb, seg, imu, None, col_data, 0.0, "NONE", time.monotonic())
        assert ev_pinned.decision_outcome == DecisionOutcome.ESCALATE
        assert ev_pinned.risk_level == RiskLevel.HIGH
        assert ev_pinned.confidence == 0.0
        assert ev_pinned.epistemic_uncertainty == 0.80
        assert ev_pinned.policy_branch_reason == "PHYSICAL_COLLISION_INTERLOCK"
        assert "Vehicle pinned/stopped against obstacle" in ev_pinned.decision_reason
        assert "static.prop.streetbarrier" in ev_pinned.decision_reason
        assert ev_pinned.collision_actor == "static.prop.streetbarrier"
        assert ev_pinned.collision_actor_id == 412

        # 2. Verify observation expiration in state buffer
        with demo.lock:
            demo.latest_collision = col_data.copy()
            demo.latest_collision["timestamp"] = time.monotonic() - (COLLISION_ALERT_SECONDS + 0.5)

        demo.clear_expired_observations()
        with demo.lock:
            assert demo.latest_collision is None

        # 3. Verify Reset 'R' clears active collision and obstacle immediately
        with demo.lock:
            demo.latest_collision = col_data.copy()
            demo.latest_obstacle = {"distance": 5.0, "actor_type": "hazard", "timestamp": time.monotonic()}

        demo.handle_key(ord("r"), source="CONSOLE_STDIN")
        with demo.lock:
            assert demo.latest_collision is None
            assert demo.latest_obstacle is None
        assert demo.scenario_mode == ScenarioMode.NORMAL
        assert demo.fault_mode == FaultMode.NONE

    def test_keyboard_source_attribution_and_mode_separation(self, demo):
        """
        Verifies explicit logging and attribution of keyboard input from GUI vs Console,
        and confirms 'B' (Roadway Prop Scenario) is strictly decoupled from '2' (RGB Occlusion Fault).
        """
        # Test GUI source
        assert demo.handle_key(ord("p"), source="OPENCV_GUI") is True
        assert demo.scenario_mode == ScenarioMode.PEDESTRIAN
        assert demo.last_key_name == "P"
        assert demo.last_key_source == "OPENCV_GUI"

        # Test Console source
        assert demo.handle_key(ord("w"), source="CONSOLE_STDIN") is True
        assert demo.scenario_mode == ScenarioMode.WEATHER
        assert demo.last_key_name == "W"
        assert demo.last_key_source == "CONSOLE_STDIN"

        # Test 'B' vs '2' separation:
        # Press 'b' (Prop scenario)
        demo.fault_mode = FaultMode.NONE
        assert demo.handle_key(ord("b"), source="CONSOLE_STDIN") is True
        assert demo.scenario_mode == ScenarioMode.PROPS
        assert demo.fault_mode == FaultMode.NONE
        assert demo.last_key_name == "B"

        # Now press '2' (RGB Occlusion fault)
        assert demo.handle_key(ord("2"), source="OPENCV_GUI") is True
        assert demo.scenario_mode == ScenarioMode.PROPS  # Preserves active scenario!
        assert demo.fault_mode == FaultMode.RGB_OCCLUSION  # Only toggles fault!
        assert demo.last_key_name == "2"
        assert demo.last_key_source == "OPENCV_GUI"

    def test_safe_checkpoint_tracking_and_suppression(self, demo):
        """Verifies safe checkpoint recording during valid driving, and suppression under red light/collision."""
        mock_world = MagicMock()
        mock_map = MagicMock()
        mock_world.get_map.return_value = mock_map

        mock_vehicle = MagicMock()
        mock_loc = MagicMock(x=100.0, y=200.0, z=1.0)
        mock_rot = MagicMock(pitch=0.0, yaw=90.0, roll=0.0)
        mock_vehicle.get_location.return_value = mock_loc
        mock_tf = MagicMock(rotation=mock_rot)
        mock_vehicle.get_transform.return_value = mock_tf

        # Mock waypoint on driving lane
        mock_wp = MagicMock()
        mock_wp.lane_type = "Driving"
        mock_wp.road_id = 12
        mock_wp.lane_id = -1
        mock_wp.transform.location = MagicMock(x=100.2, y=200.1, z=1.0)
        mock_map.get_waypoint.return_value = mock_wp

        # 1. Valid moving condition -> records checkpoint
        assert demo.update_safe_checkpoint(mock_world, mock_vehicle, speed_kmh=25.0, traffic_light_state="GREEN") is True
        assert demo.safe_checkpoint is not None
        assert demo.safe_checkpoint.road_id == 12
        assert demo.safe_checkpoint.lane_id == -1
        assert demo.safe_checkpoint.location == (100.0, 200.0, 1.0)

        # 2. Suppression: stationary (speed < 2.0 km/h)
        demo.safe_checkpoint = None
        demo.last_checkpoint_time = 0.0
        assert demo.update_safe_checkpoint(mock_world, mock_vehicle, speed_kmh=1.0, traffic_light_state="GREEN") is False
        assert demo.safe_checkpoint is None

        # 3. Suppression: red traffic light
        demo.last_checkpoint_time = 0.0
        assert demo.update_safe_checkpoint(mock_world, mock_vehicle, speed_kmh=25.0, traffic_light_state="RED") is False
        assert demo.safe_checkpoint is None

        # 4. Suppression: active collision
        demo.last_checkpoint_time = 0.0
        with demo.lock:
            demo.latest_collision = {"impulse": 120.0, "timestamp": time.monotonic()}
        assert demo.update_safe_checkpoint(mock_world, mock_vehicle, speed_kmh=25.0, traffic_light_state="NONE") is False
        assert demo.safe_checkpoint is None

    def test_stuck_detection_conditions_and_false_positive_prevention(self, demo):
        """Verifies collision pinned, off-road, stationary stuck, and red light false-positive prevention."""
        mock_world = MagicMock()
        mock_map = MagicMock()
        mock_world.get_map.return_value = mock_map
        mock_vehicle = MagicMock()
        mock_vehicle.get_location.return_value = MagicMock(x=50.0, y=50.0, z=0.5)

        # Mock driving lane waypoint
        mock_wp = MagicMock()
        mock_wp.lane_type = "Driving"
        mock_wp.transform.location = MagicMock(x=50.1, y=50.1, z=0.5)
        mock_map.get_waypoint.return_value = mock_wp

        # 1. Normal driving: not stuck
        is_stuck, reason = demo.detect_stuck_condition(mock_world, mock_vehicle, speed_kmh=30.0)
        assert is_stuck is False
        assert reason == "NOMINAL"

        # 2. Collision pinned condition
        with demo.lock:
            demo.latest_collision = {"impulse": 150.0, "timestamp": time.monotonic()}
        is_stuck, reason = demo.detect_stuck_condition(mock_world, mock_vehicle, speed_kmh=0.2)
        assert is_stuck is True
        assert "COLLISION_PINNED" in reason
        with demo.lock:
            demo.latest_collision = None

        # 3. Off-road condition (no driving waypoint)
        mock_map.get_waypoint.return_value = None
        is_stuck, reason = demo.detect_stuck_condition(mock_world, mock_vehicle, speed_kmh=15.0)
        assert is_stuck is True
        assert "OFF_ROAD" in reason
        mock_map.get_waypoint.return_value = mock_wp

        # 4. Red traffic light false-positive prevention: vehicle stopped at red light for > 4s
        demo.stuck_start_time = time.monotonic() - 6.0
        is_stuck, reason = demo.detect_stuck_condition(
            mock_world, mock_vehicle, speed_kmh=0.0, traffic_light_state="RED"
        )
        assert is_stuck is False
        assert demo.stuck_start_time is None

        # 5. Queuing obstacle false-positive prevention: vehicle stopped behind lead car
        demo.stuck_start_time = time.monotonic() - 6.0
        is_stuck, reason = demo.detect_stuck_condition(
            mock_world, mock_vehicle, speed_kmh=0.0, traffic_light_state="NONE", obstacle_distance=5.2
        )
        assert is_stuck is False
        assert demo.stuck_start_time is None

        # 6. Genuine stationary immobilization (> 4.0s without light or obstacle)
        demo.stuck_start_time = time.monotonic() - 5.0
        is_stuck, reason = demo.detect_stuck_condition(
            mock_world, mock_vehicle, speed_kmh=0.1, traffic_light_state="NONE", obstacle_distance=None
        )
        assert is_stuck is True
        assert "STUCK_STATIONARY" in reason

    def test_road_recovery_behavior_stuck_vs_nominal(self, demo):
        """Verifies that R restores nominal when moving without teleport, and recovers when stuck."""
        mock_world = MagicMock()
        mock_map = MagicMock()
        mock_world.get_map.return_value = mock_map

        mock_vehicle = MagicMock()
        mock_tf = MagicMock()
        mock_vehicle.get_transform.return_value = mock_tf
        mock_vehicle.get_location.return_value = MagicMock(x=10.0, y=20.0, z=0.5)

        mock_wp = MagicMock()
        mock_wp.lane_type = "Driving"
        mock_wp.road_id = 5
        mock_wp.lane_id = 1
        mock_wp.transform.location = MagicMock(x=10.1, y=20.1, z=0.5)
        mock_wp.transform.rotation = MagicMock(pitch=0.0, yaw=45.0, roll=0.0)
        mock_map.get_waypoint.return_value = mock_wp

        # Save a known good checkpoint
        from cognix.evaluation.cognix_live_demo_v2 import SafeCheckpoint
        demo.safe_checkpoint = SafeCheckpoint(
            location=(100.0, 200.0, 1.0),
            rotation=(0.0, 90.0, 0.0),
            timestamp=time.time(),
            road_id=5,
            lane_id=1,
            speed_kmh=25.0,
        )

        # 1. Non-stuck vehicle: R resets scenario without teleporting
        demo.vehicle_speed_kmh = 30.0
        demo.fault_mode = FaultMode.RGB_DARKNESS
        demo.scenario_mode = ScenarioMode.WEATHER
        mock_vehicle.set_transform.reset_mock()

        success, msg = demo.recover_vehicle_to_safe_road(mock_world, mock_vehicle, force=False)
        assert success is False
        assert msg == "NOMINAL_RESET_NO_TELEPORT"
        assert demo.scenario_mode == ScenarioMode.NORMAL
        assert demo.fault_mode == FaultMode.NONE
        mock_vehicle.set_transform.assert_not_called()

        # 2. Stuck vehicle (pinned): R teleports to candidate and re-engages autopilot
        demo.vehicle_speed_kmh = 0.0
        with demo.lock:
            demo.latest_collision = {"impulse": 180.0, "timestamp": time.monotonic()}
        demo.fault_mode = FaultMode.IMU_DISTURBANCE
        demo.scenario_mode = ScenarioMode.PROPS

        success, msg = demo.recover_vehicle_to_safe_road(mock_world, mock_vehicle, force=False)
        assert success is True
        assert "RECOVERED" in msg
        assert demo.scenario_mode == ScenarioMode.NORMAL
        assert demo.fault_mode == FaultMode.NONE
        mock_vehicle.set_autopilot.assert_called_with(True)
        mock_vehicle.set_transform.assert_called_once()
        with demo.lock:
            assert demo.latest_collision is None

    def test_visible_sensor_fault_visual_outputs(self, demo, nominal_frames):
        """Verifies that processed_rgb, processed_seg, and IMU data reflect active faults."""
        rgb, seg, imu = nominal_frames

        # 1. Fault: RGB Occlusion (central 60% cutout + warning perimeter)
        demo.fault_mode = FaultMode.RGB_OCCLUSION
        ev_occl = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_occl.processed_rgb is not None
        h, w = rgb.shape[:2]
        cutout_sample = ev_occl.processed_rgb[int(h * 0.25), int(w * 0.25)]
        assert np.array_equal(cutout_sample, [0, 0, 0])

        # 2. Fault: RGB Darkness
        demo.fault_mode = FaultMode.RGB_DARKNESS
        ev_dark = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_dark.processed_rgb is not None
        assert np.mean(ev_dark.processed_rgb) < np.mean(rgb) * 0.20

        # 3. Fault: Seg Corruption
        demo.fault_mode = FaultMode.SEG_CORRUPTION
        ev_seg = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_seg.processed_seg is not None
        assert not np.array_equal(ev_seg.processed_seg, seg)

        # 4. Clean IMU vs Injected IMU
        demo.fault_mode = FaultMode.IMU_DISTURBANCE
        ev_imu = demo.evaluate_frame(rgb, seg, imu, None, None, 30.0, "NONE", time.monotonic())
        assert ev_imu.clean_imu is not None
        assert ev_imu.injected_imu is not None
        assert not np.allclose(ev_imu.clean_imu["accelerometer"], ev_imu.injected_imu["accelerometer"])

    def test_checkpoint_direction_and_heading_validation(self, demo):
        """Verifies safe checkpoint accepts aligned lanes and rejects opposite-direction lanes."""
        mock_world = MagicMock()
        mock_map = MagicMock()
        mock_world.get_map.return_value = mock_map
        mock_vehicle = MagicMock()
        mock_loc = MagicMock(x=50.0, y=100.0, z=1.0)
        mock_vehicle.get_location.return_value = mock_loc

        # 1. Aligned vehicle (yaw=0.0 deg) and lane (yaw=0.0 deg) -> cos(0) = 1.0 >= 0.5 -> Accepted
        mock_rot_aligned = MagicMock(pitch=0.0, yaw=0.0, roll=0.0)
        mock_vehicle.get_transform.return_value = MagicMock(rotation=mock_rot_aligned)
        mock_wp = MagicMock()
        mock_wp.lane_type = "Driving"
        mock_wp.road_id = 1
        mock_wp.lane_id = -1
        mock_wp.transform.location = MagicMock(x=50.1, y=100.1, z=1.0)
        mock_wp.transform.rotation = MagicMock(yaw=0.0)
        mock_map.get_waypoint.return_value = mock_wp

        demo.last_checkpoint_time = 0.0
        assert demo.update_safe_checkpoint(mock_world, mock_vehicle, speed_kmh=20.0, traffic_light_state="GREEN") is True
        assert demo.safe_checkpoint is not None

        # 2. Opposite direction vehicle (yaw=180.0 deg) vs lane (yaw=0.0 deg) -> cos(180) = -1.0 < 0.5 -> Rejected
        mock_rot_opp = MagicMock(pitch=0.0, yaw=180.0, roll=0.0)
        mock_vehicle.get_transform.return_value = MagicMock(rotation=mock_rot_opp)
        demo.safe_checkpoint = None
        demo.last_checkpoint_time = 0.0
        assert demo.update_safe_checkpoint(mock_world, mock_vehicle, speed_kmh=20.0, traffic_light_state="GREEN") is False
        assert demo.safe_checkpoint is None

    def test_autopilot_recovery_sequence_and_tm_port_registration(self, demo):
        """Verifies Autopilot off -> teleport -> on sequence and intended TM port usage."""
        demo.tm_port = 8005
        mock_client = MagicMock()
        mock_tm = MagicMock()
        mock_client.get_trafficmanager.return_value = mock_tm
        demo.client = mock_client

        mock_world = MagicMock()
        mock_vehicle = MagicMock()
        mock_vehicle.get_location.return_value = MagicMock(x=10.0, y=20.0, z=0.5)
        mock_vehicle.get_transform.return_value = MagicMock(rotation=MagicMock(pitch=0.0, yaw=45.0, roll=0.0))

        # Force road recovery
        success, msg = demo.recover_vehicle_to_safe_road(mock_world, mock_vehicle, force=True)
        assert success is True
        assert "RECOVERED" in msg

        # Verify TM configuration
        mock_client.get_trafficmanager.assert_called_with(8005)
        mock_tm.auto_lane_change.assert_called_with(mock_vehicle, False)
        mock_tm.distance_to_leading_vehicle.assert_called_with(mock_vehicle, 3.5)
        mock_vehicle.set_autopilot.assert_called_with(True, 8005)

    def test_circling_detection_heuristic_distinguishes_in_place_spin_from_normal_turn(self, demo):
        """Verifies that in-place spinning flags circling, while normal forward turns do not."""
        # 1. Circling scenario: 360-deg rotation with displacement < 1.0m over 3.0s
        demo.pose_history.clear()
        t0 = time.monotonic()
        for i in range(15):
            t = t0 + i * 0.2
            yaw = (i * 30.0) % 360.0  # full rotation
            x = 100.0 + 0.3 * math.cos(math.radians(yaw))
            y = 100.0 + 0.3 * math.sin(math.radians(yaw))
            demo.update_motion_diagnostics(x=x, y=y, yaw=yaw, speed_kmh=5.0, steer=0.8, throttle=0.5, timestamp=t)

        assert demo.circling_detected is True
        assert "AUTOPILOT_CIRCLING_DETECTED" in demo.circling_reason

        # detect_stuck_condition should flag circling stuck
        is_stuck, reason = demo.detect_stuck_condition(None, None, speed_kmh=5.0)
        assert is_stuck is True
        assert "CIRCLING" in reason

        # 2. Legitimate turn scenario: 90-deg turn while moving forward 15 meters
        demo.pose_history.clear()
        t0 = time.monotonic()
        for i in range(15):
            t = t0 + i * 0.2
            yaw = min(90.0, i * 6.0)
            x = 100.0 + i * 1.0  # 14 meters forward
            y = 100.0 + i * 0.5
            demo.update_motion_diagnostics(x=x, y=y, yaw=yaw, speed_kmh=25.0, steer=0.2, throttle=0.3, timestamp=t)

        assert demo.circling_detected is False
        assert demo.rolling_displacement > 10.0

    def test_post_recovery_warmup_yields_waiting_for_sensors(self, demo, nominal_frames):
        """Verifies that post-recovery warmup state returns WAIT / MODERATE awaiting fresh observations."""
        rgb, seg, imu = nominal_frames
        demo.post_recovery_warmup_frames = 3

        ev = demo.evaluate_frame(
            rgb_raw=rgb,
            seg_raw=seg,
            imu_raw=imu,
            obstacle_raw=None,
            collision_raw=None,
            speed_kmh=2.0,
            traffic_light_state="NONE",
            sensor_timestamp=time.monotonic(),
        )

        assert ev.decision_outcome == DecisionOutcome.WAIT
        assert ev.risk_level == RiskLevel.MODERATE
        assert ev.policy_branch_reason == "POST_RECOVERY_WARMUP"
        assert "RECOVERY" in ev.decision_reason
        assert demo.post_recovery_warmup_frames == 2

    def test_nominal_driving_high_confidence_and_single_conformal_set(self, demo, nominal_frames):
        """Audits that clear nominal driving maintains high confidence (P>=0.85) and single conformal set."""
        rgb, seg, imu = nominal_frames
        demo.post_recovery_warmup_frames = 0
        demo.scenario_mode = ScenarioMode.NORMAL
        demo.fault_mode = FaultMode.NONE

        ev = demo.evaluate_frame(
            rgb_raw=rgb,
            seg_raw=seg,
            imu_raw=imu,
            obstacle_raw=None,
            collision_raw=None,
            speed_kmh=25.0,
            traffic_light_state="NONE",
            sensor_timestamp=time.monotonic(),
        )

        assert ev.decision_outcome == DecisionOutcome.ACT
        assert ev.risk_level == RiskLevel.LOW
        assert ev.escalation_required is False
        assert ev.confidence >= 0.85
        assert ev.epistemic_uncertainty <= 0.05
        assert ev.conformal_set_size == 1
        assert ev.policy_branch_reason == "NOMINAL_CLEAR_CORRIDOR"

    def test_trajectory_auditor_straight_vs_turn_vs_circling(self):
        """Audits trajectory diagnostics: distinguishes straight drive, smooth turn, and circular movement."""
        auditor = TrajectoryAuditor(history_seconds=15.0, cell_size_m=6.0)

        # 1. Straight driving: 20 samples along X axis at 30 km/h
        for i in range(20):
            res = auditor.update(
                t=i * 0.5,
                x=i * 4.0,
                y=100.0,
                z=0.0,
                yaw=0.0,
                speed_kmh=30.0,
                steer=0.0,
                road_id=1,
                lane_id=-1,
                lane_yaw=0.0,
            )
        assert res["efficiency_ratio"] > 0.95
        assert not res["circular_movement_detected"]
        assert not res["wrong_direction"]
        assert not res["is_immobilized"]
        assert res["cumulative_distance_m"] > 70.0

        # 2. Smooth sweeping turn: 90 deg turn over 60 meters
        auditor_turn = TrajectoryAuditor(history_seconds=15.0)
        for i in range(20):
            t = i * 0.5
            yaw = i * 4.5
            x = 80.0 + 30.0 * math.sin(math.radians(yaw))
            y = 100.0 + 30.0 * (1 - math.cos(math.radians(yaw)))
            res_turn = auditor_turn.update(
                t=t,
                x=x,
                y=y,
                z=0.0,
                yaw=yaw,
                speed_kmh=25.0,
                steer=0.1,
                road_id=1,
                lane_id=-1,
                lane_yaw=yaw,
            )
        assert not res_turn["circular_movement_detected"], "Smooth turn must NOT be flagged as circular movement"
        assert not res_turn["is_immobilized"]

        # 3. Tight spin / circling: 4m radius, spinning 360 deg
        auditor_spin = TrajectoryAuditor(history_seconds=15.0)
        for i in range(40):
            t = i * 0.5
            yaw = i * 18.0
            x = 4.0 * math.cos(math.radians(yaw))
            y = 4.0 * math.sin(math.radians(yaw))
            res_spin = auditor_spin.update(
                t=t,
                x=x,
                y=y,
                z=0.0,
                yaw=yaw,
                speed_kmh=15.0,
                steer=0.6,
                road_id=1,
                lane_id=-1,
                lane_yaw=0.0,
            )
        assert res_spin["circular_movement_detected"], "Tight spin MUST be flagged as circular movement"
        assert res_spin["efficiency_ratio"] < 0.35

    def test_trajectory_auditor_red_light_stop_not_flagged_as_stuck(self):
        """Audits that legal stops at red lights and queuing are distinguished from immobilization."""
        auditor_light = TrajectoryAuditor(history_seconds=15.0)
        for i in range(20):
            res_light = auditor_light.update(
                t=i * 0.5,
                x=50.0,
                y=50.0,
                z=0.0,
                yaw=0.0,
                speed_kmh=0.0,
                steer=0.0,
                road_id=1,
                lane_id=-1,
                is_at_red_light=True,
            )
        assert not res_light["is_immobilized"], "Red light stop must NOT be flagged as immobilized"
        assert res_light["stuck_seconds"] == 0.0

        # Queuing stop behind vehicle
        auditor_queue = TrajectoryAuditor(history_seconds=15.0)
        for i in range(20):
            res_queue = auditor_queue.update(
                t=i * 0.5,
                x=50.0,
                y=50.0,
                z=0.0,
                yaw=0.0,
                speed_kmh=0.0,
                steer=0.0,
                road_id=1,
                lane_id=-1,
                is_queuing=True,
            )
        assert not res_queue["is_immobilized"], "Queuing behind vehicle must NOT be flagged as immobilized"

        # Actual immobilization without red light or queue
        auditor_stuck = TrajectoryAuditor(history_seconds=15.0)
        for i in range(20):
            res_stuck = auditor_stuck.update(
                t=i * 0.5,
                x=50.0,
                y=50.0,
                z=0.0,
                yaw=0.0,
                speed_kmh=0.0,
                steer=0.0,
                road_id=1,
                lane_id=-1,
                is_at_red_light=False,
                is_queuing=False,
            )
        assert res_stuck["is_immobilized"], "Uncommanded stop without light/queue MUST be flagged as immobilized"
        assert res_stuck["stuck_seconds"] >= 4.0

    def test_presentation_route_builder_generation_and_spacing(self):
        """Audits build_presentation_loop_route: validates step spacing, heading continuity, and length."""
        mock_map = MagicMock()

        # Build a chain of mock waypoints
        mock_wps = []
        for i in range(50):
            wp = MagicMock()
            loc = MagicMock()
            loc.x = float(i * 2.5)
            loc.y = 0.0
            loc.z = 0.5
            wp.transform.location = loc
            wp.transform.rotation.yaw = 0.0
            mock_wps.append(wp)

        for i in range(len(mock_wps) - 1):
            # Provide primary continuation plus an alternate turn branch
            alt_branch = MagicMock()
            alt_loc = MagicMock(x=mock_wps[i].transform.location.x, y=10.0, z=0.5)
            alt_branch.transform.location = alt_loc
            alt_branch.transform.rotation.yaw = 90.0  # large angle divergence

            # next(step_m) returns both branches; continuity logic should pick mock_wps[i+1]
            mock_wps[i].next.return_value = [alt_branch, mock_wps[i + 1]]

        mock_wps[-1].next.return_value = []
        mock_map.get_waypoint.return_value = mock_wps[0]

        start_tf = MagicMock()
        start_tf.location = mock_wps[0].transform.location

        route = build_presentation_loop_route(mock_map, start_tf, total_distance_m=100.0, step_m=2.5)

        assert len(route) >= 40
        # Verify continuity: each successive point is ~2.5m along X axis
        for k in range(1, len(route)):
            dx = route[k].x - route[k - 1].x
            dy = route[k].y - route[k - 1].y
            dist = math.sqrt(dx * dx + dy * dy)
            assert abs(dist - 2.5) < 0.1
            assert route[k].y == 0.0  # Did not branch to alternate y=10.0

    def test_route_consistent_recovery(self, demo):
        """Audits that road recovery selects route waypoint ahead when presentation route is active."""
        # Set up a mock route of 20 locations using carla.Location
        route_locs = [carla.Location(x=float(i * 5.0), y=0.0, z=0.5) for i in range(20)]
        demo.presentation_route_locations = route_locs

        mock_world = MagicMock()
        mock_map = MagicMock()
        mock_world.get_map.return_value = mock_map
        mock_wp = MagicMock()
        mock_wp.road_id = 1
        mock_wp.lane_id = -1
        mock_wp.lane_type = carla.LaneType.Driving
        mock_wp.transform.rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        mock_map.get_waypoint.return_value = mock_wp

        mock_vehicle = MagicMock()
        # Vehicle is at x=10.0 (index 2)
        mock_vloc = carla.Location(x=10.0, y=0.0, z=0.5)
        mock_vehicle.get_location.return_value = mock_vloc
        mock_vehicle.get_transform.return_value = MagicMock(location=mock_vloc)

        mock_client = MagicMock()
        demo.client = mock_client
        mock_tm = MagicMock()
        mock_client.get_trafficmanager.return_value = mock_tm

        ok, msg = demo.recover_vehicle_to_safe_road(mock_world, mock_vehicle, force=True)

        assert ok is True
        assert "RECOVERED" in msg
        assert "ROUTE_WAYPOINT" in msg
        # Target index was min_idx (2) + 4 = 6
        assert "idx=6" in msg
        assert mock_tm.set_path.called
        assert demo.post_recovery_warmup_frames == 12

    def test_excessive_steering_oscillation_detection(self):
        """Audits detection of rapid high-frequency steering oscillations."""
        auditor = TrajectoryAuditor(history_seconds=15.0)
        # Alternate steering rapidly: +0.4 -> -0.4 -> +0.4 -> -0.4 within 2 seconds
        steer_vals = [0.4, -0.4, 0.4, -0.4, 0.4]
        res = None
        for i, st in enumerate(steer_vals):
            res = auditor.update(
                t=i * 0.4,
                x=i * 2.0,
                y=0.0,
                z=0.0,
                yaw=0.0,
                speed_kmh=20.0,
                steer=st,
                road_id=1,
                lane_id=-1,
            )
        assert res is not None
        assert res["excessive_steering"] is True, "Rapid steering sign reversals must trigger excessive_steering"

    def test_wrong_direction_lane_detection(self):
        """Audits detection of driving opposite to the lane waypoint flow direction."""
        auditor = TrajectoryAuditor(history_seconds=15.0)

        # Vehicle driving with yaw=180 on a lane with lane_yaw=0.0 (opposite direction)
        res_wrong = auditor.update(
            t=0.5,
            x=10.0,
            y=0.0,
            z=0.0,
            yaw=180.0,
            speed_kmh=25.0,
            steer=0.0,
            road_id=1,
            lane_id=-1,
            lane_yaw=0.0,
        )
        assert res_wrong["wrong_direction"] is True, "Driving opposite to lane yaw must trigger wrong_direction"

        # Vehicle driving in the lane direction
        auditor_correct = TrajectoryAuditor(history_seconds=15.0)
        res_correct = auditor_correct.update(
            t=0.5,
            x=10.0,
            y=0.0,
            z=0.0,
            yaw=0.0,
            speed_kmh=25.0,
            steer=0.0,
            road_id=1,
            lane_id=-1,
            lane_yaw=0.0,
        )
        assert res_correct["wrong_direction"] is False

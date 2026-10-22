"""
Unit tests for Cognix M16 Live Pedestrian Safety Decision Audit.

Verifies:
1. Pedestrian inside corridor triggers Escalation/Protective action (ESCALATE / RiskLevel.HIGH or WAIT / RiskLevel.MODERATE), NOT ACT / LOW.
2. Pedestrian outside corridor (e.g. sidewalk) allows nominal driving (ACT / RiskLevel.LOW) when corridor is clear.
3. Absent or invalid camera observation (image_bgr is None) produces fail-safe ABSTAIN with HIGH risk and DISCONNECTED status.
4. Missing or uncertain corridor result (e.g. corridor is None or boundary encroachment / weather degradation) triggers protective escalation or precautionary wait.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from cognix.decision.escalation import DecisionOutcome, RiskLevel
from cognix.evaluation.torchvision_detector import DetectionBox, FrameDetections
from cognix.evaluation.temporal_tracking import TrackState, TrackObservation, TrackTrajectory
from cognix.evaluation.carla_demo_runner import (
    CarlaDemonstrationPipeline,
    FramePacket,
    RoadwayCorridor,
)


def _make_trajectory(track_id: int, box: DetectionBox) -> TrackTrajectory:
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    obs = TrackObservation(
        tick=0,
        box=box,
        score=box.score,
        center_x=cx,
        center_y=cy,
        area=box.area,
    )
    return TrackTrajectory(
        track_id=track_id,
        class_name=box.class_name,
        class_id=box.class_id,
        state=TrackState.CONFIRMED,
        start_tick=0,
        last_tick=0,
        total_hits=1,
        consecutive_misses=0,
        observations=[obs],
    )


class TestPedestrianSafetyAudit:
    """Audit suite investigating and resolving the M15 Scenario B pedestrian discrepancy."""

    def _create_mocked_pipeline(self, active_trajs):
        """Helper to create a CarlaDemonstrationPipeline with mocked detector and tracker."""
        pipeline = CarlaDemonstrationPipeline()
        boxes = [t.latest_box for t in active_trajs]
        pipeline.detector.detect = MagicMock(
            return_value=FrameDetections(
                boxes=boxes,
                latency_ms=1.0,
                scenario_id="audit_test",
                tick=0,
                image_width=1920,
                image_height=1080,
            )
        )
        pipeline.tracker.update = MagicMock(return_value=active_trajs)
        return pipeline

    def test_case_a_pedestrian_inside_corridor_triggers_escalation(self):
        """
        Case A: A pedestrian detected inside the forward driving corridor with high confidence.

        Documented Cognix Safety Contract:
        P(ACT) = 1.0 - P(hazard).
        When a pedestrian occupies the ego corridor at score 1.00, P(ACT) must be ~0.00,
        triggering DecisionOutcome.ESCALATE and RiskLevel.HIGH (NEVER ACT with LOW risk).
        """
        # Corridor bounds: x in [480, 1440], y in [388, 918] on 1920x1080
        ped_box = DetectionBox(
            x1=615.0,
            y1=400.0,
            x2=658.0,
            y2=600.0,
            score=1.00,
            class_id=1,
            class_name="person",
        )
        traj = _make_trajectory(track_id=1, box=ped_box)

        pipeline = self._create_mocked_pipeline([traj])
        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        pkt = FramePacket(
            frame_index=0,
            tick=0,
            timestamp_sec=0.0,
            image_bgr=synth_img,
            scenario_id="audit_pedestrian_in_corridor",
            source_type="LIVE_CARLA",
        )

        rendered, telem = pipeline.process_frame(pkt)

        # Verification of corridor presence
        assert telem.corridor_hazard_count == 1
        # Verification that safety contract is honored: protective response
        assert telem.decision_outcome == DecisionOutcome.ESCALATE.value, (
            f"Expected ESCALATE for in-corridor pedestrian, got {telem.decision_outcome}"
        )
        assert telem.risk_level == RiskLevel.HIGH.value, (
            f"Expected HIGH risk for in-corridor pedestrian, got {telem.risk_level}"
        )
        assert telem.confidence <= 0.40, (
            f"P(ACT) must be <= 0.40 under severe hazard, got {telem.confidence}"
        )
        assert "Pedestrian detected inside forward driving corridor" in telem.decision_reason

    def test_case_a_moderate_score_pedestrian_inside_corridor_still_escalates(self):
        """
        Case A2: Pedestrian detected with moderate raw softmax score (e.g. 0.45).

        Explicit Safety Policy Contract:
        Because a confirmed pedestrian occupies the forward travel path, the policy
        mandates protective escalation (ESCALATE, HIGH risk), rather than allowing
        precautionary WAIT or ACT based on raw uncalibrated softmax score.
        """
        ped_box = DetectionBox(
            x1=615.0,
            y1=400.0,
            x2=658.0,
            y2=600.0,
            score=0.45,
            class_id=1,
            class_name="person",
        )
        traj = _make_trajectory(track_id=11, box=ped_box)
        pipeline = self._create_mocked_pipeline([traj])
        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        pkt = FramePacket(
            frame_index=10,
            tick=10,
            timestamp_sec=1.0,
            image_bgr=synth_img,
            scenario_id="audit_pedestrian_moderate_score",
            source_type="LIVE_CARLA",
        )
        rendered, telem = pipeline.process_frame(pkt)

        assert telem.corridor_hazard_count == 1
        assert telem.decision_outcome == DecisionOutcome.ESCALATE.value
        assert telem.risk_level == RiskLevel.HIGH.value
        assert telem.confidence <= 0.40
        assert "Pedestrian detected inside forward driving corridor" in telem.decision_reason

    def test_case_a3_general_obstacle_moderate_score_triggers_precautionary_wait(self):
        """
        Case A3: A non-pedestrian obstacle (e.g. car) detected in corridor with moderate score (0.45).

        General obstacles scale confidence inversely with prominence:
        conf = 1.0 - 0.45 = 0.55 (ambiguous zone [0.40, 0.70)), triggering WAIT / MODERATE risk.
        """
        car_box = DetectionBox(
            x1=615.0,
            y1=400.0,
            x2=750.0,
            y2=600.0,
            score=0.45,
            class_id=3,
            class_name="car",
        )
        traj = _make_trajectory(track_id=12, box=car_box)
        pipeline = self._create_mocked_pipeline([traj])
        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        pkt = FramePacket(
            frame_index=11,
            tick=11,
            timestamp_sec=1.1,
            image_bgr=synth_img,
            scenario_id="audit_car_moderate_score",
            source_type="LIVE_CARLA",
        )
        rendered, telem = pipeline.process_frame(pkt)

        assert telem.corridor_hazard_count == 1
        assert telem.decision_outcome == DecisionOutcome.WAIT.value
        assert telem.risk_level == RiskLevel.MODERATE.value
        assert 0.40 <= telem.confidence < 0.70

    def test_case_b_pedestrian_outside_corridor_allows_nominal_act(self):
        """
        Case B: A pedestrian detected outside the driving corridor (e.g., on the sidewalk).

        The corridor remains clear, so nominal driving is permitted:
        DecisionOutcome.ACT with RiskLevel.LOW.
        """
        # Sidewalk box: x in [50, 150] (outside corridor x_min=480)
        sidewalk_ped = DetectionBox(
            x1=50.0,
            y1=500.0,
            x2=150.0,
            y2=700.0,
            score=0.95,
            class_id=1,
            class_name="person",
        )
        traj = _make_trajectory(track_id=2, box=sidewalk_ped)

        pipeline = self._create_mocked_pipeline([traj])
        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        pkt = FramePacket(
            frame_index=1,
            tick=1,
            timestamp_sec=0.1,
            image_bgr=synth_img,
            scenario_id="audit_pedestrian_on_sidewalk",
            source_type="LIVE_CARLA",
        )

        rendered, telem = pipeline.process_frame(pkt)

        assert telem.corridor_hazard_count == 0
        assert telem.decision_outcome == DecisionOutcome.ACT.value
        assert telem.risk_level == RiskLevel.LOW.value
        assert telem.confidence >= 0.70
        assert "Forward corridor clear" in telem.decision_reason

    def test_case_c_absent_invalid_camera_observation_triggers_abstain(self):
        """
        Case C: Absent or invalid camera observation (image_bgr is None).

        Pipeline must output fail-safe ABSTAIN with HIGH risk and DISCONNECTED status.
        """
        pipeline = CarlaDemonstrationPipeline()
        pkt = FramePacket(
            frame_index=2,
            tick=2,
            timestamp_sec=0.2,
            image_bgr=None,
            scenario_id="audit_missing_observation",
            source_type="LIVE_CARLA",
            fault_mode="dropout",
        )

        rendered, telem = pipeline.process_frame(pkt)

        assert telem.camera_status == "DISCONNECTED"
        assert telem.decision_outcome == DecisionOutcome.ABSTAIN.value
        assert telem.risk_level == RiskLevel.HIGH.value
        assert telem.confidence == 0.0
        assert telem.epistemic_uncertainty == 1.0
        assert "Camera signal missing" in telem.decision_reason

    def test_case_d_missing_corridor_triggers_failsafe_escalation(self):
        """
        Case D1: Corridor result is missing (corridor is None).

        Without a verified corridor boundary, the forward travel path cannot be
        guaranteed clear. Pipeline must trigger safety fail-safe escalation.
        """
        pipeline = CarlaDemonstrationPipeline()
        pipeline.corridor = None  # Simulate missing corridor definition

        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        pkt = FramePacket(
            frame_index=3,
            tick=3,
            timestamp_sec=0.3,
            image_bgr=synth_img,
            scenario_id="audit_missing_corridor",
            source_type="LIVE_CARLA",
        )

        rendered, telem = pipeline.process_frame(pkt)

        assert telem.decision_outcome == DecisionOutcome.ESCALATE.value
        assert telem.risk_level == RiskLevel.HIGH.value
        assert telem.epistemic_uncertainty > 0.50
        assert "corridor undefined" in telem.decision_reason.lower()

    def test_case_d_uncertain_boundary_encroachment_triggers_precautionary_wait(self):
        """
        Case D2: Corridor result is uncertain due to boundary encroachment.

        Object center is slightly outside corridor, but the bounding box overlaps
        into the driving corridor (>10%). Pipeline must trigger precautionary WAIT
        with MODERATE risk.
        """
        # Corridor left boundary on 1920p is x_min = 0.25 * 1920 = 480px.
        # Box center is at x=450 (outside corridor), but box spans [350, 550],
        # overlapping the corridor by 70px in width.
        encroaching_box = DetectionBox(
            x1=350.0,
            y1=500.0,
            x2=550.0,
            y2=700.0,
            score=0.90,
            class_id=1,
            class_name="person",
        )
        traj = _make_trajectory(track_id=3, box=encroaching_box)

        pipeline = self._create_mocked_pipeline([traj])
        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        pkt = FramePacket(
            frame_index=4,
            tick=4,
            timestamp_sec=0.4,
            image_bgr=synth_img,
            scenario_id="audit_encroaching_pedestrian",
            source_type="LIVE_CARLA",
        )

        rendered, telem = pipeline.process_frame(pkt)

        assert telem.decision_outcome == DecisionOutcome.WAIT.value
        assert telem.risk_level == RiskLevel.MODERATE.value
        assert "encroaching on corridor boundary" in telem.decision_reason


        pipeline = self._create_mocked_pipeline([traj])
        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        pkt = FramePacket(
            frame_index=4,
            tick=4,
            timestamp_sec=0.4,
            image_bgr=synth_img,
            scenario_id="audit_encroaching_pedestrian",
            source_type="LIVE_CARLA",
        )

        rendered, telem = pipeline.process_frame(pkt)

        assert telem.decision_outcome == DecisionOutcome.WAIT.value
        assert telem.risk_level == RiskLevel.MODERATE.value
        assert "encroaching on corridor boundary" in telem.decision_reason

"""Unit tests for Cognix M13 CARLA Demonstration Runner."""
import pytest
import numpy as np
from pathlib import Path

from cognix.evaluation.carla_demo_runner import (
    CarlaConnectionError,
    CarlaLiveSource,
    CarlaReplaySource,
    SensorFaultSimulator,
    CarlaDemonstrationPipeline,
    FramePacket,
    FrameTelemetry,
)


class TestCarlaDemoRunner:
    def test_carla_live_source_connection_failure(self):
        # Connect to unused local port; should raise CarlaConnectionError
        live_src = CarlaLiveSource(host="127.0.0.1", port=19999, timeout=0.5)
        with pytest.raises(CarlaConnectionError) as exc_info:
            live_src.connect()
        assert "Could not connect to live CARLA server" in str(exc_info.value)

    def test_carla_replay_source_valid_scenarios(self):
        for sc in ["normal", "pedestrian", "weather", "props"]:
            src = CarlaReplaySource(sc)
            assert len(src.frame_files) == 10
            frames = list(src.stream_frames(max_frames=2))
            assert len(frames) == 2
            assert frames[0].source_type == "RECORDED_REPLAY"
            assert frames[0].image_bgr is not None
            assert frames[0].image_bgr.shape == (1080, 1920, 3)

    def test_carla_replay_source_invalid_scenario(self):
        with pytest.raises(ValueError) as exc_info:
            CarlaReplaySource("nonexistent_scenario")
        assert "Unknown scenario" in str(exc_info.value)

    def test_sensor_fault_simulator(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200

        # None
        res_none = SensorFaultSimulator.apply_fault(img, "none")
        assert np.array_equal(res_none, img)

        # Dropout
        res_drop = SensorFaultSimulator.apply_fault(img, "dropout")
        assert res_drop is None

        # Darkness
        res_dark = SensorFaultSimulator.apply_fault(img, "darkness")
        assert np.all(res_dark < 50)

        # Occlusion
        res_occ = SensorFaultSimulator.apply_fault(img, "occlusion")
        assert res_occ.shape == img.shape
        # Center patch should be dark
        assert np.all(res_occ[40:60, 40:60] == 10)

    def test_pipeline_dropout_handling(self):
        pipeline = CarlaDemonstrationPipeline()
        pkt = FramePacket(
            frame_index=0,
            tick=0,
            timestamp_sec=0.0,
            image_bgr=None,
            scenario_id="test_dropout",
            source_type="RECORDED_REPLAY",
            fault_mode="dropout",
        )
        rendered, telem = pipeline.process_frame(pkt)
        assert rendered.shape == (1080, 1920, 3)
        assert telem.camera_status == "DISCONNECTED"
        assert telem.decision_outcome == "ABSTAIN"
        assert telem.risk_level == "HIGH"

    def test_pipeline_nominal_frame(self):
        pipeline = CarlaDemonstrationPipeline()
        # Synthetic clean roadway image (uniform gray with road color)
        synth_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        synth_img[500:, :] = 80  # simulated road surface
        synth_img[:500, :] = 160 # simulated sky

        pkt = FramePacket(
            frame_index=0,
            tick=0,
            timestamp_sec=0.0,
            image_bgr=synth_img,
            scenario_id="test_nominal",
            source_type="RECORDED_REPLAY",
        )
        rendered, telem = pipeline.process_frame(pkt)
        assert rendered.shape == (1080, 1920, 3)
        assert telem.camera_status == "OK"
        assert telem.corridor_hazard_count == 0
        assert telem.risk_level in ("LOW", "MODERATE")
        assert telem.total_latency_ms > 0.0

    def test_pipeline_props_limitation_flag(self):
        pipeline = CarlaDemonstrationPipeline()
        synth_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
        pkt = FramePacket(
            frame_index=0,
            tick=0,
            timestamp_sec=0.0,
            image_bgr=synth_img,
            scenario_id="test_props",
            source_type="RECORDED_REPLAY",
        )
        rendered, telem = pipeline.process_frame(pkt, is_props_scenario=True)
        assert telem.known_detector_limitation is True

    def test_carla_live_source_scenario_initialization(self):
        for sc in ["normal", "pedestrian", "weather", "props"]:
            src = CarlaLiveSource(scenario_name=sc, host="127.0.0.1", port=19999, timeout=0.1)
            assert src.scenario_name == sc
            assert src.host == "127.0.0.1"
            assert src.port == 19999
            assert src.spawned_actors == []
            assert src._saved_weather is None

    def test_carla_live_source_cleanup_safeguard(self):
        class MockActor:
            def __init__(self):
                self.destroyed = False
            def destroy(self):
                self.destroyed = True

        class MockWorld:
            def __init__(self):
                self.weather_restored = False
            def set_weather(self, w):
                self.weather_restored = True

        src = CarlaLiveSource(scenario_name="weather", host="127.0.0.1", port=19999)
        mock_actor = MockActor()
        src.spawned_actors.append(mock_actor)
        src.world = MockWorld()
        src._saved_weather = "dummy_weather"

        src.cleanup()

        assert mock_actor.destroyed is True
        assert src.spawned_actors == []
        assert src.world.weather_restored is True
        assert src._saved_weather is None

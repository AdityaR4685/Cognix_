"""
Unit tests for Perception-Aware Hazard Features and Experimental Gating (Milestone 4).
"""
import numpy as np
import pytest

from cognix.evaluation.perception_features import (
    extract_corridor_hazard_occupancy,
    extract_sensor_health_signal,
    evaluate_perception_variants,
    DEFAULT_HAZARD_CLASSES,
)


class TestPerceptionFeatures:
    def test_corridor_hazard_occupancy_2d_basic(self):
        # 100 x 100 image
        img = np.zeros((100, 100), dtype=np.uint8)
        # Set a 20x20 block of Class 4 (Pedestrian) in lower central region: y in [60, 80], x in [40, 60]
        img[60:80, 40:60] = 4

        # Corridor: y in [50, 100], x in [33, 67] -> height=50, width=34 -> size = 1700
        # Hazard pixels in corridor: 20 * 20 = 400
        occ = extract_corridor_hazard_occupancy(
            img, y_range=(0.5, 1.0), x_range=(0.33, 0.67), hazard_classes=(4,)
        )
        expected = 400.0 / (50 * 34)
        assert abs(occ - expected) < 1e-5

    def test_corridor_hazard_occupancy_3d_channel0(self):
        # 100 x 100 x 4 RGBA image
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[70:90, 40:60, 0] = 3  # Class 3 in channel 0
        img[70:90, 40:60, 1:] = 255  # Other channels should be ignored

        occ = extract_corridor_hazard_occupancy(
            img, y_range=(0.5, 1.0), x_range=(0.33, 0.67), hazard_classes=(3,)
        )
        assert occ > 0.0

    def test_corridor_hazard_occupancy_empty_bounds(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        assert extract_corridor_hazard_occupancy(img, y_range=(0.5, 0.5)) == 0.0
        assert extract_corridor_hazard_occupancy(img, x_range=(0.5, 0.5)) == 0.0

    def test_sensor_health_monitor_healthy(self):
        preds = {"Camera": 0.85, "IMU": 0.90, "Seg": 0.88}
        degraded, failing, min_p = extract_sensor_health_signal(preds, min_loss_threshold=0.10)
        assert degraded is False
        assert failing is None
        assert abs(min_p - 0.85) < 1e-5

    def test_sensor_health_monitor_dropout(self):
        preds = {"Camera": 0.02, "IMU": 0.90, "Seg": 0.88}
        degraded, failing, min_p = extract_sensor_health_signal(preds, min_loss_threshold=0.10)
        assert degraded is True
        assert failing == "Camera"
        assert abs(min_p - 0.02) < 1e-5

    def test_evaluate_perception_variants(self):
        # Case 1: Baseline ACT, no hazard, healthy sensors -> All ACT
        res = evaluate_perception_variants(
            baseline_decision="ACT",
            hazard_occupancy=0.01,
            agent_predictions={"Camera": 0.80, "IMU": 0.85, "Seg": 0.82},
            hazard_occupancy_threshold=0.05,
            sensor_loss_threshold=0.10,
        )
        assert res.baseline_decision == "ACT"
        assert res.perception_hazard_decision == "ACT"
        assert res.sensor_health_decision == "ACT"
        assert res.combined_decision == "ACT"

        # Case 2: Hazard detected (> 0.05), healthy sensors -> V2 and V4 ESCALATE
        res = evaluate_perception_variants(
            baseline_decision="ACT",
            hazard_occupancy=0.12,
            agent_predictions={"Camera": 0.80, "IMU": 0.85, "Seg": 0.82},
            hazard_occupancy_threshold=0.05,
            sensor_loss_threshold=0.10,
        )
        assert res.baseline_decision == "ACT"
        assert res.perception_hazard_decision == "ESCALATE"
        assert res.sensor_health_decision == "ACT"
        assert res.combined_decision == "ESCALATE"

        # Case 3: Sensor loss (< 0.10), no hazard -> V3 and V4 ESCALATE
        res = evaluate_perception_variants(
            baseline_decision="ACT",
            hazard_occupancy=0.01,
            agent_predictions={"Camera": 0.04, "IMU": 0.85, "Seg": 0.82},
            hazard_occupancy_threshold=0.05,
            sensor_loss_threshold=0.10,
        )
        assert res.baseline_decision == "ACT"
        assert res.perception_hazard_decision == "ACT"
        assert res.sensor_health_decision == "ESCALATE"
        assert res.combined_decision == "ESCALATE"
        assert res.failing_agent == "Camera"

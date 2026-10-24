"""
Unit tests for Pretrained Perception Benchmark and Feasibility Module (Milestone 5).
"""
import numpy as np
import pytest

from cognix.evaluation.pretrained_perception_benchmark import (
    audit_local_perception_environment,
    compute_corridor_hog_features,
    evaluate_rgb_gradient_shift,
)


class TestPretrainedPerceptionBenchmark:
    def test_audit_local_perception_environment(self):
        audit = audit_local_perception_environment()
        assert "installed_packages" in audit
        assert "candidates" in audit
        assert "has_offline_deep_detector" in audit
        assert audit["has_offline_deep_detector"] is False
        assert len(audit["candidates"]) >= 4

        # Verify candidate schemas
        for cand in audit["candidates"]:
            assert "name" in cand
            assert "architecture" in cand
            assert "available_locally" in cand
            assert "dependencies" in cand

    def test_compute_corridor_hog_features_basic(self):
        # 128 x 128 synthetic test patch
        patch = np.zeros((128, 128), dtype=np.uint8)
        patch[32:96, 48:80] = 255  # Vertical bright bar

        feat = compute_corridor_hog_features(patch, orientations=8, pixels_per_cell=(32, 32))
        assert feat.ndim == 1
        assert len(feat) > 0
        assert np.isfinite(feat).all()

    def test_compute_corridor_hog_features_invalid_shape(self):
        # 3D input should raise ValueError
        bad_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D grayscale"):
            compute_corridor_hog_features(bad_img)

    def test_compute_corridor_hog_features_empty(self):
        empty = np.zeros((0, 0), dtype=np.uint8)
        res = compute_corridor_hog_features(empty)
        assert len(res) == 0

    def test_evaluate_rgb_gradient_shift(self):
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0, 0.0])
        shift = evaluate_rgb_gradient_shift(v1, v2)
        assert abs(shift["euclidean_distance"] - 0.0) < 1e-6
        assert abs(shift["cosine_similarity"] - 1.0) < 1e-6

        v3 = np.array([0.0, 1.0, 0.0, 0.0])
        shift2 = evaluate_rgb_gradient_shift(v1, v3)
        assert abs(shift2["euclidean_distance"] - np.sqrt(2.0)) < 1e-6
        assert abs(shift2["cosine_similarity"] - 0.0) < 1e-6

    def test_evaluate_rgb_gradient_shift_shape_mismatch(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_rgb_gradient_shift(v1, v2)

"""Unit tests for isolated TorchVision object detector research adapter."""
import pytest
import math
import numpy as np
from PIL import Image

from cognix.evaluation.torchvision_detector import (
    DetectionBox,
    FrameDetections,
    TorchVisionObjectDetector,
)


class TestTorchVisionDetection:
    def test_detection_box_validation_valid(self):
        box = DetectionBox(
            x1=100.0,
            y1=150.0,
            x2=200.0,
            y2=300.0,
            score=0.85,
            class_id=1,
            class_name="person",
        )
        assert box.is_valid(image_width=640, image_height=480) is True
        assert box.area == 15000.0
        d = box.to_dict()
        assert d["class_name"] == "person"
        assert d["score"] == 0.85

    def test_detection_box_validation_invalid(self):
        # NaN coordinate
        b_nan = DetectionBox(x1=float("nan"), y1=10.0, x2=50.0, y2=50.0, score=0.9, class_id=1, class_name="person")
        assert b_nan.is_valid(640, 480) is False

        # Inverted coordinates
        b_inv = DetectionBox(x1=100.0, y1=10.0, x2=50.0, y2=50.0, score=0.9, class_id=1, class_name="person")
        assert b_inv.is_valid(640, 480) is False

        # Negative coordinate
        b_neg = DetectionBox(x1=-10.0, y1=10.0, x2=50.0, y2=50.0, score=0.9, class_id=1, class_name="person")
        assert b_neg.is_valid(640, 480) is False

        # Score out of bounds
        b_score = DetectionBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0, score=1.5, class_id=1, class_name="person")
        assert b_score.is_valid(640, 480) is False

    def test_input_validation_invalid_shape(self):
        detector = TorchVisionObjectDetector(confidence_threshold=0.5)
        # 2D grayscale array
        with pytest.raises(ValueError, match="Expected 3-channel RGB image array"):
            detector.preprocess_image(np.zeros((100, 100), dtype=np.uint8))

        # 4-channel RGBA array
        with pytest.raises(ValueError, match="Expected 3-channel RGB image array"):
            detector.preprocess_image(np.zeros((100, 100, 4), dtype=np.uint8))

    def test_input_validation_invalid_type(self):
        detector = TorchVisionObjectDetector(confidence_threshold=0.5)
        with pytest.raises(TypeError, match="Unsupported image type"):
            detector.preprocess_image("invalid_path.jpg")  # type: ignore

    def test_preprocessing_deterministic(self):
        detector = TorchVisionObjectDetector(confidence_threshold=0.5)
        raw_img = np.full((300, 400, 3), 128, dtype=np.uint8)
        tensor1 = detector.preprocess_image(raw_img)
        tensor2 = detector.preprocess_image(raw_img)

        assert tensor1.shape == (3, 300, 400)
        assert tensor1.dtype == detector.torch.float32
        assert detector.torch.equal(tensor1, tensor2)
        # Scaled to [0, 1] range: 128 / 255 ≈ 0.50196
        assert float(tensor1.min()) >= 0.0
        assert float(tensor1.max()) <= 1.0

    def test_category_metadata_mapping(self):
        detector = TorchVisionObjectDetector()
        categories = detector.get_supported_categories()
        assert len(categories) == 91
        assert categories[0] == "__background__"
        assert categories[1] == "person"
        assert categories[3] == "car"
        assert "bicycle" in categories
        assert "truck" in categories
        assert "bus" in categories

    def test_corridor_filtering(self):
        # Create a FrameDetections with 2 boxes: one in corridor, one on sidewalk
        in_corridor = DetectionBox(x1=300, y1=500, x2=500, y2=800, score=0.9, class_id=1, class_name="person")
        on_sidewalk = DetectionBox(x1=10, y1=500, x2=80, y2=800, score=0.9, class_id=1, class_name="person")
        frame = FrameDetections(
            scenario_id="test",
            tick=1,
            latency_ms=10.0,
            image_width=1000,
            image_height=1000,
            boxes=[in_corridor, on_sidewalk],
        )

        corridor_boxes = frame.filter_by_corridor(
            y_min_frac=0.45, y_max_frac=0.95, x_min_frac=0.25, x_max_frac=0.75
        )
        assert len(corridor_boxes) == 1
        assert corridor_boxes[0] == in_corridor

    def test_empty_detections_handling(self):
        detector = TorchVisionObjectDetector(confidence_threshold=0.999)  # Ultra-high threshold
        # Black image has no objects with 99.9% confidence
        black_img = np.zeros((200, 200, 3), dtype=np.uint8)
        frame = detector.detect(black_img, scenario_id="dummy", tick=1)

        assert isinstance(frame, FrameDetections)
        assert frame.tick == 1
        assert len(frame.boxes) == 0
        assert frame.latency_ms > 0.0

    def test_proxy_segmentation_overlap(self):
        # Synthetic segmentation mask: 100x100, class 4 (pedestrian) in region [40:60, 40:60]
        seg_mask = np.zeros((100, 100), dtype=np.uint8)
        seg_mask[40:60, 40:60] = 4  # Pedestrian tag

        # Box covering exact region [40, 40, 60, 60] -> 100% overlap
        box_perfect = DetectionBox(x1=40, y1=40, x2=60, y2=60, score=0.9, class_id=1, class_name="person")
        overlap_perf = TorchVisionObjectDetector.evaluate_proxy_overlap(box_perfect, seg_mask, [4])
        assert math.isclose(overlap_perf, 1.0, abs_tol=1e-3)

        # Box covering [0, 0, 20, 20] -> 0% overlap
        box_none = DetectionBox(x1=0, y1=0, x2=20, y2=20, score=0.9, class_id=1, class_name="person")
        overlap_none = TorchVisionObjectDetector.evaluate_proxy_overlap(box_none, seg_mask, [4])
        assert math.isclose(overlap_none, 0.0, abs_tol=1e-3)

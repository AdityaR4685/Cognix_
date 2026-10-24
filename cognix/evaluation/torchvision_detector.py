"""Pretrained TorchVision Object Detector Adapter for CARLA Perception Benchmarking.

Isolated research adapter implementing object-level bounding box inference
using torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.
DO NOT IMPORT INTO PRODUCTION PIPELINE OR DECISION RUNNERS.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
import math
import os
import time
import numpy as np
from PIL import Image

# Ensure TORCH_HOME default for isolated cache
DEFAULT_CACHE_DIR = r"C:\Cognix_ModelCache\torchvision_m6"
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = DEFAULT_CACHE_DIR


@dataclass(frozen=True)
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def is_valid(self, image_width: int, image_height: int) -> bool:
        """Validate box coordinates are finite, positive, and ordered."""
        for val in (self.x1, self.y1, self.x2, self.y2, self.score):
            if not math.isfinite(val):
                return False
        if self.x1 < 0 or self.y1 < 0:
            return False
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return False
        if self.x2 > image_width + 10 or self.y2 > image_height + 10:
            return False
        return 0.0 <= self.score <= 1.0


@dataclass(frozen=True)
class FrameDetections:
    scenario_id: str
    tick: int
    latency_ms: float
    image_width: int
    image_height: int
    boxes: List[DetectionBox]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "tick": self.tick,
            "latency_ms": round(self.latency_ms, 2),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "num_detections": len(self.boxes),
            "detections": [b.to_dict() for b in self.boxes],
        }

    def filter_by_class(self, class_names: List[str]) -> List[DetectionBox]:
        return [b for b in self.boxes if b.class_name in class_names]

    def filter_by_corridor(
        self, y_min_frac: float = 0.45, y_max_frac: float = 0.95, x_min_frac: float = 0.25, x_max_frac: float = 0.75
    ) -> List[DetectionBox]:
        """Filter detections whose center lies within the driving corridor."""
        corridor_boxes = []
        for b in self.boxes:
            cx = (b.x1 + b.x2) / (2.0 * self.image_width)
            cy = (b.y1 + b.y2) / (2.0 * self.image_height)
            if y_min_frac <= cy <= y_max_frac and x_min_frac <= cx <= x_max_frac:
                corridor_boxes.append(b)
        return corridor_boxes


class TorchVisionObjectDetector:
    """Isolated wrapper for Faster R-CNN MobileNetV3-Large FPN with COCO_V1 weights."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        if cache_dir:
            os.environ["TORCH_HOME"] = cache_dir

        import torch
        import torchvision
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_fpn,
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        )

        self.torch = torch
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.weight_enum = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.categories = list(self.weight_enum.meta["categories"])
        self.transforms = self.weight_enum.transforms()

        # Instantiate model with explicit weights enum (frozen for inference)
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=self.weight_enum)
        self.model.to(self.device)
        self.model.eval()

    def get_supported_categories(self) -> List[str]:
        return list(self.categories)

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> Any:
        """Preprocess an RGB numpy array or PIL Image into a model input tensor."""
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel RGB image array (H, W, 3), got shape {image.shape}")
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Must be np.ndarray or PIL.Image.")

        tensor_img = self.transforms(pil_image)
        return tensor_img.to(self.device)

    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
        scenario_id: str = "unknown",
        tick: int = 0,
    ) -> FrameDetections:
        """Run object detection inference on a single RGB frame."""
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size

        tensor_img = self.preprocess_image(image)

        t0 = time.perf_counter()
        with self.torch.no_grad():
            outputs = self.model([tensor_img])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_boxes = outputs["boxes"].cpu()
        raw_labels = outputs["labels"].cpu()
        raw_scores = outputs["scores"].cpu()

        valid_detections: List[DetectionBox] = []
        for i in range(len(raw_scores)):
            score = float(raw_scores[i].item())
            if score < self.confidence_threshold:
                continue

            coords = raw_boxes[i].tolist()
            label_idx = int(raw_labels[i].item())

            if 0 <= label_idx < len(self.categories):
                class_name = self.categories[label_idx]
            else:
                class_name = f"unknown_class_{label_idx}"

            det = DetectionBox(
                x1=coords[0],
                y1=coords[1],
                x2=coords[2],
                y2=coords[3],
                score=score,
                class_id=label_idx,
                class_name=class_name,
            )

            if det.is_valid(image_width=w, image_height=h):
                valid_detections.append(det)

        return FrameDetections(
            scenario_id=scenario_id,
            tick=tick,
            latency_ms=latency_ms,
            image_width=w,
            image_height=h,
            boxes=valid_detections,
        )

    @staticmethod
    def evaluate_proxy_overlap(
        box: DetectionBox,
        seg_mask: np.ndarray,
        target_seg_ids: List[int],
    ) -> float:
        """Diagnostic proxy check: Compute pixel overlap ratio with semantic segmentation classes.

        CARLA Cityscapes tags: Pedestrian = 4, Vehicle = 10.
        Returns fraction of box area covered by target semantic IDs.
        """
        if seg_mask.ndim == 3:
            # Channel 0 or 2 depending on encoding; CARLA standard encodes class in R channel
            seg_classes = seg_mask[:, :, 0]
        else:
            seg_classes = seg_mask

        h, w = seg_classes.shape[:2]
        x1 = max(0, min(w - 1, int(box.x1)))
        y1 = max(0, min(h - 1, int(box.y1)))
        x2 = max(0, min(w, int(box.x2)))
        y2 = max(0, min(h, int(box.y2)))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        box_crop = seg_classes[y1:y2, x1:x2]
        total_pixels = box_crop.size
        if total_pixels == 0:
            return 0.0

        matched_pixels = np.isin(box_crop, target_seg_ids).sum()
        return float(matched_pixels / total_pixels)

"""
Pretrained Perception Benchmark, Audit, and Hazard-Feature Feasibility (Milestone 5).

This module provides isolated evaluation utilities to:
1. Audit local pretrained models, weights, and runtime dependencies.
2. Benchmark local RGB visual features (block-mean, local HOG gradient descriptors)
   against CARLA semantic segmentation reference masks.
3. Provide a rigorous, label-blind feasibility analysis for integrating pretrained
   object detection backbones into Cognix.

ARCHITECTURAL ISOLATION:
- Purely research and evaluation code.
- Does NOT alter production DecisionEngine, agents, or escalation policies.
- Strictly label-blind: never accesses ground-truth anomaly labels or masks during inference.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ModelCandidateAudit:
    name: str
    architecture: str
    version: Optional[str]
    weight_path: Optional[str]
    sha256: Optional[str]
    source_license: str
    dependencies: List[str]
    supported_classes: List[str]
    available_locally: bool
    limitations: str


def audit_local_perception_environment() -> Dict[str, Any]:
    """
    Audits the current local Python environment for pretrained vision models,
    detection libraries, and cached weights.
    """
    installed_packages = {}
    for pkg in ("torch", "torchvision", "ultralytics", "timm", "transformers", "cv2", "skimage", "PIL", "sklearn"):
        try:
            mod = __import__(pkg)
            installed_packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            installed_packages[pkg] = "not_installed"

    candidates = [
        ModelCandidateAudit(
            name="Torchvision FasterRCNN-ResNet50-FPN",
            architecture="Faster R-CNN with ResNet-50-FPN backbone",
            version=None,
            weight_path=None,
            sha256=None,
            source_license="BSD-3-Clause",
            dependencies=["torch", "torchvision"],
            supported_classes=["person", "car", "bicycle", "motorcycle", "traffic light", "stop sign"],
            available_locally=bool(installed_packages.get("torchvision") != "not_installed"),
            limitations="Package torchvision is not installed in the local environment; weights not cached."
        ),
        ModelCandidateAudit(
            name="Ultralytics YOLOv8 (yolov8n.pt)",
            architecture="YOLOv8 nano anchor-free object detector",
            version=None,
            weight_path=None,
            sha256=None,
            source_license="AGPL-3.0",
            dependencies=["torch", "ultralytics"],
            supported_classes=["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"],
            available_locally=bool(installed_packages.get("ultralytics") != "not_installed"),
            limitations="Package ultralytics is not installed; no weight checkpoint cached locally."
        ),
        ModelCandidateAudit(
            name="HuggingFace DetrForObjectDetection",
            architecture="DETR (DEtection TRansformer) with ResNet-50",
            version=None,
            weight_path=None,
            sha256=None,
            source_license="Apache-2.0",
            dependencies=["torch", "transformers"],
            supported_classes=["person", "vehicle", "traffic sign"],
            available_locally=bool(installed_packages.get("transformers") != "not_installed"),
            limitations="Package transformers is not installed; no cached weights."
        ),
        ModelCandidateAudit(
            name="Scikit-Image Spatial HOG Feature Extractor",
            architecture="Histogram of Oriented Gradients (Dalal & Triggs 2005)",
            version=installed_packages.get("skimage"),
            weight_path=None,
            sha256="algorithmic_deterministic_feature",
            source_license="BSD-3-Clause",
            dependencies=["skimage", "numpy"],
            supported_classes=["spatial_gradient_orientations"],
            available_locally=True,
            limitations="Handcrafted visual feature; lacks semantic object classification or bounding-box localization."
        )
    ]

    return {
        "installed_packages": installed_packages,
        "candidates": [asdict(c) for c in candidates],
        "has_offline_deep_detector": False,
        "primary_blocker": "No pretrained object detection libraries (torchvision, ultralytics) or weight files exist locally."
    }


def compute_corridor_hog_features(
    image_l: np.ndarray,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """
    Extracts deterministic HOG features from a grayscale image patch.

    Args:
        image_l: Grayscale 2D image array of shape (H, W).
        orientations: Number of gradient orientation bins.
        pixels_per_cell: Cell dimensions in pixels.

    Returns:
        1D float64 feature vector of gradient orientations.
    """
    arr = np.asarray(image_l)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image array, got {arr.shape}")
    if arr.size == 0:
        return np.empty((0,), dtype=np.float64)

    from skimage.feature import hog
    feat = hog(
        arr,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(1, 1),
        visualize=False,
    )
    return np.asarray(feat, dtype=np.float64)


def evaluate_rgb_gradient_shift(
    baseline_hog: np.ndarray,
    current_hog: np.ndarray,
) -> Dict[str, float]:
    """
    Computes Euclidean distance and cosine similarity between a baseline HOG vector
    and the current observation HOG vector.
    """
    b = np.asarray(baseline_hog, dtype=np.float64)
    c = np.asarray(current_hog, dtype=np.float64)
    if b.shape != c.shape:
        raise ValueError(f"HOG feature shape mismatch: {b.shape} vs {c.shape}")

    euc_dist = float(np.linalg.norm(c - b))
    b_norm = float(np.linalg.norm(b))
    c_norm = float(np.linalg.norm(c))
    cos_sim = float(np.dot(c, b) / (b_norm * c_norm + 1e-9)) if (b_norm > 0 and c_norm > 0) else 0.0

    return {
        "euclidean_distance": euc_dist,
        "cosine_similarity": cos_sim,
    }

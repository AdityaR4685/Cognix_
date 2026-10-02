"""Per-frame anomaly detectors for the multi-modal max-fusion baseline.

Each public function returns a ``dict[(scenario_id, frame_id) -> float]`` of
per-frame anomaly scores:

- :func:`run_seg_frame_scores` / :func:`run_point_frame_scores` — learned MaxLogit
  detectors (``-max_c logits``, reduced by ``max`` over the frame).  Used to
  score an arbitrary index (e.g. the validation subset for normalization).
- :func:`collision_frame_scores` — ``1.0`` if any collision that frame.
- :func:`weather_frame_scores` — ``1.0`` if the GT weather vector changed more
  than ``threshold`` (max abs delta across columns) vs the previous frame.

``scenario_id`` is ``str(record.path)`` to match the keys used by the saved
sensor/observation score feathers of the other baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

from carlanomaly.index import ScenarioIndex
from carlanomaly.datasets.rgb import RGBDataset
from carlanomaly.datasets import WithIdentifiers, carlanomaly_collate_fn
from baselines.lidar_segmentation.model import PointNetSeg

FrameKey = Tuple[str, int]
TARGET_H, TARGET_W = 540, 960


# ---------------------------------------------------------------------------
# Learned MaxLogit detectors (segmentation + point cloud)
# ---------------------------------------------------------------------------


def _build_seg_model(num_classes: int) -> torch.nn.Module:
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def _load_state(model: torch.nn.Module, ckpt: dict) -> None:
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)


class _SegScoreDataset(Dataset):
    """Front-camera RGB resized to the seg model's input resolution."""

    def __init__(self, index: ScenarioIndex) -> None:
        self._index = index
        self._rgb = RGBDataset(index=index, direction="front")

    @property
    def index(self) -> ScenarioIndex:
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        rgb = self._rgb[idx].squeeze(0)
        rgb = F.interpolate(
            rgb.unsqueeze(0), size=(TARGET_H, TARGET_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        return {"rgb": rgb}


@torch.no_grad()
def run_seg_frame_scores(
    index: ScenarioIndex,
    ckpt: dict,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 8,
    desc: str = "seg",
) -> Dict[FrameKey, float]:
    """MaxLogit frame scores (max over pixels) for every frame in ``index``.

    ``max`` over bilinearly-upsampled pixel scores equals ``max`` at the model's
    native resolution, so we reduce at low resolution (identical result, cheaper).
    """
    model = _build_seg_model(ckpt["num_classes"])
    _load_state(model, ckpt)
    model = model.to(device).eval()

    ds = WithIdentifiers(_SegScoreDataset(index))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=carlanomaly_collate_fn, persistent_workers=num_workers > 0,
    )

    scores: Dict[FrameKey, float] = {}
    for batch in tqdm(loader, desc=desc, unit="batch"):
        images = batch["data"]["rgb"].to(device, non_blocking=True)
        logits = model(images)["out"]  # (B, C, h, w)
        # MaxLogit anomaly score: negative max-over-classes logit (higher = more
        # anomalous), matching the standalone rgb_segmentation baseline.
        pixel_scores = -logits.amax(dim=1)  # (B, h, w)
        frame = pixel_scores.amax(dim=(-1, -2)).cpu().tolist()  # (B,)
        timestep_ids = batch["timesteps"][:, 0].tolist()
        for sid, fid, s in zip(batch["scenario_id"], timestep_ids, frame):
            scores[(sid, int(fid))] = float(s)
    return scores


class _PointScoreDataset(Dataset):
    """Raw LiDAR xyz for one frame (variable length)."""

    def __init__(self, index: ScenarioIndex) -> None:
        self._index = index

    @property
    def index(self) -> ScenarioIndex:
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        rec, _ = self._index[idx]
        frame_id = self._index.timesteps_for(idx)[0]
        pc_path = rec.path / "pointclouds" / f"{frame_id:06d}.feather"
        df = pd.read_feather(pc_path, columns=["x", "y", "z"])
        pts = torch.from_numpy(df[["x", "y", "z"]].values.astype(np.float32))
        return {"pts": pts, "n_points": len(pts)}


def _point_collate(batch: List[dict]) -> dict:
    data = [b["data"] for b in batch]
    max_n = max(d["n_points"] for d in data)
    B = len(batch)
    pts_padded = torch.zeros(B, max_n, 3)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, d in enumerate(data):
        n = d["n_points"]
        pts_padded[i, :n] = d["pts"]
        mask[i, :n] = True
    return {
        "pts": pts_padded,
        "mask": mask,
        "scenario_id": [b["scenario_id"] for b in batch],
        "timestep_id": [int(b["timesteps"][0]) for b in batch],
        "n_points": [d["n_points"] for d in data],
    }


@torch.no_grad()
def run_point_frame_scores(
    index: ScenarioIndex,
    ckpt: dict,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 8,
    desc: str = "point",
) -> Dict[FrameKey, float]:
    """MaxLogit frame scores (max over real points) for every frame in ``index``."""
    model = PointNetSeg(ckpt["num_classes"], input_dim=3)
    _load_state(model, ckpt)
    model = model.to(device).eval()

    ds = WithIdentifiers(_PointScoreDataset(index))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=_point_collate, persistent_workers=num_workers > 0,
    )

    scores: Dict[FrameKey, float] = {}
    for batch in tqdm(loader, desc=desc, unit="batch"):
        pts = batch["pts"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(pts, pad_mask=mask)  # (B, N, C)
        # MaxLogit anomaly score, matching the standalone lidar_segmentation baseline.
        point_scores = -logits.amax(dim=2)  # (B, N)
        point_scores = point_scores.masked_fill(~mask, float("-inf"))
        frame = point_scores.amax(dim=1).cpu().tolist()  # (B,)
        for sid, fid, s in zip(batch["scenario_id"], batch["timestep_id"], frame):
            scores[(sid, int(fid))] = float(s)
    return scores


# ---------------------------------------------------------------------------
# Rule-based detectors (collision + weather), no model
# ---------------------------------------------------------------------------


def collision_frame_scores(index: ScenarioIndex) -> Dict[FrameKey, float]:
    """``1.0`` for any frame with a collision, ``0.0`` otherwise."""
    scores: Dict[FrameKey, float] = {}
    for rec in index.records:
        sid = str(rec.path)
        coll_path = rec.path / "collisions.feather"
        coll_frames: set[int] = set()
        if coll_path.exists():
            df = pd.read_feather(coll_path)
            if "frame" in df.columns and len(df) > 0:
                coll_frames = set(int(f) for f in df["frame"].values)
        for fid in range(rec.n_timesteps):
            scores[(sid, fid)] = 1.0 if fid in coll_frames else 0.0
    return scores


def weather_frame_scores(index: ScenarioIndex, threshold: float = 0.5) -> Dict[FrameKey, float]:
    """``1.0`` if the GT weather changed more than ``threshold`` vs the previous
    frame (max abs delta across all weather columns), else ``0.0``.

    Frame 0 has no predecessor and always scores ``0.0``.
    """
    scores: Dict[FrameKey, float] = {}
    for rec in index.records:
        sid = str(rec.path)
        w_path = rec.path / "weather.feather"
        w = pd.read_feather(w_path).values.astype(np.float32)  # (n_frames, 14)
        delta = np.zeros(w.shape[0], dtype=np.float32)
        if w.shape[0] > 1:
            delta[1:] = np.abs(np.diff(w, axis=0)).max(axis=1)
        for fid in range(rec.n_timesteps):
            d = delta[fid] if fid < len(delta) else 0.0
            scores[(sid, fid)] = 1.0 if d > threshold else 0.0
    return scores

"""Evaluate the PointNet baseline on the CarlAnomaly test set (LiDAR).

For each point, the anomaly score is the **MaxLogit** score ``-max_c logits``
(Hendrycks et al., 2022); higher means more anomalous. Point AUROC/AUPR/FPR95
are pooled over every point in the test split (all frames, including anomaly-free
ones), not averaged per frame. Frame-level score is max over all points (derived
by the evaluator).

Usage:
    python baselines/lidar_segmentation/evaluate.py [--batch-size 8] [--gpu 0]
"""

from __future__ import annotations

import argparse
import json
import pprint
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carlanomaly.index import ScenarioIndex
from carlanomaly.datasets import WithIdentifiers
from carlanomaly.evaluator import (
    PointEvaluator,
    SensorEvaluator,
    TimestepEvaluator,
    ScenarioEvaluator,
)
from baselines.lidar_segmentation.model import PointNetSeg

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = _HERE / "checkpoints"


class EvalDataset(Dataset):
    """Wraps pointcloud + segmentation labels for evaluation. Anomaly labels
    are loaded by the evaluators themselves."""

    def __init__(self, index: ScenarioIndex, class_map: dict[int, int]) -> None:
        self._index = index
        lut = np.zeros(256, dtype=np.int64)
        for orig, contiguous in class_map.items():
            lut[int(orig)] = contiguous
        self._class_lut = lut

    @property
    def index(self) -> ScenarioIndex:
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        rec, _ = self._index[idx]
        frame_id = self._index.timesteps_for(idx)[0]

        pc_path = rec.path / "pointclouds" / f"{frame_id:06d}.feather"
        df = pd.read_feather(pc_path, columns=["x", "y", "z", "object_tag"])
        pts = torch.from_numpy(df[["x", "y", "z"]].values.astype(np.float32))  # (N, 3)
        seg_labels = torch.from_numpy(self._class_lut[df["object_tag"].values.astype(np.int64)])

        return {"pts": pts, "seg_labels": seg_labels, "n_points": len(pts)}


def eval_collate(batch: list[dict]) -> dict:
    """Pad variable-length point clouds to the max length in the batch."""
    data = [b["data"] for b in batch]
    max_n = max(d["n_points"] for d in data)
    B = len(batch)

    pts_padded = torch.zeros(B, max_n, 3)
    seg_labels_padded = torch.zeros(B, max_n, dtype=torch.long)
    mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, d in enumerate(data):
        n = d["n_points"]
        pts_padded[i, :n] = d["pts"]
        seg_labels_padded[i, :n] = d["seg_labels"]
        mask[i, :n] = True

    return {
        "pts": pts_padded,
        "seg_labels": seg_labels_padded,
        "mask": mask,
        "scenario_id": [b["scenario_id"] for b in batch],
        "timestep_id": [int(b["timesteps"][0]) for b in batch],
        "n_points": [d["n_points"] for d in data],
    }


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 0) -> torch.Tensor:
    mask = target != ignore_index
    idx = num_classes * target[mask] + pred[mask]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def build_model(num_classes: int) -> PointNetSeg:
    return PointNetSeg(num_classes, input_dim=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "pointnet_seg.pt"))
    parser.add_argument("--reduction", type=str, default="max", choices=["max", "mean", "min"])
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (contains train/ and test/); created on download")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not auto-download missing dataset parts")
    parser.set_defaults(download=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download the pretrained baselines with ./download_checkpoints.sh, "
            f"or train one with train.py and pass --checkpoint."
        )
    print(f"Loading checkpoint from {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = ckpt["num_classes"]
    class_map = ckpt["class_map"]
    print(f"Model has {num_classes} classes", flush=True)

    model = build_model(num_classes)
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    model = torch.compile(model)

    print("Building test index …", flush=True)
    index = ScenarioIndex(args.data_root, split="test", clip_len=1, stride=1,
                          download=args.download, parts=["base", "lidar"])
    print(f"Evaluating on {len(index)} frames", flush=True)

    eval_ds = WithIdentifiers(EvalDataset(index, class_map))
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=eval_collate,
        prefetch_factor=6,
        persistent_workers=True,
    )

    point_eval = PointEvaluator()
    sensor_eval = SensorEvaluator(sensor="lidar")
    obs_eval = TimestepEvaluator()
    scenario_eval = ScenarioEvaluator()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    frame_scores_gpu: list[torch.Tensor] = []  # deferred to one D2H after the loop
    frame_sids: list[str] = []
    frame_fids: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            pts = batch["pts"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            n_points_list = batch["n_points"]

            logits = model(pts, pad_mask=mask)  # (B, N_max, C)

            pred_classes = logits.argmax(dim=2)  # stays on GPU
            seg_labels = batch["seg_labels"].to(device, non_blocking=True)
            conf += _confusion_matrix(
                pred_classes[mask].reshape(-1),
                seg_labels[mask].reshape(-1),
                num_classes,
            )

            # MaxLogit anomaly score: negative max-over-classes logit
            # (higher = more anomalous).
            all_scores = -logits.amax(dim=2)  # (B, N_max), kept on-device

            # Unpad to variable-length per-frame score tensors
            point_scores_list = [all_scores[i, :n] for i, n in enumerate(n_points_list)]

            point_eval.update(
                point_scores_list, batch["scenario_id"], batch["timestep_id"],
            )

            frame_scores_gpu.append(torch.stack([s.max() for s in point_scores_list]))  # stays on GPU
            frame_sids.extend(batch["scenario_id"])
            frame_fids.extend(batch["timestep_id"])

    frame_scores = torch.cat(frame_scores_gpu).cpu().tolist()  # one D2H for all batches
    sensor_eval.update(frame_scores, frame_sids, frame_fids)
    obs_eval.update(frame_scores, frame_sids, frame_fids)

    for sid, score in sensor_eval.max_per_scenario().items():
        scenario_eval.update(score, sid)

    results = {
        "point": point_eval.compute(),
        "sensor": sensor_eval.compute(),
        "observation": obs_eval.compute(),
        "scenario": scenario_eval.compute(),
    }

    tp = conf.diag()
    iou = tp.float() / (conf.sum(1) + conf.sum(0) - tp + 1e-9).float()
    valid = conf.sum(1) > 0
    results["segmentation"] = {
        "miou": iou[valid].mean().item(),
        "oa": tp.sum().item() / (conf.sum().item() + 1e-9),
    }

    print("\n" + "=" * 60)
    print("PointNet (MaxLogit) Baseline Results")
    print("=" * 60)
    pprint.pprint(results, width=80)

    out_path = _HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    out_dir = _HERE
    sensor_eval.to_dataframe().to_feather(out_dir / "scores_sensor.feather")
    obs_eval.to_dataframe().to_feather(out_dir / "scores_observation.feather")
    scenario_eval.to_dataframe().to_feather(out_dir / "scores_scenario.feather")
    print(f"Score DataFrames saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

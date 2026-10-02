"""Train a PointNet for semantic segmentation on CarlAnomaly LiDAR data.

At test time the MSP baseline scores each point as 1 - max(softmax(logits)).

Usage:
    python baselines/lidar_segmentation/train.py [--epochs 1] [--batch-size 16] [--gpu 0]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carlanomaly.index import ScenarioIndex
from carlanomaly.datasets.pointcloud import PointCloudDataset
from baselines.lidar_segmentation.model import PointNetSeg

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent
# Class map is produced by the RGB-segmentation baseline (discover_classes.py)
# and shared here so LiDAR labels use the same contiguous class IDs.
CLASS_MAP_PATH = _HERE.parent / "rgb_segmentation" / "class_map.json"
CHECKPOINT_DIR = _HERE / "checkpoints"


# ─── augmentations ────────────────────────────────────────────────

def rand_rotate_z(pts: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0, 2 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pts.dtype)
    return pts @ R.T


def rand_scale(pts: np.ndarray, lo: float = 0.95, hi: float = 1.05) -> np.ndarray:
    return pts * np.random.uniform(lo, hi)


def jitter(pts: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    return pts + np.random.normal(scale=sigma, size=pts.shape).astype(pts.dtype)


def augment(pts: np.ndarray) -> np.ndarray:
    pts = rand_rotate_z(pts)
    pts = rand_scale(pts)
    pts = jitter(pts)
    return pts


# ─── dataset ──────────────────────────────────────────────────────

class PointCloudTrainDataset(Dataset):
    """Wraps PointCloudDataset with subsampling and class remapping."""

    def __init__(
        self,
        pc_ds: PointCloudDataset,
        class_map: dict[int, int],
        num_points: int = 32_768,
        train: bool = True,
    ) -> None:
        self._pc = pc_ds
        self._num_points = num_points
        self._train = train
        self._lut = self._build_lut(class_map)

    @staticmethod
    def _build_lut(class_map: dict[int, int]) -> np.ndarray:
        lut = np.zeros(256, dtype=np.int64)
        for orig, contiguous in class_map.items():
            lut[int(orig)] = contiguous
        return lut

    def __len__(self) -> int:
        return len(self._pc)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self._pc[idx]  # List[pd.DataFrame] of length T=1
        df = frames[0]

        pts = df[["x", "y", "z"]].values.astype(np.float32)
        raw_tags = df["object_tag"].values.astype(np.int64)
        labels = self._lut[raw_tags]

        N = len(pts)
        sel = np.random.choice(N, self._num_points, replace=(N < self._num_points))
        pts = pts[sel]
        labels = labels[sel]

        if self._train:
            pts = augment(pts)

        return torch.from_numpy(pts), torch.from_numpy(labels)


class PointCloudValDataset(Dataset):
    """Full point clouds (no subsampling) with class remapping for validation."""

    def __init__(self, pc_ds: PointCloudDataset, class_map: dict[int, int]) -> None:
        self._pc = pc_ds
        self._lut = PointCloudTrainDataset._build_lut(class_map)

    def __len__(self) -> int:
        return len(self._pc)

    def __getitem__(self, idx: int) -> dict:
        frames = self._pc[idx]
        df = frames[0]
        pts = df[["x", "y", "z"]].values.astype(np.float32)
        raw_tags = df["object_tag"].values.astype(np.int64)
        labels = self._lut[raw_tags]
        return {
            "pts": torch.from_numpy(pts),
            "labels": torch.from_numpy(labels),
            "n_points": len(pts),
        }


def val_collate(batch: list[dict]) -> dict:
    max_n = max(b["n_points"] for b in batch)
    B = len(batch)
    pts_padded = torch.zeros(B, max_n, 3)
    labels_padded = torch.zeros(B, max_n, dtype=torch.long)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["n_points"]
        pts_padded[i, :n] = b["pts"]
        labels_padded[i, :n] = b["labels"]
        mask[i, :n] = True
    return {"pts": pts_padded, "labels": labels_padded, "mask": mask}


def build_model(num_classes: int) -> PointNetSeg:
    return PointNetSeg(num_classes, input_dim=3)


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 0) -> torch.Tensor:
    mask = target != ignore_index
    idx = num_classes * target[mask] + pred[mask]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _split_by_scenario(index: ScenarioIndex, val_fraction: float, seed: int = 42) -> tuple[list[int], list[int]]:
    n_val = max(1, int(len(index.records) * val_fraction))
    rng = random.Random(seed)
    val_rec_idxs = set(rng.sample(range(len(index.records)), n_val))
    train_items, val_items = [], []
    for i, (rec_idx, _) in enumerate(index._index):
        (val_items if rec_idx in val_rec_idxs else train_items).append(i)
    return train_items, val_items


@torch.no_grad()
def validate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, num_classes: int,
             writer: torch.utils.tensorboard.SummaryWriter | None = None, global_step: int = 0) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    n_batches = 0

    sample_pred, sample_target = None, None

    for batch in tqdm(val_loader, desc="Validating", unit="batch", leave=False):
        pts = batch["pts"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(pts, pad_mask=mask)  # (B, N_max, C)

        logits_real = logits[mask]   # (total_real, C)
        labels_real = labels[mask]   # (total_real,)

        loss = F.cross_entropy(logits_real, labels_real, ignore_index=0)
        total_loss += loss.item()
        n_batches += 1

        pred_real = logits_real.argmax(dim=1).cpu()
        conf += _confusion_matrix(pred_real, labels_real.cpu(), num_classes)

        if sample_pred is None:
            n = mask[0].sum().item()
            sample_pred = logits[0, :n].argmax(dim=1).cpu()
            sample_target = labels[0, :n].cpu()

    model.train()

    tp = conf.diag()
    iou = tp.float() / (conf.sum(1) + conf.sum(0) - tp + 1e-9).float()
    valid = conf.sum(1) > 0
    miou = iou[valid].mean().item()
    oa = tp.sum().item() / (conf.sum().item() + 1e-9)

    if writer is not None and sample_pred is not None:
        writer.add_histogram("val/pred_classes", sample_pred.float(), global_step, bins=num_classes)
        writer.add_histogram("val/target_classes", sample_target.float(), global_step, bins=num_classes)

    return {
        "loss": total_loss / max(n_batches, 1),
        "oa": oa,
        "miou": miou,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=32_768)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-dir", type=str, default=str(_HERE / "runs"))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (contains train/ and test/); created on download")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not auto-download missing dataset parts")
    parser.set_defaults(download=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    with open(CLASS_MAP_PATH) as f:
        class_map = {int(k): v for k, v in json.load(f).items()}
    num_classes = max(class_map.values()) + 1
    print(f"Class map: {len(class_map)} original IDs → {num_classes} contiguous classes", flush=True)

    print("Building ScenarioIndex …", flush=True)
    index = ScenarioIndex(args.data_root, split="train", clip_len=1, stride=1,
                          download=args.download, parts=["base", "lidar"])

    train_items, val_items = _split_by_scenario(index, args.val_fraction)
    print(f"Train: {len(train_items)} frames, Val: {len(val_items)} frames", flush=True)

    pc_ds = PointCloudDataset(index=index)
    full_ds = PointCloudTrainDataset(pc_ds, class_map, num_points=args.num_points, train=True)
    val_ds = PointCloudValDataset(pc_ds, class_map)

    train_loader = DataLoader(
        Subset(full_ds, train_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_items),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_collate,
    )

    model = build_model(num_classes).to(device)
    model = torch.compile(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    run_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(run_dir))
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        running_loss = 0.0
        for step, (pts, targets) in enumerate(pbar):
            pts = pts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(pts)  # (B, N, C)
            loss = F.cross_entropy(
                logits.reshape(-1, num_classes),
                targets.reshape(-1),
                ignore_index=0,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss = 0.95 * running_loss + 0.05 * loss.item() if step > 0 else loss.item()
            pbar.set_postfix(loss=f"{running_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if global_step % args.val_every == 0:
                metrics = validate(model, val_loader, device, num_classes, writer, global_step)
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, global_step)
                tqdm.write(f"  [step {global_step}] val loss={metrics['loss']:.4f} OA={metrics['oa']:.4f} mIoU={metrics['miou']:.4f}")

    writer.close()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "pointnet_seg.pt"
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "num_classes": num_classes,
        "class_map": class_map,
        "num_points": args.num_points,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()

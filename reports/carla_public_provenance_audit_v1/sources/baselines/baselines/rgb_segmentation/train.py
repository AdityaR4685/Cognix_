"""Train a DeepLabV3-ResNet50 for semantic segmentation on CarlAnomaly (front camera).

Usage:
    python baselines/rgb_segmentation/train.py [--epochs 1] [--batch-size 16] [--lr 1e-4] [--gpu 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carlanomaly.index import ScenarioIndex
from carlanomaly.datasets.rgb import RGBDataset
from carlanomaly.datasets.segmentation import SegmentationDataset

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent
CLASS_MAP_PATH = _HERE / "class_map.json"
CHECKPOINT_DIR = _HERE / "checkpoints"

TARGET_H, TARGET_W = 540, 960


class SegTrainDataset(Dataset):
    """Wraps RGB + Segmentation datasets with resize and class remapping."""

    def __init__(self, rgb_ds: RGBDataset, seg_ds: SegmentationDataset, class_map: dict[int, int]) -> None:
        self._rgb = rgb_ds
        self._seg = seg_ds
        self._class_map = class_map
        self._map_tensor = self._build_map_tensor()

    def _build_map_tensor(self) -> torch.Tensor:
        lut = torch.zeros(256, dtype=torch.long)
        for orig, contiguous in self._class_map.items():
            lut[int(orig)] = contiguous
        return lut

    def __len__(self) -> int:
        return len(self._rgb)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = self._rgb[idx]          # (1, 3, H, W)
        seg = self._seg[idx]          # dict with 'semantic': (1, H, W)
        semantic = seg["semantic"]

        rgb = rgb.squeeze(0)          # (3, H, W)
        semantic = semantic.squeeze(0)  # (H, W)

        rgb = F.interpolate(rgb.unsqueeze(0), size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False).squeeze(0)
        semantic = F.interpolate(
            semantic.float().unsqueeze(0).unsqueeze(0),
            size=(TARGET_H, TARGET_W),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        semantic = self._map_tensor[semantic]

        return rgb, semantic


def build_model(num_classes: int) -> torch.nn.Module:
    model = deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


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

    sample_image, sample_pred, sample_target = None, None, None

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)["out"]
        loss = F.cross_entropy(logits, targets, ignore_index=0)
        total_loss += loss.item()
        n_batches += 1

        pred = logits.argmax(dim=1).cpu()
        conf += _confusion_matrix(pred.reshape(-1), targets.cpu().reshape(-1), num_classes)

        if sample_image is None:
            sample_image = images[0].cpu()
            sample_pred = pred[0]
            sample_target = targets[0].cpu()

    model.train()

    tp = conf.diag()
    iou = tp.float() / (conf.sum(1) + conf.sum(0) - tp + 1e-9).float()
    valid = conf.sum(1) > 0
    miou = iou[valid].mean().item()
    oa = tp.sum().item() / (conf.sum().item() + 1e-9)

    if writer is not None and sample_pred is not None:
        pred_colored = torch.zeros(3, *sample_pred.shape, dtype=torch.uint8)
        target_colored = torch.zeros(3, *sample_target.shape, dtype=torch.uint8)
        colors = torch.randint(0, 256, (num_classes, 3), dtype=torch.uint8)
        for c in range(num_classes):
            pred_colored[:, sample_pred == c] = colors[c].unsqueeze(1)
            target_colored[:, sample_target == c] = colors[c].unsqueeze(1)

        stack = torch.stack([sample_image.clamp(0, 1), pred_colored.float() / 255, target_colored.float() / 255])
        writer.add_images("val/image_pred_target", stack, global_step)

    return {
        "loss": total_loss / max(n_batches, 1),
        "oa": oa,
        "miou": miou,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
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
                          download=args.download, parts=["base"])

    train_items, val_items = _split_by_scenario(index, args.val_fraction)
    print(f"Train: {len(train_items)} frames, Val: {len(val_items)} frames", flush=True)

    rgb_ds = RGBDataset(index=index, direction="front")
    seg_ds = SegmentationDataset(index=index, direction="front")
    full_ds = SegTrainDataset(rgb_ds, seg_ds, class_map)

    train_loader = DataLoader(
        Subset(full_ds, train_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(num_classes).to(device)
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
        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            out = model(images)
            logits = out["out"]
            loss = F.cross_entropy(logits, targets, ignore_index=0)

            if "aux" in out:
                aux_loss = F.cross_entropy(out["aux"], targets, ignore_index=0)
                loss = loss + 0.4 * aux_loss

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
    ckpt_path = CHECKPOINT_DIR / "deeplabv3_r50_msp.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_map": class_map,
        "target_size": (TARGET_H, TARGET_W),
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()

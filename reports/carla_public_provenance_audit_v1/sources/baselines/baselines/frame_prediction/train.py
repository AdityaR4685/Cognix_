"""Train a U-Net for future frame prediction on CarlAnomaly (front camera).

The model takes T context frames and predicts the next frame.
At test time, per-pixel prediction error is used as the anomaly score.

Usage (single GPU):
    python baselines/frame_prediction/train.py [--epochs 1] [--batch-size 8] [--gpu 0]

Usage (multi-GPU via DDP):
    torchrun --nproc_per_node=4 baselines/frame_prediction/train.py [--epochs 1] [--batch-size 8]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carlanomaly.index import ScenarioIndex
from carlanomaly.datasets.rgb import RGBDataset
from carlanomaly.download import ensure_parts, splits_for
from baselines.frame_prediction.model import UNetFFP

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = _HERE / "checkpoints"

TARGET_H, TARGET_W = 540, 960


class FFPDataset(Dataset):
    """Wraps RGBDataset: first T frames → input, last frame → target."""

    def __init__(self, rgb_ds: RGBDataset, context_frames: int) -> None:
        self._rgb = rgb_ds
        self._T = context_frames

    def __len__(self) -> int:
        return len(self._rgb)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip = self._rgb[idx]  # (T+1, 3, H, W) in [0, 1]

        clip = F.interpolate(
            clip, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False,
        )

        context = clip[: self._T].reshape(-1, TARGET_H, TARGET_W)  # (3T, H, W)
        target = clip[self._T]  # (3, H, W)
        return context, target


def gradient_diff_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def _split_by_scenario(index: ScenarioIndex, val_fraction: float, seed: int = 42) -> tuple[list[int], list[int]]:
    n_val = max(1, int(len(index.records) * val_fraction))
    rng = random.Random(seed)
    val_rec_idxs = set(rng.sample(range(len(index.records)), n_val))
    train_items, val_items = [], []
    for i, (rec_idx, _) in enumerate(index._index):
        (val_items if rec_idx in val_rec_idxs else train_items).append(i)
    return train_items, val_items


@torch.no_grad()
def validate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device,
             gdl_weight: float, writer: SummaryWriter | None, global_step: int,
             rank: int = 0, ddp: bool = False) -> dict[str, float]:
    model.eval()
    total_l2 = 0.0
    total_gdl = 0.0
    n_batches = 0

    sample_pred, sample_target = None, None

    for context, target in tqdm(val_loader, desc="Validating", unit="batch", leave=False, disable=(rank != 0)):
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        pred = model(context)
        l2 = F.mse_loss(pred, target)
        gdl = gradient_diff_loss(pred, target)

        total_l2 += l2.item()
        total_gdl += gdl.item()
        n_batches += 1

        if sample_pred is None:
            sample_pred = pred[0].cpu()
            sample_target = target[0].cpu()

    if ddp:
        stats = torch.tensor([total_l2, total_gdl, float(n_batches)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_l2, total_gdl, n_batches = stats.tolist()

    model.train()

    avg_mse = total_l2 / max(n_batches, 1)
    psnr = 10 * math.log10(1.0 / max(avg_mse, 1e-10))

    if writer is not None and sample_pred is not None:
        pair = torch.stack([sample_target.clamp(0, 1), sample_pred.clamp(0, 1)])
        writer.add_images("val/target_vs_pred", pair, global_step)

    return {
        "loss": avg_mse + gdl_weight * total_gdl / max(n_batches, 1),
        "l2": avg_mse,
        "gdl": total_gdl / max(n_batches, 1),
        "psnr": psnr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--gdl-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-dir", type=str, default=str(_HERE / "runs"))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (contains train/ and test/); created on download")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not auto-download missing dataset parts")
    parser.set_defaults(download=True)
    args = parser.parse_args()

    # ── DDP or single-GPU setup ───────────────────────────────────────
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    clip_len = args.context_frames + 1
    if rank == 0:
        print(f"Context frames: {args.context_frames}, clip_len: {clip_len}", flush=True)
        print(f"World size: {world_size}, batch per GPU: {args.batch_size}", flush=True)

    # Download on rank 0 only, then barrier so other ranks discover a complete
    # dataset (avoids concurrent writes to the same archive).
    if args.download and rank == 0:
        ensure_parts(args.data_root, ["base"], splits_for("train"))
    if ddp:
        dist.barrier()

    if rank == 0:
        print("Building ScenarioIndex …", flush=True)
    index = ScenarioIndex(args.data_root, split="train", clip_len=clip_len, stride=1)

    train_items, val_items = _split_by_scenario(index, args.val_fraction)
    if rank == 0:
        print(f"Train: {len(train_items)} clips, Val: {len(val_items)} clips", flush=True)

    rgb_ds = RGBDataset(index=index, direction="front")
    full_ds = FFPDataset(rgb_ds, context_frames=args.context_frames)

    train_subset = Subset(full_ds, train_items)
    val_subset = Subset(full_ds, val_items)

    sampler = DistributedSampler(train_subset, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_sampler = DistributedSampler(val_subset, shuffle=False) if ddp else None
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=min(args.num_workers, 4),
        pin_memory=True,
    )

    model = UNetFFP(context_frames=args.context_frames).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    model = torch.compile(model)
    model.train()

    # Unwrapped model reference for validation (avoids DDP forward-sync deadlock)
    val_model = model
    if hasattr(val_model, '_orig_mod'):
        val_model = val_model._orig_mod
    if isinstance(val_model, DDP):
        val_model = val_model.module

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    run_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
    writer = SummaryWriter(log_dir=str(run_dir)) if rank == 0 else None
    global_step = 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch",
                    disable=(rank != 0))
        running_loss = 0.0
        for step, (context, target) in enumerate(pbar):
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(context)
            loss_l2 = F.mse_loss(pred, target)
            loss_gdl = gradient_diff_loss(pred, target)
            loss = loss_l2 + args.gdl_weight * loss_gdl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss = 0.95 * running_loss + 0.05 * loss.item() if step > 0 else loss.item()
            pbar.set_postfix(
                loss=f"{running_loss:.4f}",
                l2=f"{loss_l2.item():.4f}",
                gdl=f"{loss_gdl.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/l2", loss_l2.item(), global_step)
                writer.add_scalar("train/gdl", loss_gdl.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if global_step % args.val_every == 0:
                metrics = validate(val_model, val_loader, device, args.gdl_weight, writer, global_step, rank, ddp)
                if rank == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, global_step)
                    tqdm.write(f"  [step {global_step}] val loss={metrics['loss']:.4f} PSNR={metrics['psnr']:.2f}dB")

    if writer is not None:
        writer.close()

    if rank == 0:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CHECKPOINT_DIR / "unet_ffp.pt"
        raw_model = model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if isinstance(raw_model, DDP):
            raw_model = raw_model.module
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "context_frames": args.context_frames,
            "target_size": (TARGET_H, TARGET_W),
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

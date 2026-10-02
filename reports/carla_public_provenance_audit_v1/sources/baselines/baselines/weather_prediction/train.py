"""Train a ResNet-18 to predict weather parameters from front-camera images.

Usage:
    python baselines/weather_prediction/train.py [--epochs 5] [--batch-size 64] [--gpu 0]
    python baselines/weather_prediction/train.py --cache-root ./weather_cache_rgb224
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize, center_crop, normalize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carlanomaly.index import ScenarioIndex
from carlanomaly.datasets.weather import WeatherDataset
from baselines.weather_prediction.model import WeatherPredictor

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = _HERE / "checkpoints"

TARGET_COLS = ["cloudiness", "precipitation", "sun_altitude_angle", "fog_density", "dust_storm"]
TARGET_IDX = [0, 1, 2, 4, 13]
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class WeatherDatasetPair(Dataset):
    """RGB frame (front camera) + weather targets.

    When cache_root is set, loads pre-resized 224×224 JPEGs from fast storage
    instead of decoding full-res frames from slow HDD.
    """

    def __init__(
        self,
        index: ScenarioIndex,
        weather_ds: WeatherDataset,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        data_root: Path,
        cache_root: Path | None = None,
    ) -> None:
        self._index = index
        self._weather = weather_ds
        self._mean = target_mean
        self._std = target_std
        self._data_root = data_root
        self._cache_root = cache_root

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec, _ = self._index[idx]
        frame = self._index.timesteps_for(idx)[0]

        weather = self._weather[idx].squeeze(0)
        target = weather[TARGET_IDX]
        target = (target - self._mean) / (self._std + 1e-8)

        cached_path = (
            self._cache_root / rec.path.relative_to(self._data_root) / "rgb-front" / f"{frame:06d}.jpg"
            if self._cache_root is not None else None
        )
        if cached_path is not None and cached_path.exists():
            img = read_image(str(cached_path), ImageReadMode.RGB).float().div_(255.0)
        else:
            img_path = rec.path / "rgb-front" / f"{frame:06d}.jpg"
            arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)
            img = resize(img, [IMG_SIZE], antialias=True)
            img = center_crop(img, [IMG_SIZE, IMG_SIZE])

        img = normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        return img, target


def _split_by_scenario(index: ScenarioIndex, val_fraction: float, seed: int = 42) -> tuple[list[int], list[int]]:
    n_val = max(1, int(len(index.records) * val_fraction))
    rng = random.Random(seed)
    val_rec_idxs = set(rng.sample(range(len(index.records)), n_val))
    train_items, val_items = [], []
    for i, (rec_idx, _) in enumerate(index._index):
        (val_items if rec_idx in val_rec_idxs else train_items).append(i)
    return train_items, val_items


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            preds = model(imgs)
        total_loss += F.mse_loss(preds.float(), targets).item() * imgs.size(0)
        n += imgs.size(0)
    model.train()
    return total_loss / max(n, 1)


def compute_train_stats(index: ScenarioIndex) -> tuple[torch.Tensor, torch.Tensor]:
    weather_ds = WeatherDataset(index=index)
    vals = []
    print("Computing target normalisation stats …", flush=True)
    for i in tqdm(range(len(weather_ds)), unit="clip", leave=False):
        w = weather_ds[i].squeeze(0)
        vals.append(w[TARGET_IDX])
    stacked = torch.stack(vals)
    return stacked.mean(0), stacked.std(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--cache-root", type=str, default=None,
                        help="optional fast-storage cache of pre-resized 224x224 frames "
                             "(see cache_images.py); falls back to the slow path if unset")
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

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root) if args.cache_root else None
    if cache_root is not None and not cache_root.exists():
        print(f"Warning: cache root {cache_root} not found — falling back to slow path. "
              f"Run cache_images.py first for best performance.", flush=True)
        cache_root = None

    print("Building ScenarioIndex …", flush=True)
    index = ScenarioIndex(data_root, split="train", clip_len=1, stride=1,
                          download=args.download, parts=["base"])
    print(f"Total frames: {len(index)}", flush=True)

    target_mean, target_std = compute_train_stats(index)
    print(f"Target mean: {target_mean.tolist()}", flush=True)
    print(f"Target std:  {target_std.tolist()}", flush=True)

    train_items, val_items = _split_by_scenario(index, args.val_fraction)
    print(f"Train: {len(train_items)}, Val: {len(val_items)}", flush=True)
    print(f"Using {'cached' if cache_root else 'slow'} image loading", flush=True)

    weather_ds = WeatherDataset(index=index)
    full_ds = WeatherDatasetPair(index, weather_ds, target_mean, target_std, data_root, cache_root)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    train_loader = DataLoader(Subset(full_ds, train_items), shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(Subset(full_ds, val_items),   shuffle=False, **loader_kwargs)

    model = WeatherPredictor(n_outputs=len(TARGET_COLS)).to(device)
    model = torch.compile(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    run_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(run_dir))
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        running_loss = 0.0
        for step, (imgs, targets) in enumerate(pbar):
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                preds = model(imgs)
                loss  = F.mse_loss(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            running_loss = 0.95 * running_loss + 0.05 * loss.item() if step > 0 else loss.item()
            pbar.set_postfix(loss=f"{running_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if global_step % args.val_every == 0:
                val_loss = validate(model, val_loader, device)
                writer.add_scalar("val/loss", val_loss, global_step)
                tqdm.write(f"  [step {global_step}] val MSE (normalised) = {val_loss:.4f}")

    writer.close()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "weather_resnet18.pt"
    torch.save(
        {
            "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
            "target_cols": TARGET_COLS,
            "target_idx": TARGET_IDX,
            "target_mean": target_mean,
            "target_std": target_std,
            "img_size": IMG_SIZE,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}", flush=True)

    stats_path = CHECKPOINT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "target_cols": TARGET_COLS,
                "target_idx": TARGET_IDX,
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
            },
            f, indent=2,
        )
    print(f"Stats saved to {stats_path}", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate weather predictor on test/normal split. Reports per-column MAE/RMSE.

Usage:
    python baselines/weather_prediction/evaluate.py [--gpu 0] [--batch-size 128]
    python baselines/weather_prediction/evaluate.py --cache-root ./weather_cache_rgb224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
CKPT_PATH = CHECKPOINT_DIR / "weather_resnet18.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class WeatherEvalPair(Dataset):
    def __init__(
        self,
        index: ScenarioIndex,
        weather_ds: WeatherDataset,
        target_idx: list[int],
        img_size: int,
        data_root: Path,
        cache_root: Path | None = None,
    ) -> None:
        self._index = index
        self._weather = weather_ds
        self._target_idx = target_idx
        self._img_size = img_size
        self._data_root = data_root
        self._cache_root = cache_root

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec, _ = self._index[idx]
        frame = self._index.timesteps_for(idx)[0]
        weather = self._weather[idx].squeeze(0)
        target = weather[self._target_idx]

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
            img = resize(img, [self._img_size], antialias=True)
            img = center_crop(img, [self._img_size, self._img_size])

        img = normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        return img, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--cache-root", type=str, default=None,
                        help="optional fast-storage cache of pre-resized 224x224 frames "
                             "(see cache_images.py); falls back to the slow path if unset")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (contains train/ and test/); created on download")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not auto-download missing dataset parts")
    parser.set_defaults(download=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if not CKPT_PATH.exists():
        raise SystemExit(
            f"Checkpoint not found: {CKPT_PATH}\n"
            f"The weather baseline has no published checkpoint — train it with "
            f"train.py first."
        )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    target_cols: list[str] = ckpt["target_cols"]
    target_idx: list[int]  = ckpt["target_idx"]
    target_mean: torch.Tensor = ckpt["target_mean"]
    target_std: torch.Tensor  = ckpt["target_std"]
    img_size: int = ckpt["img_size"]

    model = WeatherPredictor(n_outputs=len(target_cols))
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model = torch.compile(model)
    model.eval().to(device)

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root) if args.cache_root else None
    if cache_root is not None and not cache_root.exists():
        print(f"Warning: cache root {cache_root} not found — using slow path.", flush=True)
        cache_root = None

    print("Building ScenarioIndex (test_normal) …", flush=True)
    index = ScenarioIndex(data_root, split="test_normal", clip_len=1, stride=1,
                          download=args.download, parts=["base"])
    print(f"Total frames: {len(index)}", flush=True)

    weather_ds = WeatherDataset(index=index)
    eval_ds = WeatherEvalPair(index, weather_ds, target_idx, img_size, data_root, cache_root)

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    n_cols = len(target_cols)
    sum_se = torch.zeros(n_cols)
    sum_ae = torch.zeros(n_cols)
    n = 0

    with torch.no_grad():
        for imgs, targets in tqdm(loader, unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                preds_norm = model(imgs).float().cpu()

            preds = preds_norm * (target_std + 1e-8) + target_mean
            sum_se += ((preds - targets) ** 2).sum(0)
            sum_ae += (preds - targets).abs().sum(0)
            n += imgs.size(0)

    mse  = sum_se / n
    mae  = sum_ae / n
    rmse = mse.sqrt()

    print("\nResults on test_normal split:")
    print(f"{'Column':<30} {'MAE':>10} {'RMSE':>10}")
    print("-" * 52)
    for i, col in enumerate(target_cols):
        print(f"{col:<30} {mae[i].item():>10.4f} {rmse[i].item():>10.4f}")
    print("-" * 52)
    print(f"{'Mean':<30} {mae.mean().item():>10.4f} {rmse.mean().item():>10.4f}")


if __name__ == "__main__":
    main()

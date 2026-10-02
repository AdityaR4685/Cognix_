"""Pre-resize all front-camera frames to 224×224 JPEG on fast NVMe storage.

Run once before training. Idempotent — skips already-cached files.

Usage:
    python baselines/weather_prediction/cache_images.py [--workers 32] [--quality 85]
    python baselines/weather_prediction/cache_images.py --out-root ./weather_cache_rgb224
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from torchvision.io import read_image, write_jpeg, ImageReadMode
from torchvision.transforms.functional import resize, center_crop
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path("./data")
DEFAULT_OUT = Path("./weather_cache_rgb224")
IMG_SIZE = 224
JPEG_QUALITY = 85


def _process(args: tuple[Path, Path]) -> bool:
    """Resize + crop one image; return True if written, False if skipped."""
    src, dst = args
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = read_image(str(src), ImageReadMode.RGB)
    img = center_crop(resize(img, [IMG_SIZE], antialias=True), [IMG_SIZE, IMG_SIZE])
    write_jpeg(img, str(dst), quality=JPEG_QUALITY)
    return True


def collect_paths(data_root: Path, out_root: Path) -> list[tuple[Path, Path]]:
    """Return (src, dst) pairs for all front-camera JPEGs."""
    pairs: list[tuple[Path, Path]] = []
    for jpg in data_root.rglob("rgb-front/*.jpg"):
        rel = jpg.relative_to(data_root)
        pairs.append((jpg, out_root / rel))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (contains train/ and test/); created on download")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not auto-download missing dataset parts")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY)
    parser.set_defaults(download=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.download:
        from carlanomaly.download import ensure_parts
        ensure_parts(data_root, ["base"])

    out_root = Path(args.out_root)
    pairs = collect_paths(data_root, out_root)
    print(f"Found {len(pairs)} images → {out_root}", flush=True)

    n_written = n_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process, p): p for p in pairs}
        for fut in tqdm(as_completed(futs), total=len(pairs), unit="img"):
            if fut.result():
                n_written += 1
            else:
                n_skipped += 1

    print(f"Done. Written: {n_written}, skipped (already cached): {n_skipped}", flush=True)


if __name__ == "__main__":
    main()

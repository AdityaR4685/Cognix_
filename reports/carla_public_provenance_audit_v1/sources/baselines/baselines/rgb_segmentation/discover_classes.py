"""Scan all segmentation-front PNGs to discover unique CARLA class IDs.

Builds a contiguous class mapping {original_id: contiguous_id} and saves it
as JSON.  Index 0 is reserved for void/unlabeled (original class 0).

Uses multiprocessing to parallelize the I/O-bound PNG reads.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent


def _find_seg_dirs(root: Path) -> list[Path]:
    """Find all segmentation-front directories without rglob.

    Walks the known directory structure directly so we can show progress.
    """
    seg_dirs: list[Path] = []

    # train/{town}/{scenario}/segmentation-front
    train_root = root / "train"
    if train_root.exists():
        towns = sorted(d for d in train_root.iterdir() if d.is_dir())
        print(f"  train: {len(towns)} towns", flush=True)
        for town in towns:
            scenarios = sorted(d for d in town.iterdir() if d.is_dir())
            for sc in scenarios:
                sd = sc / "segmentation-front"
                if sd.exists():
                    seg_dirs.append(sd)
        print(f"  train: {len(seg_dirs)} scenarios found", flush=True)

    # test/normal/{town}/{scenario}/segmentation-front
    n_before = len(seg_dirs)
    normal_root = root / "test" / "normal"
    if normal_root.exists():
        for town in sorted(d for d in normal_root.iterdir() if d.is_dir()):
            for sc in sorted(d for d in town.iterdir() if d.is_dir()):
                sd = sc / "segmentation-front"
                if sd.exists():
                    seg_dirs.append(sd)
    print(f"  test/normal: {len(seg_dirs) - n_before} scenarios found", flush=True)

    # test/anomaly/{town}/{atype}/{scenario}/segmentation-front
    n_before = len(seg_dirs)
    anomaly_root = root / "test" / "anomaly"
    if anomaly_root.exists():
        for town in sorted(d for d in anomaly_root.iterdir() if d.is_dir()):
            for atype in sorted(d for d in town.iterdir() if d.is_dir()):
                for sc in sorted(d for d in atype.iterdir() if d.is_dir()):
                    sd = sc / "segmentation-front"
                    if sd.exists():
                        seg_dirs.append(sd)
    print(f"  test/anomaly: {len(seg_dirs) - n_before} scenarios found", flush=True)

    return seg_dirs


def _collect_png_paths(seg_dirs: list[Path]) -> list[Path]:
    """List PNGs in each segmentation-front dir with a progress bar."""
    paths: list[Path] = []
    for seg_dir in tqdm(seg_dirs, desc="Listing PNGs", unit="scenario"):
        paths.extend(sorted(seg_dir.glob("*.png")))
    return paths


MAX_VALID_CLASS = 28


def _unique_classes_in_batch(png_paths: list[str]) -> set[int]:
    """Return the set of unique R-channel values across a batch of PNGs.

    Values > MAX_VALID_CLASS are rendering artifacts and are excluded.
    """
    unique: set[int] = set()
    for p in png_paths:
        img = np.array(Image.open(p).convert("RGBA"))
        r = img[:, :, 0]
        unique.update(np.unique(r[r <= MAX_VALID_CLASS]).tolist())
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (contains train/ and test/); created on download")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not auto-download missing dataset parts")
    parser.set_defaults(download=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.download:
        from carlanomaly.download import ensure_parts
        ensure_parts(data_root, ["base"])

    print("discover_classes.py starting", flush=True)
    print(f"data_root = {data_root}", flush=True)
    print(f"data_root exists = {data_root.exists()}", flush=True)

    out_path = _HERE / "class_map.json"

    print("Step 1/3: Finding segmentation-front directories …", flush=True)
    seg_dirs = _find_seg_dirs(data_root)
    print(f"Total: {len(seg_dirs)} scenario directories\n", flush=True)

    print("Step 2/3: Listing PNG files …")
    all_paths = _collect_png_paths(seg_dirs)
    print(f"Total: {len(all_paths):,} images\n", flush=True)

    batch_size = 200
    batches = [
        [str(p) for p in all_paths[i : i + batch_size]]
        for i in range(0, len(all_paths), batch_size)
    ]

    global_unique: set[int] = set()
    n_workers = 16

    print(f"Step 3/3: Scanning pixel values ({n_workers} workers, {len(batches)} batches) …", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_unique_classes_in_batch, b): i for i, b in enumerate(batches)}
        pbar = tqdm(total=len(all_paths), desc="Scanning", unit="img")
        for fut in as_completed(futures):
            global_unique.update(fut.result())
            pbar.update(batch_size)
        pbar.close()

    sorted_ids = sorted(global_unique)
    print(f"\nFound {len(sorted_ids)} unique class IDs: {sorted_ids}")

    class_map = {cid: idx for idx, cid in enumerate(sorted_ids)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"Saved class map ({len(class_map)} classes) to {out_path}")


if __name__ == "__main__":
    main()

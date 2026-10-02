"""Evaluate the segmentation baseline on the CarlAnomaly test set (front camera).

For each pixel, the anomaly score is the **MaxLogit** score ``-max_c logits``
(Hendrycks et al., 2022); higher means more anomalous. Pixel AUROC/AUPR/FPR95
are pooled over every pixel in the test split (all frames, including anomaly-free
ones), not averaged per frame. Frame-level score is derived as max over all
pixels by the evaluator.

Usage:
    python baselines/rgb_segmentation/evaluate.py [--batch-size 16] [--gpu 0]
"""

from __future__ import annotations

import argparse
import json
import pprint
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carlanomaly.index import ScenarioIndex
from carlanomaly.datasets.rgb import RGBDataset
from carlanomaly.datasets.segmentation import SegmentationDataset
from carlanomaly.datasets import WithIdentifiers, carlanomaly_collate_fn
from carlanomaly.evaluator import (
    PixelEvaluator,
    SensorEvaluator,
    TimestepEvaluator,
    ScenarioEvaluator,
)

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = _HERE / "checkpoints"

TARGET_H, TARGET_W = 540, 960
ORIG_H, ORIG_W = 1080, 1920


class EvalDataset(Dataset):
    """Wraps RGB + segmentation for evaluation. Anomaly labels are loaded by
    the evaluators themselves."""

    def __init__(self, index: ScenarioIndex, class_map: dict[int, int]) -> None:
        self._index = index
        self._rgb = RGBDataset(index=index, direction="front")
        self._seg = SegmentationDataset(index=index, direction="front")
        lut = torch.zeros(256, dtype=torch.long)
        for orig, contiguous in class_map.items():
            lut[int(orig)] = contiguous
        self._class_lut = lut

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

        seg = self._seg[idx]["semantic"].squeeze(0)  # (H, W) original res
        seg = F.interpolate(
            seg.float().unsqueeze(0).unsqueeze(0),
            size=(TARGET_H, TARGET_W), mode="nearest",
        ).squeeze(0).squeeze(0).long()
        seg_labels = self._class_lut[seg]

        return {"rgb": rgb, "seg_labels": seg_labels}


_REDUCE = {
    "max": lambda x: x.amax(dim=(-1, -2)),
    "mean": lambda x: x.mean(dim=(-1, -2)),
    "min": lambda x: x.amin(dim=(-1, -2)),
}


def build_model(num_classes: int) -> torch.nn.Module:
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 0) -> torch.Tensor:
    mask = target != ignore_index
    idx = num_classes * target[mask] + pred[mask]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "deeplabv3_r50_msp.pt"))
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
                          download=args.download, parts=["base"])
    print(f"Evaluating on {len(index)} frames", flush=True)

    eval_ds = WithIdentifiers(EvalDataset(index, class_map))
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=carlanomaly_collate_fn,
        prefetch_factor=6,
        persistent_workers=True,
    )

    pixel_eval = PixelEvaluator(sensor="front")
    sensor_eval = SensorEvaluator(sensor="front")
    obs_eval = TimestepEvaluator()
    scenario_eval = ScenarioEvaluator()
    reduce = _REDUCE[args.reduction]
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    frame_scores_gpu: list[torch.Tensor] = []  # deferred to one D2H after the loop
    frame_sids: list[str] = []
    frame_fids: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            images = batch["data"]["rgb"].to(device, non_blocking=True)
            scenario_ids = batch["scenario_id"]
            timestep_ids = batch["timesteps"][:, 0].tolist()

            logits = model(images)["out"]  # (B, C, H_small, W_small)

            pred_classes = logits.argmax(dim=1)  # stays on GPU
            seg_labels = batch["data"]["seg_labels"].to(device, non_blocking=True)
            conf += _confusion_matrix(
                pred_classes.reshape(-1), seg_labels.reshape(-1), num_classes,
            )

            # MaxLogit anomaly score: negative max-over-classes logit
            # (higher = more anomalous).
            pixel_scores_small = -logits.amax(dim=1)  # (B, H_small, W_small)

            pixel_scores = F.interpolate(
                pixel_scores_small.unsqueeze(1),
                size=(ORIG_H, ORIG_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            pixel_eval.update(pixel_scores, scenario_ids, timestep_ids)

            frame_scores_gpu.append(reduce(pixel_scores))  # stays on GPU
            frame_sids.extend(scenario_ids)
            frame_fids.extend(timestep_ids)

    frame_scores = torch.cat(frame_scores_gpu).cpu().tolist()  # one D2H for all batches
    sensor_eval.update(frame_scores, frame_sids, frame_fids)
    obs_eval.update(frame_scores, frame_sids, frame_fids)

    for sid, score in sensor_eval.max_per_scenario().items():
        scenario_eval.update(score, sid)

    results = {
        "pixel": pixel_eval.compute(),
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
    print("Segmentation (MaxLogit) Baseline Results")
    print("=" * 60)
    pprint.pprint(results, width=80)

    out_path = _HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    out_dir = _HERE
    sensor_eval.to_dataframe().to_feather(out_dir / "scores_sensor.feather")
    obs_eval.to_dataframe().to_feather(out_dir / "scores_observation.feather")
    scenario_eval.to_dataframe().to_feather(out_dir / "scores_scenario.feather")
    print(f"Score DataFrames saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Multi-modal max-fusion baseline on CarlAnomaly (observation + scenario level).

Four per-frame detectors, each producing a score in ``[0, 1]``:

- **seg**     — DeepLabV3 MaxLogit (``-max logit``), max over the frame's pixels.
- **point**   — PointNet MaxLogit, max over the frame's points.
- **collision** — ``1.0`` if any collision that frame, else ``0.0``.
- **weather** — ``1.0`` if the GT weather changed more than ``--weather-threshold``
  vs the previous frame, else ``0.0``.

The two learned MaxLogit detectors are uncalibrated and live on very different
scales, so before fusion each is min-max normalized into ``[0, 1]`` using
percentile bounds (``p1``/``p99``) measured on the **held-out validation split
the models were trained against** (the same ``--val-fraction``/seed used by
train.py, normal-only) — scenarios the seg/point networks never saw, and never
the test set.  Calibrating on training-seen scenarios would bias the bounds low.
Collision and weather are already binary.

The observation-level (per-frame) multi-modal score is the ``max`` over the four
normalized detectors.

The scenario-level score does **not** simply take the max-per-scenario of that
per-frame fusion: doing so lets the noisy, uncalibrated learned detectors
(seg/point) saturate normal scenarios to ``1.0`` over ~300 frames, which swamps
the reliable binary detectors.  Instead each modality is first reduced to one
score per scenario with a reducer matched to its temporal profile — a robust high
**quantile** for the dense/noisy learned detectors (``--scenario-quantile``), and
``max`` for the sparse single-frame binary detectors (collision/weather, where a
real anomaly may last a single frame, e.g. an instantaneous weather change) —
and only then fused with ``max``.  Each component detector is also evaluated
individually for comparison.

Both the validation and test seg/point scores are computed here from the same
``detectors.py`` pipeline (using the rgb_segmentation/lidar_segmentation checkpoints), so the two sides
are guaranteed to be on the same scale — a prerequisite for the normalization to
be valid.

Usage:
    python baselines/multimodal/evaluate.py [--gpu 0] [--weather-threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import pprint
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carlanomaly.index import ScenarioIndex
from carlanomaly.evaluator import TimestepEvaluator, ScenarioEvaluator
from baselines.multimodal.detectors import (
    FrameKey,
    run_seg_frame_scores,
    run_point_frame_scores,
    collision_frame_scores,
    weather_frame_scores,
)

DEFAULT_DATA_ROOT = Path("./data")
_HERE = Path(__file__).resolve().parent
SEG_CKPT = _HERE.parent / "rgb_segmentation" / "checkpoints" / "deeplabv3_r50_msp.pt"
POINTNET_CKPT = _HERE.parent / "lidar_segmentation" / "checkpoints" / "pointnet_seg.pt"
OUT_DIR = _HERE

DETECTORS = ["seg", "point", "collision", "weather", "fused"]


def _build_val_index(data_root: str, val_fraction: float, seed: int, stride: int,
                     download: bool) -> ScenarioIndex:
    """Reconstruct the model's *actual* held-out validation split.

    The seg/point checkpoints were trained with ``_split_by_scenario(index,
    val_fraction, seed)`` (train.py), which holds out ``val_fraction`` of the
    train scenarios. We replicate that exact selection here so the normalization
    bounds are measured on normal scenarios the models **never trained on** —
    otherwise (e.g. a random subset of all of train) the calibration leaks
    training scenarios, the models are overconfident on them, and the resulting
    bounds are biased low.

    Scenario membership depends only on the record count and ``seed`` (not on
    ``clip_len``/``stride``), so it matches training regardless of the ``stride``
    we use here to frame-subsample for scoring cost."""
    index = ScenarioIndex(data_root, split="train", clip_len=1, stride=stride,
                          download=download, parts=["base", "lidar"])
    recs = list(index.records)
    n_val = max(1, int(len(recs) * val_fraction))
    val_rec_idxs = sorted(random.Random(seed).sample(range(len(recs)), n_val))
    index._records = [recs[i] for i in val_rec_idxs]
    index._index = index._build_index()
    return index


def _norm_bounds(scores: Dict[FrameKey, float], lo_pct: float, hi_pct: float) -> Tuple[float, float]:
    vals = np.fromiter(scores.values(), dtype=np.float64)
    return float(np.percentile(vals, lo_pct)), float(np.percentile(vals, hi_pct))


def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


Reducer = Callable[[np.ndarray], float]


def _scenario_scores(
    frame_scores: Dict[FrameKey, float],
    sids: List[str],
    fids: List[int],
    reduce: Reducer,
) -> Dict[str, float]:
    """Reduce a detector's per-frame scores to one score per scenario.

    ``reduce`` is chosen per modality: a robust high quantile for the noisy
    learned detectors (so a few saturated frames don't pin the scenario at the
    clip ceiling), ``max`` for the sparse binary detectors (so a single-frame
    event still counts)."""
    by_scen: Dict[str, List[float]] = defaultdict(list)
    for s, f in zip(sids, fids):
        by_scen[s].append(frame_scores[(s, int(f))])
    return {sid: float(reduce(np.asarray(v, dtype=np.float64))) for sid, v in by_scen.items()}


def _eval_detector(
    frame_scores: Dict[FrameKey, float],
    sids: List[str],
    fids: List[int],
    scenario_scores: Dict[str, float],
) -> Dict:
    """Observation metrics from per-frame scores + scenario metrics from the
    pre-aggregated per-scenario scores."""
    scores = [frame_scores[(s, f)] for s, f in zip(sids, fids)]
    obs = TimestepEvaluator()
    obs.update(scores, sids, fids)
    scen = ScenarioEvaluator()
    for sid, score in scenario_scores.items():
        scen.update(score, sid)
    return {"obs": obs, "scen": scen}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seg-batch-size", type=int, default=16)
    parser.add_argument("--point-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--weather-threshold", type=float, default=0.5)
    parser.add_argument(
        "--scenario-quantile", type=float, default=0.95,
        help="per-scenario quantile for the learned (seg/point) detectors before "
             "fusion; binary detectors (collision/weather) always use max.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.05,
        help="held-out fraction of train scenarios used for normalization; must "
             "match the training-time --val-fraction so the same scenarios are used.",
    )
    parser.add_argument("--val-stride", type=int, default=10)
    parser.add_argument(
        "--val-seed", type=int, default=42,
        help="must match the training-time split seed so the held-out scenarios "
             "are exactly those the models did not train on.",
    )
    parser.add_argument("--norm-lo-pct", type=float, default=1.0)
    parser.add_argument("--norm-hi-pct", type=float, default=99.0)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (contains train/ and test/); created on download")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="do not auto-download missing dataset parts")
    parser.set_defaults(download=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    for ckpt_path in (SEG_CKPT, POINTNET_CKPT):
        if not ckpt_path.exists():
            raise SystemExit(
                f"Checkpoint not found: {ckpt_path}\n"
                f"The multimodal baseline reuses the RGB- and LiDAR-segmentation "
                f"checkpoints. Download them with ./download_checkpoints.sh, or "
                f"train them with their respective train.py scripts."
            )
    seg_ckpt = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    point_ckpt = torch.load(POINTNET_CKPT, map_location="cpu", weights_only=False)

    # --- Validation-set normalization bounds for the two learned detectors ---
    print("Building held-out validation split (train, normal-only) for normalization …", flush=True)
    val_index = _build_val_index(args.data_root, args.val_fraction, args.val_seed,
                                 args.val_stride, args.download)
    print(f"  val scenarios: {len(val_index.records)} (held out from training), "
          f"frames: {len(val_index)}", flush=True)

    print("Scoring validation frames (seg) …", flush=True)
    val_seg = run_seg_frame_scores(
        val_index, seg_ckpt, device, args.seg_batch_size, args.num_workers, desc="val-seg",
    )
    print("Scoring validation frames (point) …", flush=True)
    val_point = run_point_frame_scores(
        val_index, point_ckpt, device, args.point_batch_size, args.num_workers, desc="val-point",
    )

    seg_lo, seg_hi = _norm_bounds(val_seg, args.norm_lo_pct, args.norm_hi_pct)
    point_lo, point_hi = _norm_bounds(val_point, args.norm_lo_pct, args.norm_hi_pct)
    print(f"  seg   norm bounds: lo={seg_lo:.4f} hi={seg_hi:.4f}", flush=True)
    print(f"  point norm bounds: lo={point_lo:.4f} hi={point_hi:.4f}", flush=True)

    # --- Test scores: computed with the same pipeline as validation ---
    print("Building test index …", flush=True)
    test_index = ScenarioIndex(args.data_root, split="test", clip_len=1, stride=1,
                               download=args.download, parts=["base", "lidar"])
    print(f"  test frames: {len(test_index)}", flush=True)

    print("Scoring test frames (seg) …", flush=True)
    seg_test = run_seg_frame_scores(
        test_index, seg_ckpt, device, args.seg_batch_size, args.num_workers, desc="test-seg",
    )
    print("Scoring test frames (point) …", flush=True)
    point_test = run_point_frame_scores(
        test_index, point_ckpt, device, args.point_batch_size, args.num_workers, desc="test-point",
    )

    print("Computing collision/weather scores …", flush=True)
    collision_test = collision_frame_scores(test_index)
    weather_test = weather_frame_scores(test_index, args.weather_threshold)

    # Canonical frame set = the seg feather (same frames the other baselines used).
    keys = list(seg_test.keys())
    sids = [k[0] for k in keys]
    fids = [k[1] for k in keys]

    seg_n: Dict[FrameKey, float] = {}
    point_n: Dict[FrameKey, float] = {}
    fused: Dict[FrameKey, float] = {}
    for k in keys:
        sn = _normalize(seg_test[k], seg_lo, seg_hi)
        pn = _normalize(point_test[k], point_lo, point_hi)
        c = collision_test[k]
        w = weather_test[k]
        seg_n[k] = sn
        point_n[k] = pn
        fused[k] = max(sn, pn, c, w)

    print(f"Fused {len(keys)} frames "
          f"(collision-active: {int(sum(collision_test[k] for k in keys))}, "
          f"weather-active: {int(sum(weather_test[k] for k in keys))})", flush=True)

    # --- Per-frame (observation) scores: max-fusion is fine frame-by-frame ---
    component_scores = {
        "seg": seg_n,
        "point": point_n,
        "collision": collision_test,
        "weather": weather_test,
        "fused": fused,
    }

    # --- Scenario aggregation: reduce each modality per scenario, then fuse ---
    # Learned detectors use a robust high quantile; binary detectors use max.
    def quantile(a: np.ndarray) -> float:
        return float(np.quantile(a, args.scenario_quantile))

    def maximum(a: np.ndarray) -> float:
        return float(np.max(a))

    scen_seg = _scenario_scores(seg_n, sids, fids, quantile)
    scen_point = _scenario_scores(point_n, sids, fids, quantile)
    scen_coll = _scenario_scores(collision_test, sids, fids, maximum)
    scen_weather = _scenario_scores(weather_test, sids, fids, maximum)
    scen_fused = {
        sid: max(scen_seg[sid], scen_point[sid], scen_coll[sid], scen_weather[sid])
        for sid in scen_seg
    }
    scenario_scores = {
        "seg": scen_seg,
        "point": scen_point,
        "collision": scen_coll,
        "weather": scen_weather,
        "fused": scen_fused,
    }

    # --- Evaluate each detector + the fused score ---
    evals = {
        name: _eval_detector(component_scores[name], sids, fids, scenario_scores[name])
        for name in DETECTORS
    }

    results = {
        name: {
            "observation": ev["obs"].compute(),
            "scenario": ev["scen"].compute(),
        }
        for name, ev in evals.items()
    }
    results["_config"] = {
        "weather_threshold": args.weather_threshold,
        "seg_norm": [seg_lo, seg_hi],
        "point_norm": [point_lo, point_hi],
        "val_fraction": args.val_fraction,
        "val_seed": args.val_seed,
        "val_scenarios": len(val_index.records),
        "val_frames": len(val_index),
        "scenario_quantile": args.scenario_quantile,
        "scenario_agg": "quantile(seg,point) + max(collision,weather), fused by max",
    }

    print("\n" + "=" * 64)
    print("Multi-modal Max-Fusion Baseline Results")
    print("=" * 64)
    print(f"{'detector':<12} {'obs AUROC':>12} {'scenario AUROC':>16}")
    print("-" * 44)
    for name in DETECTORS:
        o = results[name]["observation"]["auroc"]
        s = results[name]["scenario"]["auroc"]
        print(f"{name:<12} {o:>12.4f} {s:>16.4f}")
    print("=" * 64)
    pprint.pprint(results, width=100)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'results.json'}", flush=True)

    evals["fused"]["obs"].to_dataframe().to_feather(OUT_DIR / "scores_observation.feather")
    evals["fused"]["scen"].to_dataframe().to_feather(OUT_DIR / "scores_scenario.feather")
    print(f"Fused score DataFrames saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

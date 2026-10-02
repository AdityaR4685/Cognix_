# CarlAnomaly — Baselines

Baseline anomaly-detection methods for the **CarlAnomaly** benchmark (multimodal
anomaly detection for autonomous driving, built on CARLA simulator data).

> **Baseline code only.** This repository depends on the
> [`carlanomaly`](https://github.com/carlanomaly/devkit) data-loading /
> evaluation framework and the CarlAnomaly dataset, neither of which is included
> here.

## Reproduce

```bash
# 1. Install the dev kit + baseline deps.
pip install -r requirements.txt

# 2. Fetch the pretrained checkpoints into baselines/<method>/checkpoints/.
./download_checkpoints.sh

# 3. Evaluate. The dataset auto-downloads to ./data on first run.
python baselines/rgb_segmentation/evaluate.py --gpu 0
```

Run every script from the repository root.

### Data

Each script downloads the dataset parts it needs (via the dev kit's
`ScenarioIndex(download=True)`) into the dataset root on first run, and skips
parts already present on later runs.

- `--data-root PATH` — where the dataset lives / is downloaded to (default
  `./data`).
- `--no-download` — assume the data is already present under `--data-root` and
  skip the download (e.g. on a prepared host or shared filesystem).

### Checkpoints

`download_checkpoints.sh` fetches the three published checkpoints
(`rgb_segmentation`, `lidar_segmentation`, `frame_prediction`) into
`baselines/<method>/checkpoints/`, where `evaluate.py` looks for them. The
`weather_prediction` baseline has no published checkpoint — train it locally.
Override the host with the `CARLANOMALY_BASE_URL` environment variable.

### Training

```bash
python baselines/rgb_segmentation/train.py --gpu 0 --batch-size 16
```

The `rgb_segmentation` and `lidar_segmentation` baselines need a class map
first: `python baselines/rgb_segmentation/discover_classes.py` writes
`class_map.json` (the LiDAR baseline reuses the RGB one). Frame prediction
supports multi-GPU via `torchrun`.

### GPU selection

Scripts take `--gpu <index>`. To pin a specific device by UUID, set
`CUDA_VISIBLE_DEVICES=<UUID>` (which reindexes visible devices, so `--gpu 0` is
then correct).

## Baselines

| Dir | Method | Anomaly score |
|-----|--------|---------------|
| `rgb_segmentation/` | DeepLabV3-ResNet50 semantic segmentation | per-pixel **MaxLogit** (`-max_c logits`) |
| `lidar_segmentation/` | PointNet per-point segmentation (LiDAR) | per-point **MaxLogit** |
| `frame_prediction/` | U-Net future-frame prediction | per-pixel MSE (`(pred − target)²`) |
| `multimodal/` | late fusion of seg / point / collision / weather detectors | per-scenario quantile-then-max fusion |
| `weather_prediction/` | weather-prediction baseline | — |

Each baseline provides `train.py` and/or `evaluate.py` plus a `model.py`.

## Evaluation tiers

Scores are evaluated at four tiers — **sample** (pixel/point), **sensor**,
**observation**, and **scenario** — via the `carlanomaly.evaluator` module.

## Artifacts (not tracked)

Model weights (`*.pt`), score tables (`*.feather`), `results.json`,
`class_map.json`, and TensorBoard `runs/` are git-ignored; regenerate them with
the train/evaluate scripts.

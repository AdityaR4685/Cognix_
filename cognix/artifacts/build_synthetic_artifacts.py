"""
Build Deterministic Synthetic Artifacts for COGNIX.

Generates:
  artifacts/synthetic/epistemic_gat.pt
  artifacts/synthetic/standard_gat.pt
  artifacts/synthetic/conformal_cal_scores.npy
  artifacts/synthetic/metadata.json

Strict split separation:
  TRAIN: seed 42 (n=125)
  CALIBRATION: seed 1042 (n=125)
  TEST: Never touched for artifact fitting.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from cognix.adapters.carla.agents import (
    CameraAgent, DepthAgent, LiDARAgent,
    GNSSAgent, IMUAgent, SegAgent,
)
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.artifacts.persistence import save_synthetic_artifacts
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor
from cognix.config.schema import CognixConfig
from cognix.engine.decision_engine import DecisionEngine
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.graph.standard_gat import StandardGAT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cognix.artifacts.build")


def generate_synthetic_data(seed: int, n_per_anomaly: int = 25):
    """Generate separate dataset split using CarlAnomalyDataset."""
    ds = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=n_per_anomaly, seed=seed)
    X = []
    y = []
    for anomaly_type in ds.anomalies:
        frames = ds.generate_frames(anomaly_type)
        for frame in frames:
            inputs = {
                "Camera": frame.rgb,
                "Depth": frame.depth,
                "LiDAR": frame.lidar,
                "GNSS": frame.gnss,
                "IMU": frame.imu,
                "Seg": frame.segmentation,
            }
            # label: 0 for safe (target P(ACT)=1.0), 1 for hazard (target P(ACT)=0.0)
            target = 1.0 if frame.label == 0 else 0.0
            X.append(inputs)
            y.append(target)
    return X, np.array(y, dtype=np.float32)


def build_all_artifacts(target_dir: str | Path | None = None) -> Path:
    """Train and calibrate models, then save artifacts to target_dir."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "synthetic"
    else:
        target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building synthetic artifacts in %s ...", target_dir)

    # 1. Instantiate agents
    agents = [
        CameraAgent(),
        DepthAgent(),
        LiDARAgent(),
        GNSSAgent(),
        IMUAgent(),
        SegAgent(),
    ]

    # 2. Strict split generation
    logger.info("Generating TRAIN split (seed=42, n=25/anomaly)...")
    X_train, y_train = generate_synthetic_data(seed=42, n_per_anomaly=25)
    logger.info("Generating CALIBRATION split (seed=1042, n=25/anomaly)...")
    X_cal, y_cal = generate_synthetic_data(seed=1042, n_per_anomaly=25)

    # 3. Fit EpistemicGAT on TRAIN split only
    logger.info("Fitting EpistemicGAT (25 epochs, lr=1e-3, seed=42)...")
    torch.manual_seed(42)
    np.random.seed(42)
    epistemic_gat = EpistemicGAT(
        num_layers=2,
        input_dim=3,
        hidden_dim=8,
        output_dim=1,
        use_epistemic_prior=True,
    )
    epistemic_gat.fit(agents, X_train, y_train, epochs=25, lr=1e-3, seed=42)
    assert epistemic_gat.is_trained(), "EpistemicGAT training failed"

    # 4. Fit StandardGAT on TRAIN split only
    logger.info("Fitting StandardGAT (25 epochs, lr=1e-3, seed=42)...")
    torch.manual_seed(42)
    np.random.seed(42)
    standard_gat = StandardGAT(
        num_layers=2,
        input_dim=3,
        hidden_dim=8,
        output_dim=1,
    )
    standard_gat.fit(agents, X_train, y_train, epochs=25, lr=1e-3, seed=42)
    assert standard_gat.is_trained(), "StandardGAT training failed"

    # 5. Fit ConformalPredictor strictly on CAL split
    logger.info("Running CAL split through upstream pipeline to calibrate ConformalPredictor...")
    fuser = EpistemicWeightedFusion(eps=1e-8)
    cal_engine = DecisionEngine(
        config=CognixConfig(),
        graph=epistemic_gat,
        belief=fuser,
        calibrator=None,
        mode="production",
    )
    
    cal_probs = []
    for x_c in X_cal:
        res = cal_engine.decide(
            agents=agents,
            input_data=x_c,
            context={"split": "calibration"},
            agent_reliabilities={a.agent_id: 1.0 for a in agents},
        )
        p = float(np.clip(res.confidence, 1e-7, 1.0 - 1e-7))
        cal_probs.append([1.0 - p, p])
    
    cal_outputs = np.array(cal_probs)
    cal_labels = np.array(y_cal, dtype=int)

    conformal = ConformalPredictor()
    conformal.fit(cal_outputs, cal_labels)
    assert conformal.cal_scores is not None
    assert conformal.n_cal == len(X_cal)
    logger.info("ConformalPredictor calibrated on %d samples", conformal.n_cal)

    # 6. Save metadata and artifacts
    metadata = {
        "version": "0.2.0",
        "dataset": "CarlAnomalyDataset (Synthetic)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "graph": {
            "type": "EpistemicGAT",
            "input_dim": 3,
            "hidden_dim": 8,
            "output_dim": 1,
            "num_layers": 2,
            "epochs": 25,
            "learning_rate": 0.001,
            "train_seed": 42,
            "train_samples": len(X_train),
        },
        "standard_gat": {
            "type": "StandardGAT",
            "input_dim": 3,
            "hidden_dim": 8,
            "output_dim": 1,
            "num_layers": 2,
            "epochs": 25,
            "learning_rate": 0.001,
            "train_seed": 42,
            "train_samples": len(X_train),
        },
        "calibration": {
            "method": "split_conformal",
            "alpha": 0.05,
            "target_coverage": 0.95,
            "cal_seed": 1042,
            "n_calibration_samples": len(X_cal),
        },
        "agents": [a.agent_id for a in sorted(agents, key=lambda x: x.agent_id)],
    }

    save_synthetic_artifacts(
        directory=target_dir,
        epistemic_gat=epistemic_gat,
        conformal=conformal,
        metadata=metadata,
        standard_gat=standard_gat,
    )
    logger.info("Finished building and persisting all synthetic artifacts.")
    return target_dir


if __name__ == "__main__":
    build_all_artifacts()

"""
COGNIX Artifact Persistence Module.

Provides clean, safe serialization and deserialization for:
- EpistemicGAT (PyTorch state_dict)
- StandardGAT (PyTorch state_dict)
- ConformalPredictor (NumPy nonconformity scores)
- Metadata (JSON)

Ensures no unsafe arbitrary object pickling.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from cognix.calibration.conformal import ConformalPredictor
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.graph.standard_gat import StandardGAT

logger = logging.getLogger(__name__)


def save_epistemic_gat(
    gat: EpistemicGAT,
    filepath: str | Path,
) -> None:
    """Save an EpistemicGAT model weights safely via state_dict."""
    if not gat.is_trained():
        raise RuntimeError("Refusing to save an untrained EpistemicGAT.")
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        "state_dict": gat.state_dict(),
        "input_dim": gat.input_dim,
        "hidden_dim": gat.hidden_dim,
        "output_dim": gat.output_dim,
        "num_layers": gat.num_layers,
        "use_epistemic_prior": gat.use_epistemic_prior,
    }
    torch.save(state, path)
    logger.info("Saved EpistemicGAT weights to %s", path)


def load_epistemic_gat(
    filepath: str | Path,
    expected_input_dim: int = 3,
    expected_output_dim: int = 1,
) -> EpistemicGAT:
    """Load an EpistemicGAT model safely from state_dict."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"EpistemicGAT artifact not found at {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    input_dim = checkpoint.get("input_dim", expected_input_dim)
    hidden_dim = checkpoint.get("hidden_dim", 8)
    output_dim = checkpoint.get("output_dim", expected_output_dim)
    num_layers = checkpoint.get("num_layers", 2)
    use_prior = checkpoint.get("use_epistemic_prior", True)

    if input_dim != expected_input_dim:
        raise ValueError(
            f"Artifact input_dim={input_dim} does not match expected_input_dim={expected_input_dim}"
        )

    gat = EpistemicGAT(
        num_layers=num_layers,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_epistemic_prior=use_prior,
    )
    gat.load_state_dict(checkpoint["state_dict"])
    gat._trained = True
    gat.eval()
    logger.info("Loaded EpistemicGAT artifact from %s", path)
    return gat


def save_standard_gat(
    gat: StandardGAT,
    filepath: str | Path,
) -> None:
    """Save a StandardGAT model weights safely via state_dict."""
    if not gat.is_trained():
        raise RuntimeError("Refusing to save an untrained StandardGAT.")
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        "state_dict": gat._gat.state_dict(),
        "input_dim": gat._gat.input_dim,
        "hidden_dim": gat._gat.hidden_dim,
        "output_dim": gat._gat.output_dim,
        "num_layers": gat._gat.num_layers,
        "use_epistemic_prior": False,
    }
    torch.save(state, path)
    logger.info("Saved StandardGAT weights to %s", path)


def load_standard_gat(
    filepath: str | Path,
    expected_input_dim: int = 3,
    expected_output_dim: int = 1,
) -> StandardGAT:
    """Load a StandardGAT model safely from state_dict."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"StandardGAT artifact not found at {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    input_dim = checkpoint.get("input_dim", expected_input_dim)
    hidden_dim = checkpoint.get("hidden_dim", 8)
    output_dim = checkpoint.get("output_dim", expected_output_dim)
    num_layers = checkpoint.get("num_layers", 2)

    gat = StandardGAT(
        num_layers=num_layers,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    gat._gat.load_state_dict(checkpoint["state_dict"])
    gat._gat._trained = True
    gat._gat.eval()
    logger.info("Loaded StandardGAT artifact from %s", path)
    return gat


def save_conformal(
    calibrator: ConformalPredictor,
    filepath: str | Path,
) -> None:
    """Save ConformalPredictor nonconformity scores safely to a .npy file."""
    if calibrator.cal_scores is None:
        raise RuntimeError("Cannot save uncalibrated ConformalPredictor.")
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, calibrator.cal_scores)
    logger.info("Saved ConformalPredictor calibration scores to %s", path)


def load_conformal(
    filepath: str | Path,
) -> ConformalPredictor:
    """Load ConformalPredictor nonconformity scores from a .npy file."""
    path = Path(filepath)
    if not path.is_file():
        # Also check with .npy suffix if omitted
        if Path(str(filepath) + ".npy").is_file():
            path = Path(str(filepath) + ".npy")
        else:
            raise FileNotFoundError(f"Conformal artifact not found at {filepath}")

    scores = np.load(path)
    calibrator = ConformalPredictor()
    calibrator.cal_scores = scores
    calibrator.n_cal = len(scores)
    logger.info("Loaded ConformalPredictor with %d scores from %s", calibrator.n_cal, path)
    return calibrator


def save_synthetic_artifacts(
    directory: str | Path,
    epistemic_gat: EpistemicGAT,
    conformal: ConformalPredictor,
    metadata: dict[str, Any],
    standard_gat: StandardGAT | None = None,
) -> None:
    """Save all synthetic artifacts and metadata to a target directory."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    save_epistemic_gat(epistemic_gat, target_dir / "epistemic_gat.pt")
    save_conformal(conformal, target_dir / "conformal_cal_scores.npy")

    if standard_gat is not None:
        save_standard_gat(standard_gat, target_dir / "standard_gat.pt")

    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Successfully persisted synthetic artifacts to %s", target_dir)


def load_synthetic_artifacts(
    directory: str | Path,
    graph_type: str = "EpistemicGAT",
) -> Tuple[Any, ConformalPredictor, dict[str, Any]]:
    """
    Load graph and conformal artifacts from a directory.
    Validates metadata compatibility.
    """
    target_dir = Path(directory)
    meta_path = target_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found in {target_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate metadata
    expected_dim = metadata.get("graph", {}).get("input_dim", 3)
    expected_out = metadata.get("graph", {}).get("output_dim", 1)

    if graph_type == "EpistemicGAT":
        gat_path = target_dir / "epistemic_gat.pt"
        gat = load_epistemic_gat(gat_path, expected_input_dim=expected_dim, expected_output_dim=expected_out)
    elif graph_type == "StandardGAT":
        gat_path = target_dir / "standard_gat.pt"
        gat = load_standard_gat(gat_path, expected_input_dim=expected_dim, expected_output_dim=expected_out)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    conf_path = target_dir / "conformal_cal_scores.npy"
    conformal = load_conformal(conf_path)

    return gat, conformal, metadata

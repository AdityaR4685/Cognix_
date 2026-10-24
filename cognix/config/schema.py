"""
COGNIX configuration schema.

Supports configuration via:
- Python objects (CognixConfig(...))
- Dictionaries (CognixConfig.model_validate({...}))
- YAML files (CognixConfig.from_yaml(path))
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class UncertaintyConfig(BaseModel):
    method: Literal["mc_dropout", "deep_ensemble", "none"] = "mc_dropout"
    mc_dropout_passes: int = Field(default=50, ge=1, le=500, description="MC Dropout forward passes (T)")
    ensemble_size: int = Field(default=5, ge=2, le=50, description="Number of ensemble members")
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.9)
    decompose: bool = Field(default=True, description="Decompose into aleatoric/epistemic")
    ood_detection: bool = Field(default=True, description="Enable OOD detection")
    ood_method: Literal["mahalanobis", "energy", "max_softmax"] = "mahalanobis"


class FusionConfig(BaseModel):
    method: Literal[
        "uniform", "majority", "confidence", "reliability", "epistemic_weighted", "bayesian"
    ] = "epistemic_weighted"
    eps: float = Field(default=1e-8, description="Numerical stability epsilon")


class CalibrationConfig(BaseModel):
    method: Literal["conformal", "temperature", "none"] = "conformal"
    coverage_target: float = Field(default=0.95, ge=0.5, le=1.0)
    temperature: float = Field(default=1.0, ge=0.01, le=100.0)


class CommunicationConfig(BaseModel):
    method: Literal["all_to_all", "top_k", "information_gain"] = "top_k"
    top_k: int = Field(default=3, ge=1)
    bandwidth_limit: Optional[int] = Field(default=None, description="Max messages per round")


class GraphConfig(BaseModel):
    method: Literal["gcn", "gat", "epistemic_gat", "none"] = "epistemic_gat"
    num_layers: int = Field(default=3, ge=1, le=10)
    hidden_dim: int = Field(default=64, ge=8)
    num_heads: int = Field(default=4, ge=1)


class AttributionConfig(BaseModel):
    method: Literal["epistemic_shapley", "shap", "none"] = "epistemic_shapley"
    num_samples: int = Field(default=100, ge=10)


class DecisionConfig(BaseModel):
    abstention: bool = True
    escalation: bool = True
    high_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    high_uncertainty_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_escalation_steps: int = Field(default=3, ge=1)


class ReproducibilityConfig(BaseModel):
    seed: int = Field(default=42, description="Global random seed")
    deterministic: bool = True


class CognixConfig(BaseModel):
    """Top-level COGNIX configuration."""
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    attribution: AttributionConfig = Field(default_factory=AttributionConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CognixConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "CognixConfig":
        d = self.decision
        if d.low_confidence_threshold >= d.high_confidence_threshold:
            raise ValueError(
                f"low_confidence_threshold ({d.low_confidence_threshold}) must be "
                f"< high_confidence_threshold ({d.high_confidence_threshold})"
            )
        return self

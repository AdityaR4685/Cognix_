from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(frozen=True)
class PredictionResult:
    value: Any
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class UncertaintyResult:
    prediction: float
    epistemic: float
    aleatoric: float
    total: float
    
    def __post_init__(self):
        # Validate mathematically logical bounds
        if not (0.0 <= self.prediction <= 1.0):
            raise ValueError(f"Prediction {self.prediction} out of bounds [0, 1]")
        if self.epistemic < 0.0 or self.aleatoric < 0.0 or self.total < 0.0:
            raise ValueError("Uncertainty values must be non-negative")

@dataclass(frozen=True)
class GraphResult:
    node_outputs: Any
    attention: Any
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FusionResult:
    probability: float
    weights: dict
    metadata: dict[str, Any] = field(default_factory=dict)

# ⚡ Quickstart Guide

This guide walks you through the smallest working end-to-end decision pipeline using the published `cognix` v0.1.0 API.

---

## 1. Minimal Working Example

In this example, two sensor agents (a camera and a radar) monitor a potential road obstacle. The camera has high confidence and low epistemic uncertainty, while the radar is degraded (e.g. by multipath reflections) and has higher epistemic uncertainty.

Save the following code as `quickstart_demo.py`:

```python
"""
COGNIX v0.1.0 Minimal Working Quickstart.
Demonstrates multi-agent epistemic-weighted belief fusion and risk escalation.
"""

from cognix import (
    CognixConfig,
    DecisionEngine,
    DecisionOutcome,
    EpistemicWeightedFusion,
    RiskLevel,
)


class SensorAgent:
    """A minimal sensor agent adhering to the COGNIX agent interface."""

    def __init__(self, agent_id: str, pred: float, epistemic: float, aleatoric: float):
        self.agent_id = agent_id
        self.pred = pred
        self.epistemic = epistemic
        self.aleatoric = aleatoric

    def predict(self, observation):
        """Return model prediction probability or confidence dict."""
        return {"confidence": self.pred, "prob": self.pred}

    def estimate_uncertainty(self, observation):
        """Return decomposed epistemic and aleatoric uncertainty."""
        return {
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total": self.epistemic + self.aleatoric,
        }


# Step 1: Instantiate sensor agents
# Camera: confident and certain
agent_camera = SensorAgent("camera", pred=0.85, epistemic=0.08, aleatoric=0.05)
# Radar: degraded sensor with higher model ignorance (epistemic)
agent_radar = SensorAgent("radar", pred=0.78, epistemic=0.12, aleatoric=0.08)

# Step 2: Initialize configuration
config = CognixConfig()

# Step 3: Select Epistemic-Weighted Belief Fusion
fuser = EpistemicWeightedFusion()

# Step 4: Inject components into the DecisionEngine
engine = DecisionEngine(
    config=config,
    belief=fuser,
    graph="no_graph",    # Pure Python/NumPy execution; no PyTorch needed
    mode="production",   # Fault-tolerant execution mode
)

# Step 5: Execute the decision cycle
result = engine.decide(
    agents=[agent_camera, agent_radar],
    input_data=None,     # Arbitrary observation payload passed to agents
)

# Step 6: Inspect structured DecisionResult
print("Decision Outcome:     ", result.decision.value)
print("Fused Confidence:     ", round(result.confidence, 4))
print("Epistemic Uncertainty:", round(result.epistemic_uncertainty, 4))
print("Aleatoric Uncertainty:", round(result.aleatoric_uncertainty, 4))
print("Total Uncertainty:    ", round(result.total_uncertainty, 4))
print("Risk Level:           ", result.risk_level.value)
print("Agent Trust Weights:  ", {k: round(v, 4) for k, v in result.agent_trust_weights.items()})
print("Decision Summary:     ", result.summary())
```

---

## 2. Running the Example

Run the script using your Python 3.10-3.12 interpreter:

```bash
python quickstart_demo.py
```

### Deterministic Verified Output

```text
Decision Outcome:      ACT
Fused Confidence:      0.822
Epistemic Uncertainty: 0.096
Aleatoric Uncertainty: 0.062
Total Uncertainty:     0.158
Risk Level:            LOW
Agent Trust Weights:   {'camera': 0.6, 'radar': 0.4}
Decision Summary:      Decision: ACT (Conf: 0.82, Risk: LOW, Uncertainty: 0.16)
```

---

## 3. Step-by-Step Breakdown

### Step 1: Agent Interface
The `DecisionEngine` accepts any Python object that:
- Exposes an `agent_id` or `name` attribute (or is registered in `AgentRegistry`).
- Implements `predict(input_data)`, returning a float probability in `[0, 1]`, a dictionary with a `"confidence"` / `"prob"` key, or a `PredictionResult`.
- Optionally implements `estimate_uncertainty(input_data)`, returning a dictionary or dataclass with `"epistemic"` and `"aleatoric"` numeric fields.

### Step 2 & 3: Configuration & Belief Fusion
`CognixConfig` defines decision thresholds (`high_confidence_threshold=0.70`, `low_confidence_threshold=0.40`). `EpistemicWeightedFusion` dynamically calculates normalized weights:
$$w_{\text{camera}} \propto \frac{1}{\sigma_{e, \text{cam}} + \epsilon} = \frac{1}{0.08 + 10^{-6}} \approx 12.5$$
$$w_{\text{radar}} \propto \frac{1}{\sigma_{e, \text{rad}} + \epsilon} = \frac{1}{0.12 + 10^{-6}} \approx 8.33$$
Normalized:
$$w_{\text{camera}} = 0.60, \quad w_{\text{radar}} = 0.40$$
The camera receives 60% of the voting weight due to its significantly lower epistemic uncertainty.

### Step 4 & 5: Decision Execution
The `DecisionEngine` coordinates the pipeline:
1. Gathers predictions and uncertainties.
2. Applies epistemic weighting to fuse beliefs ($0.60 \times 0.85 + 0.40 \times 0.78 = 0.822$).
3. Assesses the risk level: since confidence ($0.822$) exceeds $0.70$ and epistemic uncertainty ($0.096$) is below $0.50$, the risk level is `RiskLevel.LOW`.
4. Outputs `DecisionOutcome.ACT`.

---

## 4. Adding Conformal Calibration

To add distribution-free finite-sample statistical guarantees, plug in a fitted `ConformalPredictor`:

```python
import numpy as np
from cognix import ConformalPredictor

# 1. Instantiate the Inductive Conformal Predictor
calibrator = ConformalPredictor()

# 2. Fit on held-out calibration split: (N, C) probabilities and (N,) true labels
rng = np.random.RandomState(42)
cal_probs = rng.uniform(0.6, 0.95, size=(100, 2))
cal_probs /= cal_probs.sum(axis=1, keepdims=True)
cal_labels = rng.choice([0, 1], size=100)

calibrator.fit(cal_outputs=cal_probs, cal_labels=cal_labels)

# 3. Inject calibrator into DecisionEngine
engine_calibrated = DecisionEngine(
    config=config,
    belief=fuser,
    calibrator=calibrator,
    graph="no_graph",
    mode="production",
)

result_cal = engine_calibrated.decide(agents=[agent_camera, agent_radar], input_data=None)

print("Calibrated Confidence:", result_cal.calibrated_confidence)
print("Conformal Prediction Set:", result_cal.calibration_metrics["prediction_set"])
print("Target Coverage:", result_cal.calibration_metrics["target_coverage"])
```

### Output
```text
Calibrated Confidence: 0.8219998934052481
Conformal Prediction Set: [1]
Target Coverage: 0.95
```

When the conformal prediction set contains multiple classes (e.g. `[0, 1]`), the engine detects high ambiguity and escalates the decision.

---

## Next Steps

- Explore framework terminology in [Concepts](concepts.md).
- Review data flow and internals in [Architecture](architecture.md).
- Look up all 17 public symbols in the [API Reference](api-reference.md).

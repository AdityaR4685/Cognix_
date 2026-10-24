# 💻 Runnable Examples

This page provides practical, runnable code recipes demonstrating common multi-agent workflows with COGNIX v0.1.0.

---

## Example 1: Multi-Sensor Autonomous Fusion Under Sensor Faults

**Prerequisites**: Base package (`cognix-uncertainty-framework`). Zero PyTorch required.

This scenario models an autonomous perception system with three agents: Camera, Radar, and Lidar. When the Camera lens experiences sudden occlusion, its epistemic uncertainty spikes. COGNIX automatically detects the shift and shifts trust to the uncorrupted sensors.

```python
"""Multi-Sensor Autonomous Fusion with Fault Injection."""

from cognix import (
    CognixConfig,
    DecisionEngine,
    DecisionOutcome,
    EpistemicWeightedFusion,
    RiskLevel,
)


class VehicleSensor:
    def __init__(self, name: str, default_prob: float, default_epistemic: float, default_aleatoric: float):
        self.name = name
        self.prob = default_prob
        self.epistemic = default_epistemic
        self.aleatoric = default_aleatoric

    def inject_occlusion(self):
        """Simulate lens smudge or blackout."""
        self.prob = 0.50          # Completely ambiguous
        self.epistemic = 0.85     # High model ignorance

    def predict(self, observation):
        return {"confidence": self.prob, "prob": self.prob}

    def estimate_uncertainty(self, observation):
        return {
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total": self.epistemic + self.aleatoric,
        }


# Step 1: Initialize three sensors
camera = VehicleSensor("camera", default_prob=0.92, default_epistemic=0.06, default_aleatoric=0.04)
radar = VehicleSensor("radar", default_prob=0.88, default_epistemic=0.10, default_aleatoric=0.08)
lidar = VehicleSensor("lidar", default_prob=0.90, default_epistemic=0.08, default_aleatoric=0.05)

engine = DecisionEngine(
    config=CognixConfig(),
    belief=EpistemicWeightedFusion(),
    graph="no_graph",
    mode="production",
)

# Cycle 1: Nominal operation
print("=== Cycle 1: Nominal Sensors ===")
res1 = engine.decide([camera, radar, lidar], input_data=None)
print("Decision:    ", res1.decision.value)
print("Confidence:  ", round(res1.confidence, 4))
print("Trust Weights:", {k: round(v, 3) for k, v in res1.agent_trust_weights.items()})

# Cycle 2: Camera fault injected
print("\n=== Cycle 2: Camera Occluded ===")
camera.inject_occlusion()
res2 = engine.decide([camera, radar, lidar], input_data=None)
print("Decision:    ", res2.decision.value)
print("Confidence:  ", round(res2.confidence, 4))
print("Trust Weights:", {k: round(v, 3) for k, v in res2.agent_trust_weights.items()})
```

### Verified Output
```text
=== Cycle 1: Nominal Sensors ===
Decision:     ACT
Confidence:   0.9034
Trust Weights: {'camera': 0.426, 'radar': 0.255, 'lidar': 0.319}

=== Cycle 2: Camera Occluded ===
Decision:     ACT
Confidence:   0.8717
Trust Weights: {'camera': 0.05, 'radar': 0.422, 'lidar': 0.528}
```
*Notice how Camera's trust weight collapsed from 42.6% down to 5.0% automatically!*

---

## Example 2: Conformal Prediction & Statistical Coverage Verification

**Prerequisites**: Base package.

Fit an Inductive Conformal Predictor on validation data and compute empirical coverage over a simulated test set.

```python
"""Evaluating Conformal Prediction Coverage."""

import numpy as np
from cognix import ConformalPredictor

# 1. Generate synthetic calibration and test sets (Binary Classification)
rng = np.random.RandomState(42)
N_cal = 500
N_test = 200

# True underlying probabilities
p_cal = rng.beta(2, 5, size=N_cal)
cal_outputs = np.column_stack([1 - p_cal, p_cal])
cal_labels = (rng.uniform(size=N_cal) < p_cal).astype(int)

p_test = rng.beta(2, 5, size=N_test)
test_outputs = np.column_stack([1 - p_test, p_test])
test_labels = (rng.uniform(size=N_test) < p_test).astype(int)

# 2. Fit ConformalPredictor
calibrator = ConformalPredictor()
calibrator.fit(cal_outputs=cal_outputs, cal_labels=cal_labels)

# 3. Predict with 90% target coverage (alpha = 0.10)
alpha = 0.10
prediction_sets = calibrator.predict(test_outputs, alpha=alpha)

# 4. Measure empirical coverage
hits = sum(1 for i, ps in enumerate(prediction_sets) if test_labels[i] in ps.prediction_set)
empirical_coverage = hits / N_test

print(f"Target Coverage:    {(1 - alpha) * 100:.1f}%")
print(f"Empirical Coverage: {empirical_coverage * 100:.1f}%")
print(f"Sample Prediction Set 0: classes {prediction_sets[0].prediction_set}")
```

### Verified Output
```text
Target Coverage:    90.0%
Empirical Coverage: 89.0%
Sample Prediction Set 0: classes [0]
```

> [!NOTE]
> The empirical coverage value (89.0%) is the deterministic output of this specific synthetic simulation run ($N_{\text{test}} = 200$, random seed 42) and serves as an illustration. Split conformal prediction guarantees marginal coverage $P(Y \in C(X)) \ge 1 - \alpha$ in expectation under data exchangeability; empirical coverage on any individual finite sample fluctuates within expected binomial sampling variance.

---

## Example 3: Dynamic Agent Management via `AgentRegistry`

**Prerequisites**: Base package.

Manage a pool of dynamic sensor agents in a thread-safe registry, filter out unhealthy sensors, and inject healthy ones into the `DecisionEngine`.

```python
"""Thread-Safe Agent Registry Management."""

from cognix import AgentRegistry, DecisionEngine, EpistemicWeightedFusion


class DistributedSensor:
    def __init__(self, sensor_id: str, reading: float, healthy: bool = True):
        self.sensor_id = sensor_id
        self.reading = reading
        self.healthy = healthy

    def metadata(self):
        return {"id": self.sensor_id, "agent_id": self.sensor_id}

    def predict(self, observation):
        return self.reading

    def estimate_uncertainty(self, observation):
        return {"epistemic": 0.1 if self.healthy else 0.9, "aleatoric": 0.05}


registry = AgentRegistry()

# Register three agents
s1 = DistributedSensor("node_alpha", 0.82, healthy=True)
s2 = DistributedSensor("node_beta", 0.79, healthy=True)
s3 = DistributedSensor("node_gamma", 0.20, healthy=False)  # Faulty sensor

with registry:
    registry.register(s1)
    registry.register(s2)
    registry.register(s3)

# Filter out unhealthy agents dynamically
active_agents = registry.healthy_agents()
print(f"Registered Agents: {len(registry.list_agents())}")
print(f"Healthy Agents:    {len(active_agents)}")

engine = DecisionEngine(belief=EpistemicWeightedFusion(), graph="no_graph")
result = engine.decide(agents=active_agents, input_data=None)
print("Decision with healthy agents:", result.decision.value)
print("Confidence:", round(result.confidence, 4))
```

### Verified Output
```text
Registered Agents: 3
Healthy Agents:    2
Decision with healthy agents: ACT
Confidence: 0.805
```

---

## Example 4: Analytical Uncertainty Decomposition

**Prerequisites**: Base package (pure NumPy).

Disentangle classification predictive entropy into epistemic and aleatoric components.

```python
"""Analytical Entropy Uncertainty Decomposition."""

import numpy as np
from cognix import UncertaintyDecomposition

# Shape: (T=10 MC samples, N=1 input, C=3 classes)
mc_samples = np.array([
    [[0.70, 0.20, 0.10]],
    [[0.68, 0.22, 0.10]],
    [[0.72, 0.18, 0.10]],
    [[0.69, 0.21, 0.10]],
    [[0.71, 0.19, 0.10]],
    [[0.70, 0.20, 0.10]],
    [[0.68, 0.21, 0.11]],
    [[0.72, 0.19, 0.09]],
    [[0.69, 0.20, 0.11]],
    [[0.71, 0.20, 0.09]],
])

estimate = UncertaintyDecomposition.decompose_classification(mc_samples)
print("Method:               ", estimate.method)
print("Epistemic Uncertainty:", round(estimate.epistemic, 6))
print("Aleatoric Uncertainty:", round(estimate.aleatoric, 6))
print("Total Uncertainty:    ", round(estimate.total, 6))
```

### Verified Output
```text
Method:                entropy_decomposition
Epistemic Uncertainty: 0.000644
Aleatoric Uncertainty: 0.801175
Total Uncertainty:     0.801819
```

---

## Example 5: Bayesian Belief Updating with Beta Distributions

**Prerequisites**: Base package.

```python
"""Bayesian Belief State Tracking."""

from cognix import BayesianBelief

# Initialize belief from prior observation: p=0.80 with effective sample size 5
belief = BayesianBelief.from_prediction(agent_id="radar_1", p=0.80, n_effective=5.0)

print(f"Prior Alpha: {belief.alpha:.1f}, Beta: {belief.beta_param:.1f}")
prior_low, prior_high = belief.credible_interval(alpha=0.05)
print(f"Prior 95% Credible Interval: [{prior_low:.3f}, {prior_high:.3f}]")

# Stream new observations (binary target: 1 = detected obstacle, 0 = clear)
for obs in [1, 1, 1, 0, 1]:
    belief.update(obs)

post_low, post_high = belief.credible_interval(alpha=0.05)
print(f"\nPosterior Mean: {belief.posterior_mean[1]:.3f}")
print(f"Posterior 95% Credible Interval: [{post_low:.3f}, {post_high:.3f}]")
```

### Verified Output
```text
Prior Alpha: 4.0, Beta: 1.0
Prior 95% Credible Interval: [0.473, 0.994]

Posterior Mean: 0.800
Posterior 95% Credible Interval: [0.518, 0.972]
```

---

## Example 6: Relational Graph Reasoning with `EpistemicGAT`

> [!NOTE]
> This example requires the optional graph extra:
> `pip install "cognix-uncertainty-framework[graph]==0.1.0"`

```python
"""Graph Attention Reasoning across Agent Topologies."""

import torch
from cognix import CognixConfig, DecisionEngine, EpistemicWeightedFusion
from cognix.graph.epistemic_gat import EpistemicGAT


# Define simple agents
class SimpleSensor:
    def __init__(self, agent_id: str, p: float, epi: float):
        self.agent_id = agent_id
        self.p = p
        self.epi = epi

    def predict(self, x):
        return self.p

    def estimate_uncertainty(self, x):
        return {"epistemic": self.epi, "aleatoric": 0.05, "total": self.epi + 0.05}


agents = [
    SimpleSensor("cam", 0.90, 0.05),
    SimpleSensor("rad", 0.75, 0.15),
    SimpleSensor("lid", 0.85, 0.08),
]

# Instantiate EpistemicGAT (3 input node features: [prob, epistemic, aleatoric])
gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)

engine = DecisionEngine(
    config=CognixConfig(),
    belief=EpistemicWeightedFusion(),
    graph=gat,
    mode="production",
)

result = engine.decide(agents=agents, input_data=None)
print("Graph-Refined Decision:", result.decision.value)
print("Fused Confidence:      ", round(result.confidence, 4))
```

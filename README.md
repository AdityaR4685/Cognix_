# 🧠 COGNIX

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)]()

> A domain-agnostic, uncertainty-aware multi-agent AI research framework.

COGNIX is a framework designed to evaluate decision-making in Multi-Agent AI systems under conditions of distributional shift and sensor degradation. It provides a modular architecture for quantifying uncertainty (Epistemic vs. Aleatoric), fusing multi-agent beliefs, and generating calibrated, risk-aware decisions.

## 🏗️ 1. Framework Architecture

COGNIX is not a monolithic script, but a decoupled, dependency-injected framework built around strict API contracts. The core `DecisionEngine` coordinates the pipeline without depending on concrete algorithmic implementations.

The framework is divided into pluggable interfaces:
- **`AgentInterface`**: Standardizes agent predictions and uncertainty estimations (`PredictionResult` and `UncertaintyResult`).
- **`GraphRefinement`**: Handles inter-agent communication and attention routing (e.g., `StandardGAT`, `EpistemicGAT`, `NoGraph`).
- **`BeliefFuser`**: Implements belief fusion strategies (e.g., `EpistemicWeightedFusion`, `AverageFusion`).
- **`Calibrator`**: Enforces post-hoc calibration via Inductive Conformal Prediction.
- **`AttributionMethod`**: Explains decision contributions (e.g., `EpistemicShapley`).
- **`RiskStrategy`**: Maps calibrated uncertainties to actionable escalation levels.
- **`TransportInterface`**: Minimal abstraction for distributed multi-agent deployment.

## 🔬 2. Research Mechanism

The central research mechanism evaluated by COGNIX flows as follows:

```text
    epistemic uncertainty
            ↓
    epistemic prior
            ↓
    GAT attention
            ↓
    collective multi-agent belief
            ↓
    conformal calibration
            ↓
    risk-aware decision
            ↓
    attribution / escalation
```

By explicitly isolating **epistemic uncertainty** (model ignorance or out-of-distribution shift), COGNIX provides a mechanism to reduce the prior influence of uncertain agents before the network graph routes attention. The specific sender prior evaluated in Experiment 2B showed no descriptive evidence of additional held-out benefit over StandardGAT.

## 📊 3. Experimental Evidence

COGNIX is evaluated under a rigorous benchmarking infrastructure (Phase B). In the tested synthetic conditions:

- **Epistemic vs Standard Attention**: Ablation studies provide evidence that injecting an epistemic prior materially changes the attention redistribution of the GAT relative to a standard learned attention mechanism.
- **Out-Of-Distribution (OOD) Shift**: Observed in the tested conditions, epistemic-weighted fusion reduced the Expected Calibration Error (ECE) compared to uniform weighting when targeted agents were subjected to severe covariate shift.
- **Fail-Loud Research Integrity**: The pipeline strictly enforces a fail-loud behavior in research mode (e.g., rejecting fallback values when uncertainty decomposition fails), guaranteeing mathematically sound provenance for all reported metrics.

*Note: The magnitude of these effects depends heavily on the training quality of the agents and the severity of the distributional shift.*

**Closed Experiment 2B: official held-out TEST.** All 15 frozen runs (NoGraph, StandardGAT, and EpistemicGAT; five paired seeds) were evaluated on 627 scenarios and 187,473 scored timesteps per run. StandardGAT improved BCE relative to NoGraph for every seed, but reduced corruption F1 under the frozen GRAPH_VAL thresholds for every seed. Graph communication's benefit was therefore metric-dependent. EpistemicGAT was approximately neutral to slightly worse than StandardGAT. The prespecified 627-scenario macro-AUROC was undefined because all 107 normal scenarios were single-class; none were dropped. Five-seed comparisons are descriptive, with no prespecified or performed inferential significance test.

`q_normal` represents P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task and must not be interpreted as a probability of physical driving safety. `p_corruption = 1 - q_normal`. These findings concern the evaluated graph sender prior and do not extend the separate synthetic fusion findings above to official TEST. See the [comprehensive research report](docs/research/COGNIX_FINAL_STATUS.md#6-experiment-2b-closed-held-out-graph-evaluation) for tables, figures, protocol details, and limitations. Experiment 2B is closed and immutable; its TEST evaluation must not be rerun or retuned.

## 🎯 4. Research Targets

The ongoing development of COGNIX targets the following scientific objectives:
- Demonstrating behavioral substitutability across multiple non-automotive domains (e.g., tabular classification, distributed sensor networks).
- Expanding the conformal calibration pipeline to map empirical coverage under severe, non-exchangeable distribution shifts.
- Benchmarking the latency and exact vs. approximate runtime of Epistemic Shapley attribution as the number of agents $N$ scales.
- Evaluating resilience against conflicting agents and communication dropouts via the distributed `TransportInterface`.

## ⚠️ 5. Limitations

COGNIX is an early-stage research framework. Users must be aware of the following:
- **No Physical Safety Guarantees**: COGNIX does not guarantee safety. Conformal prediction provides empirical coverage *only* under strict calibration and exchangeability assumptions, which do not hold universally under arbitrary OOD shift.
- **No Superiority Claims**: COGNIX does not universally outperform baselines. Its mechanisms are highly dependent on the quality of the underlying MC-Dropout or Deep Ensemble uncertainty estimations.
- **Latency**: COGNIX makes no claims of "real-time" performance. Online latency for MC-Dropout, GAT inference, and exact Shapley enumeration must be explicitly measured for your specific hardware and batch size constraints.
- **Approximate Mathematics**: Uncertainty decomposition relies on approximations (e.g., Bernoulli predictive-variance decomposition) that may degrade under certain model architectures.

## 🚀 6. Example Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/zaidshaikh987/Cognix.git
cd Cognix

# Install the framework
pip install -e .
```

### Dependency Injection Pipeline

```python
from cognix.engine.decision_engine import DecisionEngine
from cognix.config.schema import CognixConfig
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor

# Initialize explicit strategies
config = CognixConfig()
gat = EpistemicGAT(input_dim=3, hidden_dim=8, output_dim=1)
fuser = EpistemicWeightedFusion()
calibrator = ConformalPredictor()

# Inject into the DecisionEngine
engine = DecisionEngine(
    config=config,
    belief=fuser,
    graph=gat,
    calibrator=calibrator,
    mode="research" # Fails loud on mathematically invalid states
)

# Agents must implement the AgentInterface
agents = [radar_agent, camera_agent, lidar_agent]

# Evaluate the pipeline
result = engine.decide(agents, input_data)

print(f"Fused Prediction: {result.confidence}")
print(f"Total Uncertainty: {result.total_uncertainty}")
```

### Running Automated Benchmarks

COGNIX supports automated, multi-seed ablation benchmarking:

```bash
python -m cognix.evaluation.run \
    --config configs/benchmark/gat_ablation.yaml \
    --output-dir results/gat_ablation/
```

### CARLA Live Demonstration V2

COGNIX includes a real-time, closed-loop demonstration in the CARLA autonomous driving simulator (`cognix/evaluation/cognix_live_demo_v2.py`). It connects to a running CARLA simulator (Town10HD_Opt), deploys multi-agent epistemic belief fusion, runs inductive conformal prediction, and performs risk escalation and route recovery under interactive or automated sensor faults.

#### Launching the Live Demonstration

Ensure CARLA 0.9.15+ is running on the target machine:

```bash
# Launch the live demonstration runner
python -m cognix.evaluation.cognix_live_demo_v2 --host 127.0.0.1 --port 2000

# Automated scenario cycling and telemetry capture
python -m cognix.evaluation.cognix_live_demo_v2 --host 127.0.0.1 --port 2000 --cycle-scenarios --duration 120 --output reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/telemetry_route_efficiency_120s.json
```

#### Interactive Controls

| Key | Mode / Action | Description |
|---|---|---|
| `N` | Nominal Driving | Clear faults and navigate nominal route |
| `P` | Pedestrian Crossing | Spawn crossing pedestrian hazard ahead |
| `W` | Adverse Weather | Shift weather to heavy rain, fog, and low sun |
| `B` | Road Obstacle / Props | Spawn dynamic obstacle ahead |
| `0` | Fault: NONE | Restore nominal sensor feeds |
| `1` | Fault: Darkness | Simulate optical blackout / underexposure |
| `2` | Fault: Occlusion | Simulate 60% camera lens occlusion |
| `3` | Fault: Seg Corruption | Corrupt semantic segmentation feed |
| `4` | Fault: IMU Disturbance | Inject Gaussian accelerometer/gyroscope disturbance |
| `R` | Road Recovery | Stabilize and align ego vehicle to safe lane waypoint |
| `Q` / `ESC` | Quit | Safe shutdown and actor cleanup |

#### Verification Artifacts & Telemetry

Full evaluation reports, multi-sensor telemetry traces, and efficiency audits for the live demonstration are archived in:
`reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/`
See [`COGNIX_LIVE_DEMO_V2_ROUTE_EFFICIENCY_REPORT.md`](reports/carla_runtime_v1/evaluation_runs/live_agent_uncertainty_demo_001/COGNIX_LIVE_DEMO_V2_ROUTE_EFFICIENCY_REPORT.md) for the benchmarked route completion, speed profiles, and conformal set size dynamics.


## 🤝 Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details on how to get started.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

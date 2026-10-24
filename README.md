# 🧠 COGNIX

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/cognix-uncertainty-framework/0.1.0/)
[![Distribution Name](https://img.shields.io/badge/pypi%20package-cognix--uncertainty--framework-orange.svg)](https://pypi.org/project/cognix-uncertainty-framework/0.1.0/)
[![Release](https://img.shields.io/badge/release-v0.1.0-brightgreen.svg)](https://github.com/AdityaR4685/Cognix_/releases/tag/v0.1.0)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A domain-agnostic, uncertainty-aware multi-agent AI decision framework.**

COGNIX coordinates and evaluates decision-making across distributed multi-agent systems operating under **distributional shift**, **sensor degradation**, and **inter-agent conflict**. By explicitly disentangling **epistemic uncertainty** (model ignorance / lack of training data) from **aleatoric uncertainty** (observational noise), COGNIX uses epistemic uncertainty as a dynamic trust signal to govern belief fusion, inter-agent communication, calibration, risk escalation, and attribution.

---

## 🎯 What Problem Does COGNIX Address?

In distributed multi-agent applications (such as autonomous robotics, sensor networks, and multi-model LLM ensembles), agents under severe sensor degradation or out-of-distribution (OOD) shift often become **confidently incorrect**. Standard unweighted averaging or naive voting schemes allow corrupted agents to degrade collective decisions.

COGNIX resolves this by:
1. **Dynamic Epistemic Weighting**: Automatically down-weighting agents exhibiting high epistemic uncertainty:
   $$w_j = \frac{\text{reliability}_j}{\sigma_{e, j} + \epsilon}$$
2. **Distribution-Free Calibration**: Enforcing finite-sample statistical coverage guarantees via Inductive Conformal Prediction.
3. **Multi-Tier Risk Escalation**: Classifying decisions into actionable outcomes (`ACT`, `WAIT`, `REQUEST_INFORMATION`, `ABSTAIN`, `ESCALATE`).
4. **Decoupled Architecture**: Providing pure-Python/NumPy core primitives with zero mandatory PyTorch or simulator dependencies.

---

## 🚀 Installation

The official distribution package is published on PyPI as **`cognix-uncertainty-framework`** (imported in Python as `cognix`):

```bash
python -m pip install cognix-uncertainty-framework==0.1.0
```

> [!TIP]
> Always run `python -m pip` to ensure installation into the active Python interpreter. On Windows systems with multiple Python installations, use the Python Launcher to select Python 3.12:
> ```powershell
> py -3.12 -m venv .venv
> .\.venv\Scripts\Activate.ps1
> python -m pip install cognix-uncertainty-framework==0.1.0
> ```

### Verification Command

Verify the installation in one step:

```bash
python -c "import cognix; print(cognix.__version__); print(cognix.__file__); print(len(cognix.__all__))"
```
**Expected Output**:
```text
0.1.0
<path-to-env>/site-packages/cognix/__init__.py
17
```

### Optional Feature Extras

Core COGNIX requires only `numpy`, `scipy`, `pydantic`, and `pyyaml`. Optional extras enable deep learning and simulator integrations:

```bash
# Relational Graph Attention Networks (PyTorch + NetworkX)
pip install "cognix-uncertainty-framework[graph]==0.1.0"

# Explainability Attribution (SHAP + LIME)
pip install "cognix-uncertainty-framework[explainability]==0.1.0"

# Telemetry Dashboard (FastAPI + Uvicorn + WebSockets)
pip install "cognix-uncertainty-framework[dashboard]==0.1.0"

# All Generic Extras Combined
pip install "cognix-uncertainty-framework[all]==0.1.0"
```

---

## ⚡ Minimal Working Quickstart

The following minimal example runs out of the box using only the base package (zero PyTorch required):

```python
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
        return {"confidence": self.pred, "prob": self.pred}

    def estimate_uncertainty(self, observation):
        return {
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total": self.epistemic + self.aleatoric,
        }


# Camera: high confidence, low model uncertainty
camera = SensorAgent("camera", pred=0.85, epistemic=0.08, aleatoric=0.05)
# Radar: degraded sensor with higher model ignorance (epistemic)
radar = SensorAgent("radar", pred=0.78, epistemic=0.12, aleatoric=0.08)

# Initialize DecisionEngine with Epistemic-Weighted Fusion
engine = DecisionEngine(
    config=CognixConfig(),
    belief=EpistemicWeightedFusion(),
    graph="no_graph",
    mode="production",
)

# Run decision cycle
result = engine.decide(agents=[camera, radar], input_data=None)

print(result.summary())
print("Decision Outcome:   ", result.decision.value)
print("Fused Confidence:   ", round(result.confidence, 4))
print("Agent Trust Weights:", {k: round(v, 3) for k, v in result.agent_trust_weights.items()})
```

### Output
```text
Decision: ACT (Conf: 0.82, Risk: LOW, Uncertainty: 0.16)
Decision Outcome:    ACT
Fused Confidence:    0.822
Agent Trust Weights: {'camera': 0.6, 'radar': 0.4}
```

*Due to its lower epistemic uncertainty, the camera automatically receives 60% of the voting weight.*

---

## 📚 Public API Overview (17 Exported Symbols)

The top-level `cognix` module exports exactly 17 symbols via `cognix.__all__`:

| Exported Symbol | Type | Primary Role |
|---|---|---|
| [`__version__`](docs/api-reference.md#1-__version__) | `str` | Package version string (`"0.1.0"`). |
| [`DecisionEngine`](docs/api-reference.md#2-decisionengine) | Class | Primary public orchestrator coordinating the 13-stage decision pipeline. |
| [`DecisionResult`](docs/api-reference.md#3-decisionresult) | Dataclass | Structured decision output containing predictions, uncertainties, and audit steps. |
| [`RiskLevel`](docs/api-reference.md#4-risklevel) | Enum | Categorical risk levels: `LOW`, `MODERATE`, `HIGH`. |
| [`DecisionOutcome`](docs/api-reference.md#5-decisionoutcome) | Enum | Actionable outcomes: `ACT`, `WAIT`, `REQUEST_INFORMATION`, `ABSTAIN`, `ESCALATE`. |
| [`StandardAgent`](docs/api-reference.md#6-standardagent) | Class | Model wrapper implementing `AgentInterface` with injected uncertainty estimator. |
| [`AgentRegistry`](docs/api-reference.md#7-agentregistry) | Class | Thread-safe (`RLock`) lifecycle manager and health tracker for dynamic agents. |
| [`MonteCarloDropout`](docs/api-reference.md#8-montecarlodropout) | Class | Monte Carlo Dropout uncertainty estimator ($T$ stochastic forward passes). |
| [`DeepEnsemble`](docs/api-reference.md#9-deepensemble) | Class | Deep Ensemble epistemic uncertainty estimator across model instances. |
| [`UncertaintyDecomposition`](docs/api-reference.md#10-uncertaintydecomposition) | Class (static methods) | Pure NumPy entropy and variance predictive uncertainty decomposition. |
| [`BayesianBelief`](docs/api-reference.md#11-bayesianbelief) | Class | Beta-distribution and Dirichlet belief state tracking with credible intervals. |
| [`EpistemicWeightedFusion`](docs/api-reference.md#12-epistemicweightedfusion) | Class | Belief fusion weighting agents inversely proportional to epistemic uncertainty. |
| [`AverageFusion`](docs/api-reference.md#13-averagefusion) | Class | Baseline uniform unweighted average fusion ($w_j = 1/N$). |
| [`ConformalPredictor`](docs/api-reference.md#14-conformalpredictor) | Class | Split Inductive Conformal Prediction with finite-sample coverage guarantees. |
| [`TemperatureScaling`](docs/api-reference.md#15-temperaturescaling) | Class | Optimization-based post-hoc logit calibration via SciPy L-BFGS-B. |
| [`EscalationEngine`](docs/api-reference.md#16-escalationengine) | Class | Multi-tier decision gating based on confidence, uncertainty, and conformal sets. |
| [`CognixConfig`](docs/api-reference.md#17-cognixconfig) | Class | Pydantic v2 configuration schema with YAML loading, exporting, and validation. |

*See the complete [API Reference](docs/api-reference.md) for signatures, defaults, return types, and code snippets.*

---

## ⚙️ Supported Environments

- **Python**: `3.10`, `3.11`, `3.12` (verified in GitHub Actions CI).
- **Operating Systems**: Linux (Ubuntu 22.04+), macOS (12+), Windows (10/11).
- **Base Dependencies**: `numpy>=1.24`, `scipy>=1.10`, `pydantic>=2.0`, `pyyaml>=6.0`.

---

## ⚠️ Limitations & Research Scope

- **No Physical Safety Guarantee**: COGNIX provides mathematical risk and coverage metrics. Conformal coverage holds strictly under the assumption of data exchangeability; severe OOD shift may violate this assumption.
- **No Universal Superiority Claims**: In closed held-out benchmark evaluations (Experiment 2B: 15 frozen runs across 5 paired seeds, 627 scenarios), graph communication benefits were metric-dependent (StandardGAT improved BCE but reduced corruption F1; EpistemicGAT was neutral to slightly worse).
- **Computational Overhead**: Monte Carlo Dropout ($T$ forward passes) and exact Shapley attribution ($O(2^N)$ subsets) incur measurable computational latency. Measure for your target hardware.

*See [Limitations](docs/limitations.md) and [Research & Provenance](docs/research.md) for detailed empirical disclosures.*

---

## 📖 Full Documentation

Browsing the complete documentation website:
- [Documentation Index](docs/index.md)
- [Installation Guide](docs/installation.md)
- [Quickstart Guide](docs/quickstart.md)
- [Framework Concepts & Terminology](docs/concepts.md)
- [Architecture & Data Flow](docs/architecture.md)
- [API Reference (17 Symbols)](docs/api-reference.md)
- [Runnable Examples](docs/examples.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [Limitations & Caveats](docs/limitations.md)
- [Research & Methodology](docs/research.md)

---

## 📄 License & Links

- **License**: MIT License - see [LICENSE](LICENSE) for details.
- **PyPI Release**: [https://pypi.org/project/cognix-uncertainty-framework/0.1.0/](https://pypi.org/project/cognix-uncertainty-framework/0.1.0/)
- **GitHub Repository**: [https://github.com/AdityaR4685/Cognix_](https://github.com/AdityaR4685/Cognix_)
- **Issue Tracker**: [https://github.com/AdityaR4685/Cognix_/issues](https://github.com/AdityaR4685/Cognix_/issues)
- **Release Tag**: [v0.1.0](https://github.com/AdityaR4685/Cognix_/releases/tag/v0.1.0) (commit `8d0432ae01265e9cec855c127c9bb00d09dea9f6`)

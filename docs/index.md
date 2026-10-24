# 🧠 COGNIX Documentation (v0.1.0)

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/cognix-uncertainty-framework/0.1.0/)
[![Distribution Name](https://img.shields.io/badge/pypi%20package-cognix--uncertainty--framework-orange.svg)](https://pypi.org/project/cognix-uncertainty-framework/0.1.0/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **COGNIX** is a domain-agnostic, uncertainty-aware multi-agent AI decision framework.

COGNIX provides a decoupled, modular architecture for evaluating and coordinating decision-making in multi-agent systems operating under **distributional shift**, **sensor degradation**, and **inter-agent conflict**. It explicitly isolates **epistemic uncertainty** (model ignorance / lack of training data) from **aleatoric uncertainty** (inherent data noise) and uses epistemic uncertainty as a dynamic trust signal to govern belief fusion, inter-agent routing, calibration, risk escalation, and attribution.

---

## What Problem Does COGNIX Solve?

In distributed multi-agent systems—such as autonomous robotics, sensor networks, or multi-model LLM ensembles—traditional fusion algorithms often compute unweighted averages or rely solely on raw output confidence scores. Under out-of-distribution (OOD) shift or hardware sensor faults, agents frequently become **confidently wrong**, dragging down collective decisions.

COGNIX addresses this failure mode by:

1. **Separating Uncertainty Components**: Disentangling model ignorance ($\sigma_e$) from observational noise ($\sigma_a$).
2. **Epistemic-Weighted Trust**: Dynamically down-weighting uncertain or degraded agents during belief fusion:
   $$w_j = \frac{\text{reliability}_j}{\sigma_{e, j} + \epsilon}$$
3. **Distribution-Free Calibration**: Enforcing finite-sample statistical coverage guarantees via Split Conformal Prediction.
4. **Graduated Risk Escalation**: Mapping fused confidence and epistemic uncertainty into explicit operational outcomes (`ACT`, `WAIT`, `REQUEST_INFORMATION`, `ABSTAIN`, `ESCALATE`).
5. **Decoupled Architecture**: Providing pure-Python/NumPy core primitives with zero mandatory PyTorch or simulator dependencies.

---

## Release v0.1.0 Overview

| Attribute | Specification |
|---|---|
| **PyPI Distribution** | [`cognix-uncertainty-framework`](https://pypi.org/project/cognix-uncertainty-framework/0.1.0/) |
| **Python Import** | `import cognix` |
| **Release Version** | `0.1.0` |
| **Verified Public Exports** | `17` public symbols in `cognix.__all__` |
| **Supported Python Versions** | `3.10`, `3.11`, `3.12` |
| **Core Dependencies** | `numpy>=1.24`, `scipy>=1.10`, `pydantic>=2.0`, `pyyaml>=6.0` |
| **License** | MIT License |

---

## Implemented vs. Research Features

COGNIX clearly distinguishes between fully implemented general-purpose software components and domain-specific or experimental research assets:

### Core Implemented Capabilities (v0.1.0)
- **`DecisionEngine`**: Public orchestrator wired with dependency-injected strategies and dual-mode execution (`production` fault tolerance vs. `research` fail-loud assertions).
- **`EpistemicWeightedFusion` & `AverageFusion`**: Analytical fusion strategies operating on arbitrary agent dictionaries.
- **`ConformalPredictor`**: Split Inductive Conformal Prediction with nonconformity quantile thresholding, coverage validation, and `.save()` / `.load()` persistence.
- **`TemperatureScaling`**: Optimization-based post-hoc logit calibration using SciPy L-BFGS-B.
- **`UncertaintyDecomposition`**: Closed-form predictive entropy decomposition (classification) and variance decomposition (regression).
- **`BayesianBelief`**: Analytical Beta-distribution and Dirichlet belief state tracking with credible intervals.
- **`EscalationEngine`**: Multi-tier decision gating based on point confidence, epistemic uncertainty thresholds, and conformal prediction set sizes.
- **`AgentRegistry`**: Thread-safe (`RLock`) lifecycle management and health monitoring for active agents.
- **`CognixConfig`**: Pydantic v2 configuration schema with YAML loading, exporting, and threshold validations.

### Optional Feature Extras
- **Graph Attention Networks (`[graph]`)**: PyTorch-based `EpistemicGAT` and `StandardGAT` inter-agent relational message passing.
- **Explainability (`[explainability]`)**: Epistemic Shapley attribution tracking agent marginal contributions to uncertainty reduction.
- **Vision & Dashboard (`[vision]`, `[dashboard]`)**: Computer vision utilities and FastAPI/WebSocket telemetry dashboard.

### Research Scope & Empirical Honesty
- **No Physical Safety Guarantee**: Conformal prediction guarantees finite-sample coverage *only* under the exchangeability assumption. Arbitrary, non-exchangeable covariate shift requires conservative thresholding.
- **No Universal Superiority**: As documented in the frozen [closed Experiment 2B evaluation](research.md), graph communication benefits are metric-dependent. COGNIX provides the measurement infrastructure; it does not claim to outperform all baselines in every scenario.

---

## Documentation Roadmap

```
├── Getting Started
│   ├── Installation (docs/installation.md)
│   └── Quickstart (docs/quickstart.md)
├── Architecture & Concepts
│   ├── Framework Concepts (docs/concepts.md)
│   └── Pipeline Architecture (docs/architecture.md)
├── Reference & Guides
│   ├── API Reference (17 Symbols) (docs/api-reference.md)
│   ├── Runnable Examples (docs/examples.md)
│   ├── Troubleshooting (docs/troubleshooting.md)
│   └── Limitations & Caveats (docs/limitations.md)
└── Research & Provenance
    └── Research Evidence & Benchmarks (docs/research.md)
```

To get started right away, proceed to the [Installation Guide](installation.md) or dive directly into the [Quickstart](quickstart.md).

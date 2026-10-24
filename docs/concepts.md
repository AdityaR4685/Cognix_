# 💡 Core Concepts & Terminology

This document defines the mathematical, algorithmic, and architectural concepts underpinning COGNIX v0.1.0.

---

## 1. Uncertainty Decomposition: Epistemic vs. Aleatoric

A cornerstone of COGNIX is the explicit disentanglement of two distinct categories of predictive uncertainty:

```
Total Predictive Uncertainty
├── Aleatoric Uncertainty (Data Noise / Environmental Stochasticity)
│   └── Irreducible with additional training data
└── Epistemic Uncertainty (Model Ignorance / Out-of-Distribution Shift)
    └── Reducible with additional training data or targeted exploration
```

### Epistemic Uncertainty ($\sigma_e$)
- **Origin**: Arises from a lack of model knowledge, insufficient training data coverage in a particular feature subspace, or severe distribution shift.
- **Role in COGNIX**: Epistemic uncertainty serves as a **dynamic trust metric**. When an agent encounters novel or shifted inputs, its epistemic uncertainty rises. COGNIX exploits this signal to down-weight the agent's influence on the collective decision.
- **Decomposition**: Computed via predictive variance across Monte Carlo dropout samples or Deep Ensemble member predictions:
  $$\sigma_e = \frac{1}{T} \sum_{t=1}^T (p_t - \bar{p})^2$$

### Aleatoric Uncertainty ($\sigma_a$)
- **Origin**: Arises from inherent measurement noise, sensor resolution limits, optical blur, or physical sensor stochasticity (e.g., rain scattering radar signals).
- **Role in COGNIX**: Informs risk assessment and abstention gating. Even if a model is fully knowledgeable about rain physics, the measurement itself contains irreducible noise.
- **Decomposition**: Under Bernoulli predictive distribution:
  $$\sigma_a = \frac{1}{T} \sum_{t=1}^T p_t (1 - p_t)$$

---

## 2. Belief & Evidence Fusion

When multiple agents provide individual probability predictions $p_1, \dots, p_N$, COGNIX fuses them into a collective belief.

### Epistemic-Weighted Fusion (`EpistemicWeightedFusion`)
Standard averaging treats all agents equally, allowing a degraded or shifted agent to corrupt the collective belief. COGNIX weights each agent inversely proportional to its epistemic uncertainty:

$$w_j = \frac{\text{reliability}_j}{\sigma_{e, j} + \epsilon}$$

Where:
- $\text{reliability}_j \in [0, 1]$ is the empirical calibration accuracy of agent $j$ (defaults to $1.0$).
- $\sigma_{e, j} \ge 0$ is the epistemic uncertainty of agent $j$.
- $\epsilon > 0$ is a numerical stabilization constant (default $10^{-6}$).

The weights are normalized:
$$\bar{w}_j = \frac{w_j}{\sum_{k=1}^N w_k}$$

The collective fused prediction is:
$$p_{\text{collective}} = \sum_{j=1}^N \bar{w}_j p_j$$

### Baseline Average Fusion (`AverageFusion`)
Uniform weighting serving as a benchmark baseline:
$$w_j = \frac{1}{N}, \quad p_{\text{collective}} = \frac{1}{N} \sum_{j=1}^N p_j$$

---

## 3. Inductive Conformal Prediction

Point-estimate probabilities from neural networks or ensembles are notoriously miscalibrated under real-world conditions. COGNIX integrates **Inductive Conformal Prediction** (`ConformalPredictor`) to construct prediction sets with finite-sample coverage guarantees.

### Nonconformity Score
For a calibration set $(x_1, y_1), \dots, (x_n, y_n)$ and predicted class probabilities $\hat{p}(y \mid x)$, the nonconformity score for classification is:
$$s_i = 1 - \hat{p}(y_i \mid x_i)$$

Higher scores correspond to predictions that deviate strongly from the true labels.

### Calibration Quantile
Given a user-specified significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.05$ for $95\%$ target coverage), the nonconformity scores are sorted in an augmented calibration set:
$$\mathcal{S}_{\text{aug}} = \{s_1, \dots, s_n, +\infty\}$$

The conformal threshold $\hat{q}$ is chosen as the $\lceil (n+1)(1-\alpha) \rceil$-th smallest score:
$$\hat{q} = \text{Quantile}_{1-\alpha}(\mathcal{S}_{\text{aug}})$$

### Prediction Sets
At inference time, the conformal prediction set $C(x_{\text{test}})$ is:
$$C(x_{\text{test}}) = \{y \in \mathcal{Y} \mid 1 - \hat{p}(y \mid x_{\text{test}}) \le \hat{q}\}$$

### Theoretical Guarantee
Provided that the calibration and test examples are **exchangeable** (e.g., i.i.d.), the prediction set contains the true label with probability at least $1 - \alpha$:
$$P(Y_{\text{test}} \in C(X_{\text{test}})) \ge 1 - \alpha$$

> [!CAUTION]
> If the test environment undergoes severe, non-exchangeable distribution shift, the empirical coverage may fall below $1 - \alpha$. COGNIX detects this by monitoring the conformal prediction set size.

---

## 4. Risk Assessment & Decision Escalation

COGNIX transforms continuous confidence and uncertainty metrics into discrete, actionable operational directives.

### Risk Levels (`RiskLevel`)
- `LOW`: Fused confidence is high and epistemic uncertainty is low.
- `MODERATE`: Confidence is in an intermediate region, or epistemic uncertainty is moderately elevated.
- `HIGH`: Confidence is low, epistemic uncertainty exceeds safe bounds, or conformal prediction sets indicate high ambiguity.

### Decision Outcomes (`DecisionOutcome`)
| Outcome | Semantic Meaning | Typical Trigger Condition |
|---|---|---|
| `ACT` | Proceed with normal execution | High confidence ($\ge 0.70$), low uncertainty ($\le 0.50$), conformal set size $= 1$. |
| `WAIT` | Precautionary delay; gather more observations | Moderate confidence ($0.40 \le p < 0.70$) in ambiguous zone. |
| `REQUEST_INFORMATION` | Query additional sensors or agents | Moderate risk; additional evidence needed before commit. |
| `ABSTAIN` | Refuse to act due to unresolvable ambiguity | No agent predictions available, or severe uncertainty impasse. |
| `ESCALATE` | Hand over control to human operator or safety fallback | High risk, epistemic uncertainty $> 0.50$, or conformal set size $\ge 3$. |

---

## 5. Execution Modes: Production vs. Research

COGNIX includes a strict mode switch via the `mode` parameter:

### `production` Mode (Default)
- **Design Goal**: Operational continuity and fault tolerance.
- **Behavior**: If an agent fails to implement an uncertainty estimator or raises an exception during inference, the pipeline logs a warning and falls back to a documented confidence-derived heuristic. Non-fatal component errors do not crash the host application.

### `research` Mode
- **Design Goal**: Methodological rigor and scientific reproducibility.
- **Behavior**: **Fails loud and immediately**. If any agent lacks an uncertainty estimator, if an untrained model is passed, if a required calibration artifact is missing, or if non-finite floats appear, the pipeline raises `RuntimeError` or `ValueError`. Heuristic fallbacks are strictly prohibited to prevent artifact contamination in benchmark reports.

---

## 6. Summary Comparison Matrix

| Component | Standard Multi-Agent Baseline | COGNIX Framework |
|---|---|---|
| **Trust Weighting** | Uniform average or static reliability | Dynamic epistemic inverse weighting: $w \propto (\sigma_e + \epsilon)^{-1}$ |
| **Uncertainty Handling** | Ignored or lumped into raw scalar | Disentangled into Epistemic vs. Aleatoric components |
| **Calibration** | Uncalibrated softmax probabilities | Split Inductive Conformal Prediction & Temperature Scaling |
| **Safety Gating** | Hard threshold on raw probability | Multi-tier escalation (`RiskLevel`, conformal set cardinality) |
| **System Provenance** | Opaque black-box outputs | Structured `DecisionResult` with module-level execution metrics |

# 📖 API Reference

This reference documents all **17 public symbols** exported through `cognix.__all__` in COGNIX v0.1.0.

```python
from cognix import (
    __version__,
    DecisionEngine,
    DecisionResult,
    RiskLevel,
    DecisionOutcome,
    StandardAgent,
    AgentRegistry,
    MonteCarloDropout,
    DeepEnsemble,
    UncertaintyDecomposition,
    BayesianBelief,
    EpistemicWeightedFusion,
    AverageFusion,
    ConformalPredictor,
    TemperatureScaling,
    EscalationEngine,
    CognixConfig,
)
```

---

## 1. `__version__`

- **Import Path**: `from cognix import __version__`
- **Type**: `str`
- **Value**: `"0.1.0"`
- **Purpose**: Semantic version identifier for the installed COGNIX distribution package.
- **Example**:
  ```python
  import cognix
  print(cognix.__version__)  # Output: 0.1.0
  ```

---

## 2. `DecisionEngine`

- **Import Path**: `from cognix import DecisionEngine` (defined in `cognix.engine.decision_engine.DecisionEngine`)
- **Purpose**: Primary orchestrator and public entry point for executing the COGNIX multi-agent decision pipeline. Wires components internally into `CognixPipeline`.

### Signature
```python
class DecisionEngine:
    def __init__(
        self,
        config: Any = None,
        uncertainty: Any = None,
        belief: Any = None,
        calibrator: Any = None,
        communication: Any = None,
        attribution: Any = None,
        escalation: Any = None,
        graph: Any = None,
        mode: str = "production",
    ) -> None
```

### Parameters
- `config`: `CognixConfig` instance or `None` (defaults to `CognixConfig()`).
- `uncertainty`: Uncertainty estimator instance, or string shorthand (`"mc_dropout"`, `"deep_ensemble"`, `"none"`).
- `belief`: Belief fuser instance, or string shorthand (`"epistemic_weighted"`, `"average"`, `"uniform"`, `"none"`).
- `calibrator`: Calibrator instance, or string shorthand (`"conformal"`, `"none"`).
- `communication`: Communication strategy instance or `None`.
- `attribution`: Attribution instance, or string shorthand (`"epistemic_shapley"`, `"shapley"`, `"none"`).
- `escalation`: Escalation engine instance, or boolean/string shorthand (`True`, `"escalation"`, `"default"`, `False`, `"none"`).
- `graph`: Graph model instance, or string shorthand (`"epistemic_gat"`, `"standard_gat"`, `"no_graph"`, `"none"`). Note: `"epistemic_gat"` and `"standard_gat"` require the `[graph]` extra.
- `mode`: String, either `"production"` (default, fault-tolerant) or `"research"` (strict assertion fail-loud).

### Public Methods
- `decide(agents: list[Any], input_data: Any, context: dict[str, Any] | None = None, agent_reliabilities: dict[str, float] | None = None) -> DecisionResult`: Run the decision pipeline across the provided agents.
- `configure(config: Any) -> None`: Dynamically update engine configuration.

### Errors
- `ValueError`: If `mode` is not `"production"` or `"research"`.
- `RuntimeError`: In `research` mode if an agent fails, if no predictions are collected, or if an uncalibrated module is executed.
- `ImportError`: If graph shorthand (`"epistemic_gat"`) is passed without `torch` installed.

### Verified Example
```python
from cognix import DecisionEngine, CognixConfig, EpistemicWeightedFusion

engine = DecisionEngine(
    config=CognixConfig(),
    belief=EpistemicWeightedFusion(),
    graph="no_graph",
    mode="production",
)
```

---

## 3. `DecisionResult`

- **Import Path**: `from cognix import DecisionResult` (defined in `cognix.engine.result.DecisionResult`)
- **Purpose**: Immutable dataclass containing the comprehensive outputs, confidence metrics, uncertainty profile, risk assessment, and reasoning steps of a pipeline cycle.

### Fields
- `decision`: `DecisionOutcome` (`ACT`, `WAIT`, `REQUEST_INFORMATION`, `ABSTAIN`, `ESCALATE`).
- `confidence`: `float` in `[0, 1]`. Collective fused prediction probability.
- `calibrated_confidence`: `float | None`. Confidence post-calibration.
- `aleatoric_uncertainty`: `float | None`. Aggregated data noise.
- `epistemic_uncertainty`: `float | None`. Aggregated model ignorance.
- `total_uncertainty`: `float`. Sum of aleatoric and epistemic uncertainties.
- `risk_level`: `RiskLevel` (`LOW`, `MODERATE`, `HIGH`).
- `escalation_required`: `bool`. True if human intervention or fallback is necessary.
- `abstained`: `bool`. True if the engine abstained from taking an action.
- `requested_information`: `bool`. True if additional sensor readings are required.
- `agent_contributions`: `dict[str, float]`. Shapley marginal contributions per agent.
- `agent_trust_weights`: `dict[str, float]`. Normalized weights applied during belief fusion.
- `agent_predictions`: `dict[str, float]`. Individual predictions from each agent.
- `communication_statistics`: `dict | None`. Communication volume or edge attention metrics.
- `calibration_metrics`: `dict | None`. Conformal prediction set, quantile, and coverage target.
- `explanation`: `str`. Concise human-readable natural language summary.
- `reasoning_steps`: `list[str]`. Ordered list of executed decision rationale steps.
- `latency_ms`: `dict[str, float]`. Latency breakdown by pipeline stage in milliseconds.
- `total_latency_ms`: `float`. Total pipeline execution time in milliseconds.
- `timestamp`: `float`. UNIX timestamp of the decision.
- `session_id`: `str | None`. Optional session identifier.
- `metadata`: `dict[str, Any]`. Arbitrary contextual metadata.

### Properties & Methods
- `is_safe_decision`: Property returning `bool`. True if `not escalation_required and not abstained`.
- `dominant_uncertainty_source`: Property returning `"epistemic"`, `"aleatoric"`, or `"unknown"`.
- `summary() -> str`: Human-readable summary string (e.g. `"Decision: ACT (Conf: 0.82, Risk: LOW, Uncertainty: 0.16)"`).
- `to_dict() -> dict`: Serializes all fields into a JSON-compatible dictionary.
- `from_dict(data: dict) -> DecisionResult`: Deserializes a dictionary back into a `DecisionResult`.

---

## 4. `RiskLevel`

- **Import Path**: `from cognix import RiskLevel` (defined in `cognix.engine.result.RiskLevel`)
- **Type**: `enum.Enum` (string values)
- **Members**:
  - `RiskLevel.LOW = "LOW"`
  - `RiskLevel.MODERATE = "MODERATE"`
  - `RiskLevel.HIGH = "HIGH"`
- **Example**:
  ```python
  from cognix import RiskLevel
  assert RiskLevel.LOW.value == "LOW"
  ```

---

## 5. `DecisionOutcome`

- **Import Path**: `from cognix import DecisionOutcome` (defined in `cognix.engine.result.DecisionOutcome`)
- **Type**: `enum.Enum` (string values)
- **Members**:
  - `DecisionOutcome.ACT = "ACT"`
  - `DecisionOutcome.WAIT = "WAIT"`
  - `DecisionOutcome.REQUEST_INFORMATION = "REQUEST_INFORMATION"`
  - `DecisionOutcome.ABSTAIN = "ABSTAIN"`
  - `DecisionOutcome.ESCALATE = "ESCALATE"`
- **Example**:
  ```python
  from cognix import DecisionOutcome
  if result.decision == DecisionOutcome.ACT:
      print("Proceed with nominal action.")
  ```

---

## 6. `StandardAgent`

- **Import Path**: `from cognix import StandardAgent` (defined in `cognix.agents.base.StandardAgent`)
- **Purpose**: Reference wrapper implementing `AgentInterface` for models with an injected `UncertaintyEstimator`.

### Signature
```python
class StandardAgent(AgentInterface):
    def __init__(
        self,
        agent_id: str,
        model: Any,
        uncertainty_estimator: UncertaintyEstimator,
        domain_metadata: dict | None = None,
    ) -> None
```

### Methods
- `predict(observation: Any) -> PredictionResult`: Runs model inference on `observation`.
- `estimate_uncertainty(observation: Any) -> UncertaintyResult`: Calls the injected uncertainty estimator.
- `metadata() -> dict`: Returns agent metadata including `"agent_id"`.

> [!CAUTION]
> The reference `StandardAgent.predict()` imports `torch` internally to evaluate PyTorch models. For non-PyTorch models (e.g., scikit-learn, rule-based, or pure Python sensors), implement a custom agent class exposing `predict()` and `estimate_uncertainty()` methods.

---

## 7. `AgentRegistry`

- **Import Path**: `from cognix import AgentRegistry` (defined in `cognix.agents.registry.AgentRegistry`)
- **Purpose**: Thread-safe (`threading.RLock`) registry for dynamically adding, querying, and monitoring active agents.

### Signature
```python
class AgentRegistry:
    def __init__(self) -> None
```

### Public Methods
- `register(agent: AgentInterface) -> None`: Registers an agent. Resolves the agent ID from `agent.metadata()`.
- `unregister(agent_id: str) -> None`: Removes an agent by ID.
- `get(agent_id: str) -> Optional[AgentInterface]`: Retrieves agent by ID, or `None`.
- `list_agents() -> list[AgentInterface]`: Returns a snapshot list of all registered agents.
- `healthy_agents() -> list[AgentInterface]`: Returns only registered agents where `getattr(agent, "healthy", True)` is `True`.
- Supports the context-manager protocol: `with registry:` acquires the internal `RLock`.

### Verified Example
```python
from cognix import AgentRegistry

registry = AgentRegistry()
registry.register(my_agent)
active = registry.healthy_agents()
```

---

## 8. `MonteCarloDropout`

- **Import Path**: `from cognix import MonteCarloDropout` (defined in `cognix.uncertainty.mc_dropout.MCDropout`)
- **Purpose**: Estimates epistemic and aleatoric uncertainty by keeping dropout active during $T$ stochastic forward passes.

### Signature
```python
class MCDropout(UncertaintyEstimator):
    def __init__(self, T: int = 30) -> None
    def estimate(self, model: Any, observation: np.ndarray) -> UncertaintyResult
```

### Formulation
- $\bar{p} = \frac{1}{T}\sum_{t=1}^T p_t$
- $U_{\text{epistemic}} = \frac{1}{T}\sum_{t=1}^T (p_t - \bar{p})^2$
- $U_{\text{aleatoric}} = \frac{1}{T}\sum_{t=1}^T p_t (1 - p_t)$
- $U_{\text{total}} = U_{\text{epistemic}} + U_{\text{aleatoric}}$

> [!NOTE]
> `MonteCarloDropout.estimate()` requires PyTorch (`[graph]` or `[vision]` extra) to execute stochastic forward passes on a `torch.nn.Module`.

---

## 9. `DeepEnsemble`

- **Import Path**: `from cognix import DeepEnsemble` (defined in `cognix.uncertainty.deep_ensemble.DeepEnsemble`)
- **Purpose**: Computes epistemic uncertainty as the prediction variance across an ensemble of independently initialized models.

### Signature
```python
class DeepEnsemble(UncertaintyEstimator):
    def __init__(self, num_models: int = 5) -> None
    def estimate(self, model_ensemble: list[Any], observation: np.ndarray) -> UncertaintyResult
```

### Formulation
- $\bar{p} = \frac{1}{M}\sum_{m=1}^M p_m$
- $U_{\text{epistemic}} = \frac{1}{M}\sum_{m=1}^M (p_m - \bar{p})^2$

> [!NOTE]
> `DeepEnsemble.estimate()` requires PyTorch to execute model inference over `model_ensemble`.

---

## 10. `UncertaintyDecomposition`

- **Import Path**: `from cognix import UncertaintyDecomposition` (defined in `cognix.uncertainty.decomposition.UncertaintyDecomposition`)
- **Type**: `class` (Utility class providing static methods; not intended to be instantiated directly).
- **Purpose**: Pure NumPy analytical decomposition of total predictive uncertainty into epistemic and aleatoric components. Zero PyTorch dependency.

### Static Methods

#### `decompose_classification(samples: np.ndarray) -> UncertaintyEstimate`
Decomposes classification probabilities using predictive entropy:
- **`samples`**: Array of shape `(T, N, C)` or `(T, C)` representing softmax outputs across $T$ samples.
- **Total Uncertainty**: $H[\bar{p}] = -\sum \bar{p} \log \bar{p}$
- **Aleatoric Uncertainty**: $\mathbb{E}[H[p]] = \frac{1}{T} \sum_t -\sum_c p_{t,c} \log p_{t,c}$
- **Epistemic Uncertainty**: $H[\bar{p}] - \mathbb{E}[H[p]]$ (Mutual Information)

#### `decompose_regression(mu_samples: np.ndarray, var_samples: np.ndarray) -> UncertaintyEstimate`
Decomposes regression models that output both mean $\mu$ and predicted variance $\sigma^2$:
- **Aleatoric**: Mean of predicted variances $\frac{1}{T}\sum_t \sigma^2_t$.
- **Epistemic**: Variance of predicted means $\text{Var}_t(\mu_t)$.
- **Total**: Aleatoric $+$ Epistemic.

### Verified Example
```python
import numpy as np
from cognix import UncertaintyDecomposition

samples = np.array([[[0.85, 0.15]], [[0.80, 0.20]], [[0.82, 0.18]]])
decomp = UncertaintyDecomposition.decompose_classification(samples)
print("Epistemic:", decomp.epistemic)
print("Aleatoric:", decomp.aleatoric)
print("Total:", decomp.total)
```

---

## 11. `BayesianBelief`

- **Import Path**: `from cognix import BayesianBelief` (defined in `cognix.belief.bayesian.BayesianBelief`)
- **Purpose**: Beta-distribution (binary) and Dirichlet-distribution (multiclass) analytical belief state tracking.

### Signature
```python
class BayesianBelief:
    def __init__(self, agent_id: str, alpha: float = 1.0, beta_param: float = 1.0, classes: int = 2) -> None
```

### Class Methods
- `from_prediction(agent_id: str, p: float, n_effective: float = 1.0) -> BayesianBelief`: Initialize from scalar probability $p$ with effective sample weight $n_{\text{effective}}$.
- `from_multiclass_prediction(agent_id: str, probs: np.ndarray, n_effective: float = 1.0) -> BayesianBelief`: Initialize Dirichlet alphas from probability vector.

### Methods & Properties
- `update(observation: int) -> None`: Performs Bayesian update on true class observation.
- `posterior_mean`: Property returning `np.ndarray` of class probabilities.
- `posterior_variance`: Property returning `np.ndarray` of distribution variances.
- `credible_interval(alpha: float = 0.05) -> tuple[float, float]`: Equal-tailed $(1 - \alpha)$ credible interval for binary positive class via SciPy `beta.ppf`.
- `to_belief_state() -> BeliefState`: Converts to `BeliefState` dataclass.

### Verified Example
```python
from cognix import BayesianBelief

belief = BayesianBelief.from_prediction("agent_cam", p=0.85, n_effective=10.0)
belief.update(1)  # Observe positive outcome
low, high = belief.credible_interval(alpha=0.05)
print(f"95% Credible Interval: [{low:.3f}, {high:.3f}]")
```

---

## 12. `EpistemicWeightedFusion`

- **Import Path**: `from cognix import EpistemicWeightedFusion` (defined in `cognix.belief.fusion.EpistemicWeightedFusion`)
- **Purpose**: Fuses multi-agent predictions into a collective probability by weighting each agent inversely to its epistemic uncertainty.

### Signature
```python
class EpistemicWeightedFusion(BeliefFuser):
    def __init__(self, eps: float = 1e-6) -> None
    def fuse(
        self,
        predictions: dict[str, float],
        uncertainties: dict[str, float],
        reliabilities: dict[str, float] | None,
    ) -> FusionResult
```

### Formulation
$$w_j = \frac{\text{reliabilities}.get(j, 1.0)}{\text{uncertainties}.get(j, 1.0) + \epsilon}$$
$$p_{\text{fused}} = \sum_{j} \left( \frac{w_j}{\sum_k w_k} \right) \cdot \text{predictions}[j]$$

### Return Value
Returns `FusionResult` dataclass with `.probability` (`float`) and `.weights` (`dict[str, float]`).

### Errors
- `ValueError`: If `predictions` dictionary is empty.

---

## 13. `AverageFusion`

- **Import Path**: `from cognix import AverageFusion` (defined in `cognix.belief.fusion.AverageFusion`)
- **Purpose**: Baseline unweighted average fusion across all contributing agents ($w_j = 1/N$).

### Signature
```python
class AverageFusion(BeliefFuser):
    def fuse(
        self,
        predictions: dict[str, float],
        uncertainties: dict[str, float] | None = None,
        reliabilities: dict[str, float] | None = None,
    ) -> FusionResult
```

### Verified Example
```python
from cognix import AverageFusion

fuser = AverageFusion()
result = fuser.fuse({"cam": 0.8, "rad": 0.6}, {}, {})
assert result.probability == 0.70
```

---

## 14. `ConformalPredictor`

- **Import Path**: `from cognix import ConformalPredictor` (defined in `cognix.calibration.conformal.ConformalPredictor`)
- **Purpose**: Inductive Split Conformal Prediction for distribution-free calibration with statistical coverage guarantees.

### Signature
```python
class ConformalPredictor(Calibrator):
    def __init__(self) -> None
    def fit(self, cal_outputs: np.ndarray, cal_labels: np.ndarray) -> None
    def predict(self, test_outputs: np.ndarray, alpha: float = 0.05) -> list[ConformalPredictionSet]
    def save(self, filepath: str) -> None
    def load(self, filepath: str) -> "ConformalPredictor"
```

### Parameters
- `cal_outputs`: `(N, C)` array of predicted probabilities on calibration split.
- `cal_labels`: `(N,)` integer array of true class labels.
- `test_outputs`: `(M, C)` array of predicted probabilities for new samples.
- `alpha`: Significance level in `(0, 1)` (default `0.05` for $95\%$ coverage).

### Return Value
`predict` returns a list of `ConformalPredictionSet` dataclasses, each containing:
- `prediction_set`: `list[int]` of included class indices.
- `confidence`: `float` ($1 - \alpha$).
- `quantile`: `float` ($\hat{q}$ nonconformity threshold).
- `coverage_target`: `float` ($1 - \alpha$).

### Errors
- `RuntimeError`: If `predict()` is called before `fit()`.

---

## 15. `TemperatureScaling`

- **Import Path**: `from cognix import TemperatureScaling` (defined in `cognix.calibration.temperature.TemperatureScaling`)
- **Purpose**: Post-hoc probability calibration by optimizing a single scalar temperature parameter $T > 0$ via SciPy L-BFGS-B.

### Signature
```python
class TemperatureScaling:
    def __init__(self, initial_temp: float = 1.0) -> None
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None
    def predict(self, logits: np.ndarray) -> np.ndarray
```

### Parameters
- `logits`: `(N, C)` raw unnormalized model outputs.
- `labels`: `(N,)` integer target labels.

### Return Value
`predict` returns calibrated probabilities of shape `(N, C)` where each row sums to $1.0$.

---

## 16. `EscalationEngine`

- **Import Path**: `from cognix import EscalationEngine` (defined in `cognix.decision.escalation.EscalationEngine`)
- **Purpose**: Evaluates point confidence, epistemic uncertainty, and conformal prediction set size to decide whether to `ACT`, `WAIT`, or `ESCALATE`.

### Signature
```python
class EscalationEngine:
    def __init__(
        self,
        high_conf_thresh: float = 0.7,
        low_conf_thresh: float = 0.4,
        high_unc_thresh: float = 0.5,
        max_conformal_set_size: int = 3,
    ) -> None

    def evaluate(
        self,
        confidence: float,
        epistemic_uncertainty: float,
        conformal_set_size: int = 1,
        max_shapley_value: float = 0.0,
    ) -> EscalationResult
```

### Return Value
Returns `EscalationResult(outcome, escalation: bool, reason: str, risk_level: RiskLevel)`.

> [!IMPORTANT]
> `EscalationEngine.evaluate()` returns an `outcome` enum from `cognix.decision.escalation.DecisionOutcome`. When comparing against `cognix.DecisionOutcome`, compare string values:
> `assert res.outcome.value == DecisionOutcome.ACT.value`

---

## 17. `CognixConfig`

- **Import Path**: `from cognix import CognixConfig` (defined in `cognix.config.schema.CognixConfig`)
- **Purpose**: Pydantic v2 schema for validating, exporting, and managing COGNIX system configuration.

### Sub-Configs
- `uncertainty`: `UncertaintyConfig(method="mc_dropout", mc_dropout_passes=50, ensemble_size=5, ...)`
- `fusion`: `FusionConfig(method="epistemic_weighted", eps=1e-8)`
- `calibration`: `CalibrationConfig(method="conformal", coverage_target=0.95, temperature=1.0)`
- `communication`: `CommunicationConfig(method="top_k", top_k=3, bandwidth_limit=None)`
- `graph`: `GraphConfig(method="epistemic_gat", num_layers=3, hidden_dim=64, num_heads=4)`
- `attribution`: `AttributionConfig(method="epistemic_shapley", num_samples=100)`
- `decision`: `DecisionConfig(high_confidence_threshold=0.70, low_confidence_threshold=0.40, high_uncertainty_threshold=0.5, max_escalation_steps=3)`
- `reproducibility`: `ReproducibilityConfig(seed=42, deterministic=True)`
- `log_level`: `"INFO"` (Literal `DEBUG`, `INFO`, `WARNING`, `ERROR`).

### Methods
- `from_yaml(path: str | Path) -> CognixConfig`: Load and validate from a YAML file.
- `to_yaml(path: str | Path) -> None`: Export configuration to a YAML file.

### Validations
- Automatically validates that `low_confidence_threshold < high_confidence_threshold`, raising `ValueError` if violated.

# ⚠️ Limitations & Methodological Boundaries

COGNIX is an uncertainty-aware research and engineering framework. Developers and researchers must understand its theoretical foundations, implementation boundaries, and empirical limitations.

---

## 1. No Physical Safety Guarantees

> [!CAUTION]
> **COGNIX does not guarantee physical vehicle or robotic safety.**
> Output decisions (`ACT`, `WAIT`, `REQUEST_INFORMATION`, `ABSTAIN`, `ESCALATE`) are mathematical risk classifications based on probabilistic models and configurable thresholds. They must not be deployed as single points of failure in safety-critical systems without certified hardware and physical interlocking mechanisms.

### Statistical Coverage Boundaries
Inductive Conformal Prediction guarantees finite-sample marginal coverage:
$$P(Y_{\text{test}} \in C(X_{\text{test}})) \ge 1 - \alpha$$
**Crucial Condition**: This guarantee holds **only** under the mathematical assumption that calibration data and test observations are **exchangeable** (e.g., drawn identically and independently from the same underlying distribution).

When an agent encounters severe, non-exchangeable out-of-distribution (OOD) shift or dynamic environmental degradation, exchangeability is violated, and empirical coverage may fall below $1 - \alpha$. COGNIX monitors conformal prediction set cardinality as a heuristic indicator of ambiguity, but this is not a mathematical proof of correctness under arbitrary covariate shift.

---

## 2. No Universal Superiority Claims

COGNIX does not claim to outperform baseline fusion or message-passing architectures in all conditions. Its mechanisms were subjected to formal held-out empirical evaluation:

### Closed Held-Out Experiment 2B Evidence
- **Scope**: 15 frozen runs across three architectures (`NoGraph`, `StandardGAT`, `EpistemicGAT`) with five paired random seeds evaluated on 627 scenarios (187,473 scored timesteps per run).
- **Metric-Dependent Trade-Offs**:
  - `StandardGAT` improved Binary Cross-Entropy (BCE) relative to `NoGraph` across all five seeds.
  - However, `StandardGAT` reduced corruption detection F1 score under frozen validation thresholds across all five seeds.
  - `EpistemicGAT` (evaluating the epistemic sender prior $w_j = \frac{1}{1 + E_j}$) showed no descriptive evidence of additional held-out benefit over `StandardGAT`, performing neutral to slightly worse.
- **Classification vs. Safety**: The variable $q_{\text{normal}}$ represents $P(\text{target} = \text{normal})$ on a constructed classification task; it must not be conflated with the probability of physical collision avoidance.
- **Descriptive Evidence**: Five-seed comparisons are descriptive; no inferential claim of universal statistical superiority is asserted.

*See the [Closed Held-Out Research Report](research.md) for complete tabular and figure evidence.*

---

## 3. Computational Latency & Scalability

COGNIX makes no blanket claims of real-time execution speeds. Real-time deadlines depend on hardware, batch size, and selected algorithms:

### Algorithmic Complexity

| Component | Time Complexity | Latency Implications |
|---|---|---|
| **`EpistemicWeightedFusion`** | $O(N)$ | Sub-millisecond in pure Python / NumPy for $N \le 100$ agents. |
| **`MonteCarloDropout`** | $O(T \cdot \text{Cost}(M))$ | Requires $T$ forward passes (typically $T = 30\text{--}50$). High GPU/CPU overhead. |
| **`DeepEnsemble`** | $O(M \cdot \text{Cost}(M))$ | Evaluates $M$ separate model instances. Memory- and compute-intensive. |
| **`EpistemicShapley`** | $O(2^N)$ | Exact Shapley value calculation enumerates all subsets. Practical only for $N \le 10$ agents. |
| **`ConformalPredictor`** | $O(M \cdot C)$ | Fast quantile binary search ($O(\log N_{\text{cal}})$) at test time. |

For high-frequency control loops ($> 50\text{ Hz}$), avoid inline Monte Carlo Dropout passes or exact Shapley enumeration on every frame.

---

## 4. Uncertainty Estimation Approximations

- **Bernoulli Variance Decomposition**: The predictive variance decomposition:
  $$\sigma_e = \text{Var}(p_t), \quad \sigma_a = \mathbb{E}[p_t(1 - p_t)]$$
  assumes a single-parameter Bernoulli distribution. For complex multi-modal or heteroscedastic continuous tasks, this approximation may mischaracterize aleatoric noise.
- **Heuristic Fallback in Production**: In `mode="production"`, if an agent lacks an uncertainty estimator, the engine approximates uncertainty from point confidence. This heuristic is intended solely for fault tolerance in legacy code and **must never be reported as empirical uncertainty quantification in research publications**.

---

## 5. Dependency Isolation Boundaries

- **Pure Base Framework**: Core belief fusion, conformal calibration, risk escalation, configuration, and uncertainty decomposition operate strictly on `numpy`, `scipy`, `pydantic`, and `pyyaml`.
- **PyTorch Isolation**: Model wrappers (`StandardAgent`), neural uncertainty estimators (`MCDropout`, `DeepEnsemble`), and graph networks (`EpistemicGAT`, `StandardGAT`) require PyTorch (`[graph]` or `[vision]` extras). They will raise explicit `ImportError` if called in a base-only installation.

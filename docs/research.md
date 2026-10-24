# 🔬 Research, Provenance & Frozen Benchmarks

This page connects the developer-facing COGNIX v0.1.0 distribution package to its underlying research assets, statistical methodology protocols, and frozen benchmark evidence.

---

## 1. Package Implementation vs. Research Assets

COGNIX originated as an empirical research project investigating whether **epistemic uncertainty** can serve as a unified trust metric in multi-agent artificial intelligence systems.

To maintain research integrity and software reliability, the repository enforces a strict boundary:

```
COGNIX Repository Architecture
├── cognix/ (Production Package: PyPI Distribution 'cognix-uncertainty-framework')
│   ├── Pure Python/NumPy core primitives (DecisionEngine, EpistemicWeightedFusion, etc.)
│   ├── Decoupled, dependency-injected interfaces
│   └── Optional extras (graph, explainability, vision, dashboard)
└── Frozen Research Assets & Historical Artifacts
    ├── docs/research/ (Immutable experimental records, logs, audit documents)
    ├── reports/ (Evaluated telemetry traces and qualification logs)
    └── configs/benchmark/ (Ablation configurations)
```

The released package `cognix-uncertainty-framework==0.1.0` is domain-agnostic software. The historical datasets, CARLA simulator traces, and evaluation reports in `docs/research/` represent frozen experimental evidence and must not be altered.

---

## 2. Statistical Methodology

COGNIX enforces strict statistical discipline to distinguish reproducible algorithmic signals from random initialization artifacts:

- **Document**: [`STATISTICAL_METHODOLOGY.md`](STATISTICAL_METHODOLOGY.md)
- **Multi-Seed Stability**: Evaluating across paired pseudo-random seeds to account for stochastic dropout masks and weight initializations.
- **Paired Hypothesis Testing**: Applying Paired Student's t-test and non-parametric Wilcoxon Signed-Rank tests with prespecified significance levels ($\alpha = 0.05$).
- **Effect Size Focus**: Prioritizing absolute and relative changes in Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and Brier Score over raw p-values.

---

## 3. Closed Held-Out Evaluation (Experiment 2B)

The definitive evaluation of the graph message-passing architecture and epistemic prior was conducted in closed Experiment 2B:

- **Comprehensive Report**: [`docs/research/COGNIX_FINAL_STATUS.md`](research/COGNIX_FINAL_STATUS.md#6-experiment-2b-closed-held-out-graph-evaluation)
- **Frozen Protocol**: 15 frozen runs comparing `NoGraph`, `StandardGAT`, and `EpistemicGAT` across 5 paired seeds (101, 202, 303, 404, 505) evaluated on 627 scenarios (187,473 scored timesteps per run).
- **Core Findings**:
  1. `StandardGAT` improved Binary Cross-Entropy (BCE) relative to `NoGraph` across all 5 seeds.
  2. `StandardGAT` reduced corruption detection F1 score under frozen validation thresholds across all 5 seeds.
  3. The epistemic sender prior ($w_j = \frac{1}{1 + E_j}$) in `EpistemicGAT` was descriptive neutral to slightly worse than `StandardGAT`.
  4. The benefit of graph communication is therefore **metric-dependent**.
- **Immutability Notice**: Experiment 2B is closed and frozen. Its held-out evaluation is an immutable benchmark record.

---

## 4. Key Research and Audit Documents

| Document | Scope & Focus |
|---|---|
| [`PHASE_B_BASELINE_AUDIT.md`](PHASE_B_BASELINE_AUDIT.md) | Architectural audit transforming monolithic experiment scripts into decoupled interfaces. |
| [`STATISTICAL_METHODOLOGY.md`](STATISTICAL_METHODOLOGY.md) | Hypothesis testing protocols, paired comparison rules, and statistical limitations. |
| [`docs/research/REPRODUCIBILITY.md`](research/REPRODUCIBILITY.md) | Seed management, deterministic environment variables, and reproduction guidelines. |
| [`docs/research/METRIC_PROVENANCE.md`](research/METRIC_PROVENANCE.md) | Mathematical formulation and provenance tracking for ECE, MCE, and calibration error. |
| [`docs/research/DATA_PROVENANCE.md`](research/DATA_PROVENANCE.md) | Audit trail of raw datasets, sensor feeds, and synthetic perturbation generators. |
| [`docs/research/CARLA_DATASET_PIPELINE_INTEGRATION.md`](research/CARLA_DATASET_PIPELINE_INTEGRATION.md) | CARLA autonomous driving dataset pipeline and closed-loop replay validation. |
| [`docs/research/MILESTONE_2_ANOMALY_RESPONSE_AND_POLICY_AUDIT.md`](research/MILESTONE_2_ANOMALY_RESPONSE_AND_POLICY_AUDIT.md) | Sensor anomaly detection and policy escalation audits. |

---

## 5. Research Mode vs. Production Mode

Researchers conducting scientific evaluations with COGNIX should always initialize the pipeline with:

```python
engine = DecisionEngine(..., mode="research")
```

In `research` mode:
- Heuristic fallbacks are disabled.
- Missing uncertainty estimators trigger immediate `RuntimeError`.
- Uncalibrated conformal predictors raise descriptive exceptions.
- Mathematical assertions guarantee that every reported number reflects genuine uncertainty quantification rather than a silent fallback.

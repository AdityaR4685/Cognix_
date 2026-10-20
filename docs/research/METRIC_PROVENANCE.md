# COGNIX Metric Provenance

To satisfy rigorous research requirements, COGNIX guarantees that all reported metrics correspond to a formal mathematical definition evaluated against a tracked dataset.

## Core Metrics

### 1. Expected Calibration Error (ECE)
**Source Function**: `calculate_ece(predictions, truths, bins)` in `examples/rq_synthetic_001.py`.
**Formula**: 
$$ \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)| $$
**Description**: Measures the absolute difference between model confidence and actual accuracy across $M$ probability bins.
**Provenance Validation**: ECE is only calculated *after* the complete dataset of `DecisionTrace` outputs is generated. The dashboard displays the mathematically computed ECE from the `DecisionTrace.metrics` object.

### 2. Expected Accuracy
**Source Function**: `np.mean(np.round(preds) == truths)`
**Description**: The proportion of correct decisions vs the ground truth.
**Provenance Validation**: Like ECE, this is calculated post-run and attached to the trace metadata.

### 3. Epistemic Shapley Attribution
**Source Function**: `EpistemicShapley.compute(agents, uncertainty_fn)`
**Formula**: Standard Shapley value computation over the power set of agents, where the characteristic function $v(S)$ is the average epistemic uncertainty of subset $S$.
**Provenance Validation**: Calculated live during `DecisionEngine.decide()` and explicitly logged in `DecisionTrace.attribution`.

### 4. Communication Reduction
**Source Function**: `InformationGainRouter.forward()`
**Formula**: $1 - \frac{\text{messages\_sent}}{\text{max\_possible\_messages}}$
**Provenance Validation**: Derived directly from the active graph topology and logged in `DecisionTrace.communication`.

## Provenance Enforcement
Every metric calculation emits a `MetricProvenance` dataclass guaranteeing traceability to the specific Run ID, Git Commit, and random Seed.

## Closed Experiment 2B provenance

Experiment 2B reporting reads the [authenticated final conclusion](../../reports/carla_train_full_v1/graph_test_final_conclusion_bundle_v1/final_conclusion.json) and [sealed Phase-B metrics](../../reports/carla_train_full_v1/graph_test_phase_B_results_v1/metrics.json); it does not rerun metric calculations. The conclusion JSON SHA-256 is `6abef23828dd6a05fa05ab7e56e8be74072d5674417aafe9e321ee959b0e2f33`, its bundle seal is `cff86433be91a3fdabd3b43568b7fb2c24c0760fa78fddeb298b9ae9cbb0cad8`, and the Phase-B result seal is `ec5ea056cb53050c284816c470845f233d4fe8c566917195658639d20bdee15e`.

`q_normal` represents P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task and must not be interpreted as a probability of physical driving safety. `p_corruption = 1 - q_normal`. BCE uses `q_normal` versus `target_normal`; corruption F1 uses the sealed decisions from per-run thresholds frozen on GRAPH_VAL; AUROC uses `p_corruption` versus `target_corruption`. Metrics are calculated per official scenario and then arithmetically macro-averaged over all 627 scenarios, with ticks 1..299 scored and tick 0 reserved for synchronization validation.

All 107 normal scenarios are single-class, so the prespecified macro-AUROC is undefined for every run; they are retained under the fail-closed rule. All 520 anomaly scenarios contain both classes. No 520-scenario substitute or paired AUROC effect is a primary result. Seed means and paired summaries are copied from the frozen conclusion. The [report-only plotting script](scripts/plot_experiment_2b.py) uses only stored, authenticated paired BCE and F1 differences, with exact values in the [figure-data record](figures/experiment_2b/figure_data.json). See [Section 6 of the final report](COGNIX_FINAL_STATUS.md#6-experiment-2b-closed-held-out-graph-evaluation).

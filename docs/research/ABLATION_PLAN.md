# COGNIX Framework — Ablation Plan

To prove that each component of the COGNIX framework actually contributes to decision safety and robustness, a systematic ablation study must be executed.

## Ablation Methodology

The baseline is the **FULL COGNIX PIPELINE**:
- Uncertainty: MC Dropout
- Fusion: Epistemic-Weighted
- Calibration: Conformal Prediction
- Graph: Epistemic GAT
- Escalation: Enabled

In each run, we disable exactly ONE component and measure the delta in system performance (Accuracy, ECE, Escalation Rate).

## Ablation Scenarios

### 1. Ablation: No Epistemic Weighting
- **Change**: Replace `FusionStrategy.EPISTEMIC_WEIGHTED` with `FusionStrategy.CONFIDENCE`.
- **Expected Observation**: System becomes overconfident when a highly confident (but wrong) agent experiences OOD data.
- **Metric**: Drop in accuracy under OOD conditions.

### 2. Ablation: No Conformal Calibration
- **Change**: Disable the `ConformalPredictor` step in the pipeline.
- **Expected Observation**: The raw fused confidence will misrepresent the true empirical coverage, leading to poor risk assessment thresholds.
- **Metric**: Increase in Expected Calibration Error (ECE).

### 3. Ablation: No Epistemic GAT
- **Change**: Replace `EpistemicGAT` with standard `GATLayer` (no epistemic prior).
- **Hypothesis**: Removing the sender prior changes attention weighting; whether it improves or reduces downstream performance must be evaluated rather than assumed.
- **Closed Experiment 2B evidence**: The evaluated EpistemicGAT sender prior showed no descriptive evidence of improvement over StandardGAT on official TEST. BCE was mixed and slightly worse on average; F1 was essentially neutral to slightly worse. StandardGAT versus NoGraph improved BCE but reduced frozen-threshold corruption F1 across all five seeds. The prespecified macro-AUROC was undefined. See the [final report](COGNIX_FINAL_STATUS.md#6-experiment-2b-closed-held-out-graph-evaluation). This closed experiment must not be rerun or retuned under this general ablation plan.

### 4. Ablation: No Escalation
- **Change**: Set `escalation=False` in `DecisionEngine` config.
- **Expected Observation**: System will ACT on low-confidence/high-risk scenarios, leading to critical failures.
- **Metric**: Increase in catastrophic errors (False Positives in high-risk zones).

## Execution

The script `experiments/run_ablation.py` (to be implemented) will automatically run these 5 configurations across the benchmark dataset and output a LaTeX-formatted table of the deltas.

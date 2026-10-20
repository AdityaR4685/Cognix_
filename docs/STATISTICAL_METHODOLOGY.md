# COGNIX Statistical Methodology

## Overview
COGNIX evaluates the robustness and utility of uncertainty-aware mechanisms through structured statistical analysis. To separate signal from noise, COGNIX requires multi-seed evaluations for all core ablation studies (e.g. `NoGraph` vs `StandardGAT` vs `EpistemicGAT`).

**Experiment 2B exception: frozen descriptive protocol.** The closed Experiment 2B used exactly five prespecified paired seeds (101, 202, 303, 404, and 505), retaining all 15 runs. Its mean paired differences, sample SD (ddof = 1), minima, and maxima are descriptive. No inferential significance test was prespecified or performed. The general guidance below does not authorize more seeds, post-hoc tests, confidence intervals, or a significance claim for this experiment. Its prespecified 627-scenario macro-AUROC was undefined, and paired AUROC effects were unavailable. See the [final report](research/COGNIX_FINAL_STATUS.md#6-experiment-2b-closed-held-out-graph-evaluation).

## Principles
1. **Multi-Seed Stability**: A single seed is anecdotal. COGNIX defaults to using 20–30 random seeds for rigorous evaluation. This allows us to assess average algorithmic behavior across multiple datasets, random initialization matrices, and dropout masks.
2. **Paired Comparisons**: To reduce variance across initialization states, ablation comparisons use the *same* sequence of random seeds. For example, Seed 42 is used for both `StandardGAT` and `EpistemicGAT`. This creates paired observations.
3. **Effect Size**: P-values measure the presence of an effect, but not its magnitude. COGNIX emphasizes reporting the absolute and relative changes in ECE, NLL, and Accuracy, rather than purely relying on binary significance thresholds.

## Methodology

### Null and Alternative Hypotheses
For comparing ablation states (e.g. A vs B, such as `StandardGAT` vs `EpistemicGAT`):
- **Null Hypothesis ($H_0$)**: There is no difference in the true mean metric (e.g., Expected Calibration Error) between Configuration A and Configuration B. ($ \mu_A = \mu_B $)
- **Alternative Hypothesis ($H_1$)**: There is a non-zero difference in the true mean metric between Configuration A and Configuration B. ($ \mu_A \neq \mu_B $)

### Statistical Tests Used
1. **Paired Student's t-test**: Used when the differences between paired observations are approximately normally distributed. (Common for large $N$ scenarios like Accuracy across 30 seeds).
2. **Wilcoxon Signed-Rank Test**: A non-parametric alternative used if the normality assumption is heavily violated (e.g., ECE often clusters near 0 or behaves non-normally).

### Significance and Confidence Intervals
- **Significance Level ($\alpha$)**: We use $\alpha = 0.05$ as the threshold for rejecting the null hypothesis.
- **Confidence Intervals (CIs)**: Where possible, 95% confidence intervals are reported for the mean difference to communicate the precision of the estimate.

### Limitations
- Statistical significance is sensitive to the number of seeds. With high N (e.g. 50+ seeds), even trivial and practically irrelevant differences might reach $p < 0.05$. Therefore, effect size and domain-level significance must always be interpreted alongside the p-value.
- The default synthetic benchmarks may not perfectly proxy the variability of real-world multi-modal data. The standard deviation across seeds in the synthetic benchmark measures stability of the pipeline, but does not guarantee the same bounds when evaluating on out-of-distribution physical robotics datasets.

# Experiment 2B Report Packaging Audit

Reporting date: 2026-10-11 (Asia/Calcutta). Experiment 2B is finished, closed, and immutable. This task packages existing authenticated results; it is not a new scientific experiment.

## Canonical report and modification scope

The repository contains no `.tex`, `.bib`, `.docx`, or existing PDF manuscript. The existing [COGNIX comprehensive report](COGNIX_FINAL_STATUS.md) is the canonical report source for this task. Its original Sections 1–5 are retained with targeted status corrections; Section 6 integrates Experiment 2B, and Section 7 distinguishes the framework contribution from experimental superiority. The opening summary was updated in place; no second paper or bibliography was created.

## Exact modified files

- [CONTINUE_FROM_HERE.md](../../CONTINUE_FROM_HERE.md)
- [README.md](../../README.md)
- [docs/STATISTICAL_METHODOLOGY.md](../STATISTICAL_METHODOLOGY.md)
- [docs/architecture/overview.md](../architecture/overview.md)
- [docs/research/ABLATION_PLAN.md](ABLATION_PLAN.md)
- [docs/research/COGNIX_FINAL_STATUS.md](COGNIX_FINAL_STATUS.md)
- [docs/research/METRIC_PROVENANCE.md](METRIC_PROVENANCE.md)
- [docs/research/REPRODUCIBILITY.md](REPRODUCIBILITY.md)

## Exact created files

- [docs/research/EXPERIMENT_2B_PACKAGING_AUDIT.md](EXPERIMENT_2B_PACKAGING_AUDIT.md)
- [docs/research/experiment_2b_packaging_validation.json](experiment_2b_packaging_validation.json)
- [docs/research/scripts/plot_experiment_2b.py](scripts/plot_experiment_2b.py)
- [docs/research/scripts/validate_experiment_2b_packaging.py](scripts/validate_experiment_2b_packaging.py)
- [docs/research/figures/experiment_2b/figure_data.json](figures/experiment_2b/figure_data.json)
- [docs/research/figures/experiment_2b/paired_bce_differences.svg](figures/experiment_2b/paired_bce_differences.svg)
- [docs/research/figures/experiment_2b/paired_bce_differences.png](figures/experiment_2b/paired_bce_differences.png)
- [docs/research/figures/experiment_2b/paired_f1_differences.svg](figures/experiment_2b/paired_f1_differences.svg)
- [docs/research/figures/experiment_2b/paired_f1_differences.png](figures/experiment_2b/paired_f1_differences.png)

## Scientific claim changes

- Replaced the handoff's pending-TEST status and execution instructions with the closed, immutable final status and report review instructions.
- Replaced an implied graph-prior reliability benefit and the ablation plan's assumed robustness gain with the observed result: StandardGAT lowered BCE and lowered frozen-threshold corruption F1 for all five seeds. The graph benefit is metric-dependent.
- Reported the evaluated EpistemicGAT sender prior as showing no descriptive evidence of improvement over StandardGAT: BCE effects were mixed and slightly worse on average; F1 was essentially neutral to slightly worse.
- Retained undefined macro-AUROC for every method under the prespecified 627-scenario fail-closed rule. Explained the exact 107 single-class normal scenarios, all 520 anomaly scenarios having both classes, and the unavailability of paired AUROC effects. No anomaly-only substitute was introduced.
- Distinguished development improvements from TEST: BCE generalized, development F1 improvement did not, and the prior remained approximately neutral. Threshold-transfer effects or TEST score-distribution shift are possible interpretations, not confirmed causes.
- Scoped inference to five descriptive paired seeds, with no prespecified or performed inferential significance test. The general statistical guidance does not imply that its tests were applied to Experiment 2B.
- Added the required limitations: upstream fitting on all 76 FIT scenarios before separating 15 GRAPH_VAL scenarios, GRAPH_VAL thresholds transferred unchanged, no TEST retuning or post-hoc threshold investigation, and task-specific probability semantics.
- Replaced the architecture's implied physically safe decision with a decision carrying uncertainty/escalation metadata. `q_normal` means P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task, and `p_corruption = 1 - q_normal`; neither establishes physical driving safety.
- Distinguished the uncertainty-aware architecture and its evaluated held-out behavior as contributions from experimental superiority of the tested sender prior. Earlier synthetic fusion and unit-test findings retain their separate historical scope.

## Tables and figure assets

The report includes the required three-method TEST table and four-row paired-effects table, rounded to six decimals. Only StandardGAT's best BCE and NoGraph's best F1 are bolded; AUROC is Undefined for all three methods. Difference directions are explicit: negative BCE differences are favorable, and positive F1 differences are favorable.

Figure 1 is [paired_bce_differences.svg](figures/experiment_2b/paired_bce_differences.svg) with a [PNG copy](figures/experiment_2b/paired_bce_differences.png). Figure 2 is [paired_f1_differences.svg](figures/experiment_2b/paired_f1_differences.svg) with a [PNG copy](figures/experiment_2b/paired_f1_differences.png). Both use 3.5 × 2.8 inch dimensions suitable for an IEEE column, 600 dpi PNGs, readable labels, distinct markers/line styles, a zero reference line, and identical full plotting ranges [-0.045, +0.009]. All five values in both comparisons are included; there is no broken axis, clipped data range, or AUROC chart.

The report-only plotting script authenticates the final conclusion and Phase-B metrics, checks their stored paired summaries agree exactly, and uses only the frozen `per_seed` values. The [figure-data record](figures/experiment_2b/figure_data.json) preserves the full-precision values and source hashes. Both PNGs were visually inspected; the F1 axis label was shortened to avoid clipping at publication dimensions.

## Authentication and consistency validation

The [validation receipt](experiment_2b_packaging_validation.json) records the final checks and hashes of the reporting files. The report validator checks authenticated source seals, stored seed-level summaries, every numerical table cell, exact figure-data equality, metric directions, probability semantics, required limitations, local links and heading targets, and SVG labels/dimensions. It checks the undefined AUROC scenario sets and class-count ranges using only stored Phase-B metrics, never raw TEST labels. The comparison of stored seed means is a reporting consistency check, not a new TEST evaluation.

Authenticated identities:

- Phase-B result seal: `ec5ea056cb53050c284816c470845f233d4fe8c566917195658639d20bdee15e`
- Conclusion bundle seal: `cff86433be91a3fdabd3b43568b7fb2c24c0760fa78fddeb298b9ae9cbb0cad8`
- Conclusion JSON SHA-256: `6abef23828dd6a05fa05ab7e56e8be74072d5674417aafe9e321ee959b0e2f33`

A pre-edit baseline hashes every file under `reports`, `cognix`, `experiments`, `configs`, `tests`, `examples`, and `artifacts`: 18,255 files totaling 5,915,955,631 bytes. It records their exact file inventory, SHA-256 contents, sizes, and modification timestamps. This includes all immutable Experiment 2B artifacts, failed/superseded evidence, and scientific source code. The temporary baseline's SHA-256 is `30de275aebb2da2b41b738236c1ada3992f5aa9e47c77248cecfaf9dd19b6e7f`. The sandbox removed its temporary file after the command ended; the final validator rebuilt the identical canonical inventory and compared its byte-level digest with the retained pre-edit digest. Matching digests establish that the complete recorded inventory and contents remained unchanged. The durable receipt retains both digests and verification counts. Integrity hashing reads bytes only and performs no label/prediction analysis.

No immutable artifact or scientific source was modified. No TEST evaluation, Phase A/B execution, retraining, threshold tuning, recalibration, checkpoint reselection, or TEST metric recomputation from raw labels occurred. Figures use only already-authenticated frozen numbers. No staging, commit, push, pull, merge, or rebase occurred; all reporting changes remain in the working tree. Pre-existing untracked files remain untouched.

## Compilation, warnings, and remaining limits

LaTeX compilation is not applicable: the canonical report is Markdown, no LaTeX source exists, and neither `latexmk` nor `pdflatex` is installed. No successful LaTeX compilation is claimed. Figure generation succeeded, numerical/link validation passed, and `git diff --check` passed. The Markdown report can be reviewed in the existing document viewer.

The following warnings and historical limits are disclosed:

- Git warns that the handoff file's LF endings will become CRLF if Git later rewrites it under the repository's line-ending policy; there is no whitespace-check error.
- The installed Python 3.14 launcher prints a location-discovery warning but successfully runs Matplotlib 3.11.1 and both reporting scripts. The bundled Python 3.12 runtime lacks Matplotlib, so the installed runtime was used. No dependency was installed or scientific environment changed.
- The first plotting invocation failed because of a report-script filename-suffix typo; it was corrected and figure generation rerun successfully. This was report-only generation, with no experimental execution or artifact modification.
- The sealed conclusion's historical status string remains `FINAL_TEST_CONCLUSION_STAGED_FOR_FREEZE`. Its authenticated final bundle seal and the user's final packaging instruction establish the closed reporting status; its bytes were preserved.
- Sealed preparation READMEs, preregistrations, and earlier synthetic reports retain their historical statements. Their contents were not rewritten to imply later results. Earlier unit-test, latency, and synthetic significance claims were not independently revalidated in this packaging task and are not used as Experiment 2B evidence.

No unresolved numerical, probability-semantic, or claim inconsistency was found in the updated Experiment 2B reporting.

## Concise diff summary

Eight existing documentation files changed: 166 lines added and 32 removed. Nine report-only files were created: two scripts, two SVG/PNG figure pairs, a full-precision data record, this audit, and a validation receipt. The main report gained the nine required Experiment 2B subsections, two tables, two figures, source identities, and a framework conclusion. No experiment directory, scientific runtime, bibliography, or Git history was changed. Changes are ready for human review; no commit authorization is requested.

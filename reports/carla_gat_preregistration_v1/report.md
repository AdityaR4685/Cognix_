# Primary Research Question

Does explicitly incorporating valid epistemic uncertainty into graph attention improve collective normal-vs-corruption prediction relative to graph attention without the epistemic prior, under matched data, architecture capacity and paired random seeds?

Primary contrast is EpistemicGAT minus StandardGAT scenario-macro validation AUROC. NoGraph is a fair trainable comparison. These are fixed-representation development results, not independent held-out performance.

# Primary Modalities

Camera, Seg and IMU only, selected solely by the frozen N=20 calibration-validity rule. Canonical node order is Camera, IMU, Seg, corresponding to handoff indices 0, 3, 1. No downstream metrics or graph training were generated.

# GNSS Handling Decision

GNSS calibrated probability and UQ remain null. Preserve its raw member scores and underflow/separation diagnostics. One secondary exploratory condition adds mean and population standard deviation of its five raw scores to each method’s collective readout, with no GNSS node or prior. Its four-parameter logistic auxiliary head starts as the identity of the primary collective prediction and is trained jointly only in that future secondary condition. Use exactly the primary samples, splits and seeds; no GNSS-only corruptions. It measures raw-context contribution, not GNSS corruption detection, and cannot replace the primary experiment.

# Graph Sample / Target

One graph per scenario/tick/constructed observation, three valid agent nodes, six directed edges. Clean target_normal=1; single valid-modality controlled corruption target_normal=0. Untouched nodes retain exact parent features and agent outputs. Per-node targets are recorded but not optimized. This is constructed normal-vs-pseudo discrimination, not official CarlAnomaly anomaly probability or driving safety. Every sample carries canonical row/pair keys, scenario/town/tick, split, source hashes, recipe/severity/seed/window and upstream/protocol hashes; graph_schema.json specifies the complete contract.

# FIT Pseudo Sampling

No pseudo generation performed. Future parents are FIT_NORMAL only, ticks 1..2999. Frozen order: camera_brightness_shift, camera_occlusion, seg_region_corruption, imu_spike, imu_bias_scale. Select recipe[t % 5], max one pseudo per tick, production base seed 0 and call seed t. Original recipe definitions/default severities and causal window 12 stay unchanged. GNSS recipes are ineligible because their corruption would be invisible to all primary nodes; this is a modality-policy decision. Clean features are reused, and untouched blocks copied. The existing compact no-effect guard removes both members of a graph pair with no replacement, preserving exact 1:1 balance. Retain and report any later p/E/A collisions instead of filtering by predictive outcomes. Maximum dataset: 71,976 graph-train and 17,994 graph-validation rows before fixed guards; realized counts await the separately authorized export. CAL pseudo is never reused as graph-training data.

# Node Features

Each node receives [prob_normal, epistemic, aleatoric], with canonical entropy quantities in nats, from the accepted fixed agents. All three methods receive the same information. Total is redundant and omitted; raw high-dimensional representations and modality one-hot features are omitted. No fitted scaler. EpistemicGAT additionally routes the exact same E through its fixed sender attention prior, so the comparison isolates prior allocation beyond E as an ordinary covariate. Cast to float32 only at graph-model input; original agent calculations remain float64.

# Graph Topology

Receiver-row/sender-column adjacency is [[0,1,1],[1,0,1],[1,1,0]]. Fully connected directed communication among the three agents, six edges, no self-edges or automatic self-loops, no CARLA spatial relationships. All methods use the same node order and graph instances.

# NoGraph Specification

Future trainable shared-node MLP: bias-free 3→12→1, ELU between layers, hidden dropout 0.1, Xavier initialization, q_normal=mean(sigmoid(node_logit)). It uses all identical node inputs with no message passing and has 48 parameters against 50 per GAT. Use the same loss, optimizer, batches, validation rule and seeds. The frozen generic identity NoGraph plugin remains unchanged and is not substituted for this primary trainable baseline.

# StandardGAT Specification

Two single-head layers, input 3, hidden 8, output 1. Bias-free projections/attention, LeakyReLU(0.2) logits, ELU hidden layer, linear node output, post-softmax attention dropout 0.1, mean of node sigmoid probabilities. Adam lr=0.001, weight_decay=0.0001, batch size 256, maximum 100 epochs; graph BCE only. Early stopping: validation scenario-macro BCE, patience 10, min_delta=1e-5; save minimum observed BCE checkpoint, earliest exact tie. Xavier initialization; seeds 101,202,303,404,505. No scheduler, AMP, TF32, gradient clipping or search. Future report/adapter-side batching composes the unchanged generic layers; existing generic .fit is per-observation and is not the mini-batch execution path.

# EpistemicGAT Specification

Same graph data, p/E/A features, architecture, 50 trainable parameters and training specification as StandardGAT, with identical cloned initial tensors per seed. Reuse the existing generic mechanism at both layers:

`w_j=1/(1+E_j)`

`e_ij_epi=e_ij+log(w_j)=e_ij-log(1+E_j)`

`alpha_ij=exp(e_ij)*w_j / sum_k(exp(e_ik)*w_k)` over incoming neighbors.

Higher sender E reduces its attention for fixed logits; target E has no separate prior factor. E is the canonical entropy gap, not a standard deviation. The prior denominator 1+E is at least one; require finite nonnegative E. Prior-disabled/all-zero E reduces to ordinary attention. No learned strength, normalization or FIT-statistic amplification: fixed strength one in raw nats. Tiny E can round to a neutral float32 prior or produce weak effects; null results are acceptable and will not trigger strength tuning. Frozen synthetic implementation is unchanged. Learned node outputs are contributions to the graph readout, not newly calibrated modality probabilities.

# Graph Train / Validation Split

Seed 2027, PCG64 permutation of sorted frozen FIT IDs; first three positions validation, remaining twelve train.

GRAPH_TRAIN: Town01/scenario-10, Town01/scenario-11, Town01/scenario-13, Town01/scenario-15, Town01/scenario-2, Town01/scenario-3, Town01/scenario-5, Town01/scenario-6, Town01/scenario-7, Town02/scenario-1, Town02/scenario-10, Town02/scenario-4.

GRAPH_VALIDATION: Town01/scenario-12, Town01/scenario-14, Town02/scenario-2.

CAL reserved: Town01/scenario-1, Town01/scenario-4, Town01/scenario-8, Town01/scenario-9, Town02/scenario-3.

Pseudo inherits parent membership; whole scenarios only. The fixed one-class agents already fitted all 15 FIT scenarios. This is a supervised graph holdout conditional on a fixed representation that saw validation scenarios, not full-pipeline unseen-scenario evaluation. No agent refit occurs. Graph-validation data alone controls early stopping and thresholds; CAL is excluded from graph development.

# Metrics

Primary: equal-scenario macro AUROC on corruption probability 1−q_normal. Secondary: average precision/AUPRC, corruption F1, accuracy, balanced accuracy, Brier, binary probability ECE, BCE and pooled/per-scenario/per-recipe versions. ECE: 15 fixed equal-width probability bins, left-closed/right-open except final inclusive edge, sample-weighted absolute mean probability/target gap. Select a per-method/per-seed corruption threshold using graph-validation macro F1 over unique probabilities plus 0/1; largest threshold on ties, corruption iff p≥threshold. F1 is development/selection-reused, not independent test performance. Report artificial 1:1 prevalence, no-effect skips and every undefined metric as null with reason. metrics_spec.json freezes every formula and tie rule.

# Paired-Seed Statistical Plan

Seeds [101,202,303,404,505] for all stochastic methods. Same rows, split and epoch shuffles for every method; identical Standard/Epistemic initialization and initial dropout RNG state. Retain all runs/failures, no best seed or replacement. Report all five AUROC differences, mean/median/sample SD/min/max, paired dz (null at zero SD), and conditional t4 seed interval. Exact two-sided 32-pattern sign-flip test has minimum attainable p=0.0625, so five pairs cannot establish p<0.05 with that test. A 10,000-draw seed-606 whole-scenario bootstrap preserves ticks/pairs/method alignment and is descriptive with only three validation scenarios. No tick-level pseudo-replication or generalization/superiority claim from one seed. Secondary metrics/condition and NoGraph contrasts remain exploratory.

# Conformal Placement

No fitting. Conformal sits after graph method/model-selection and output-pipeline freeze, under a future calibration/exchangeability protocol. All five CAL scenarios stay outside graph training and selection, but already influenced upstream probability calibrators. Reusing them naively as independent split-conformal calibration cannot establish coverage; a later approved protocol must resolve this dependence and temporal/scenario exchangeability. No new data acquisition, upstream refitting or TEST is authorized here.

# Kaggle Execution Plan

kaggle_execution_plan.md freezes verification → local FIT pseudo compact export → frozen split validation → future trainer fixture checks → NoGraph/StandardGAT/EpistemicGAT → five paired seeds → checkpoints/predictions/metrics → separately frozen secondary analysis. One T4, deterministic algorithms where supported, no distributed training. Compact N=20 data alone cannot recreate pixel corruptions; local raw TRAIN access is required for the future FIT pseudo exporter, then only compact outputs go to Kaggle. No notebook upload or training was performed.

# Frozen Protocol Hashes

- protocol.json: `68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203`
- graph_schema.json: `f1a0f509416cc2dee8265463c43c1596c4b981c1b7f6113645b53f3eda4b2fd3`
- split_manifest.json: `e95118b4149aa9d7e203f45679fb771a1123b97662dbe3548b9435ca8588b479`
- metrics_spec.json: `b6fe616c4c1049222307f1e3d914a221b4553fbb18429a739ca448413bed0fdd`
- kaggle_execution_plan.md: `24df0999ce81c697a95c085723ce4d40b430e5ad6cd6933ef126fbbe6d91a336`

All protocol files, including this report and the minimal writer/integrity records, are listed in SHA256SUMS. Its separate SHA256SUMS.sha256 hashes the complete ledger without self-reference. Protocol dependencies also carry hashes inside protocol.json. Every N=20 cache/handoff byte hash, prior repository hash and frozen synthetic artifact hash matched the initial snapshot. Existing nontraining graph-mechanism tests: 4 passed. No source/test/core changes; only the preregistration report directory was added.

# Ready / Not Ready to Generate Graph Training Data

Protocol ready for separately authorized implementation and FIT-only data generation. No graph data has been generated. First validate the future adapter against this frozen schema, source hashes, causal semantics, balancing and exact untouched-parent rules; publish the compact FIT-only export and skip/provenance ledger.

# Ready / Not Ready for Kaggle Training

Not ready now: FIT pseudo graph export, tested mini-batch trainer/NoGraph implementation and execution environment lock do not exist yet. No Kaggle upload or training occurred. Protocol readiness is not implementation readiness or authorization to launch.

# Recommended Next Step

Authorize only the frozen FIT-derived pseudo/graph compact-export adapter and its integrity validation as the next milestone. Reuse existing clean features and fixed accepted agent mappings. Then assess artifact/trainer readiness before separately authorizing the paired-seed Kaggle run. Stop here: no graph generation, training, TEST, conformal fitting, further acquisition, calibration/feature/recipe changes or commits/pushes.

# Frozen Development Findings

This is a design-only final-evaluation preregistration. The completed sealed `reports/carla_gat_paired_execution_v1/` milestone is authoritative. Both GAT methods descriptively exceed NoGraph scenario-macro AUROC in all five frozen seeds. StandardGAT and EpistemicGAT are extremely close; EpistemicGAT has no consistent superiority. Mean Epi−Standard AUROC difference is **+2.6450964370616782e−05**, exact two-sided sign-flip **p=0.875**, and the preregistered conditional seed interval is **[−5.4543476426879355e−05, +0.00010744540516811291]**. The interval concerns initialization variation on fixed development scenarios, not independent population generalization.

The canonical entropy-gap E is generally small. The fixed sender prior remains weak, including possible neutral float32 rounding. Preserve this negative result. No amplification, normalization, new architecture, experiment, best seed, representation, calibration, recipe or threshold change is justified by these outcomes. No graph training/refitting, conformal fitting, secondary GNSS, TEST payload access, actual partition, commit or push occurred here.

`frozen_method_bindings.json` identifies every selected checkpoint by local path, epoch, file/content hash, method, seed and exact recorded threshold. All three methods and seeds **[101,202,303,404,505]** remain fixed. NoGraph is the shared-node MLP actually evaluated, not the generic identity NoGraph plugin.

# End-to-End COGNIX Pipeline Map

The implemented development path is frozen compact features → M1 uncertainty agents → M4 graph communication plus M2 readout → collective normality evidence. Final official-label M3 and M5 integration remains future adapter work; a fully implemented final evaluator is not claimed.

For modality m and tick t, the frozen one-class members produce scores s; the existing upstream calibrator maps each member to u=σ(a·s+b). Then p=mean(u), A=mean(H_B(u)), and E=max(0,H_B(p)−A). H_B is binary entropy in natural-log nats with unchanged production clipping. The three nodes are **Camera, IMU, Seg**, each carrying **[p,E,A]**. Source computation is float64; model inputs are float32 and the raw float64 E supplies the frozen reciprocal prior calculation. Total entropy is omitted as redundant with E+A.

For either GAT, receiver i and sender j use l_ij=LeakyReLU(aᵀ[Wh_i∥Wh_j]). Standard uses w_j=1; Epistemic uses w_j=1/(1+E_j), at both layers. Incoming α_ij is proportional to exp(l_ij)·w_j over j≠i. The adjacency has six directed edges, zero diagonal and no self-loop. The first message layer uses ELU; the second gives node logits z_i. In inference, dropout is disabled. The fixed collective result is **q=(1/3)Σ_i σ(z_i)**. NoGraph applies its existing shared bias-free 3→12→1 MLP independently to nodes, then uses the same mean of sigmoids.

The final class interface is **P(normal=0)=q; P(anomaly=1)=1−q**. These are candidate score coordinates for official-label conformal prediction; q retains its original pseudo-trained semantics. M3 constructs label-support sets after the collective output. M5 receives q, modality UQ, support sets, provenance and diagnostic readout/attention. `pipeline_map.json` contains the full equations and implementation references.

# Graph/Fusion Relationship

**The graph readout already constitutes collective fusion.** M4 learns/routes cross-agent messages; its fixed mean(node sigmoid) readout is the experiment's M2 operation. A second average, epistemic weighted fuser, average of logits, probability recalibrator or learned head would change the method.

A distinct fusion interface is justified when graph communication returns unfused node beliefs; here that interface is fulfilled by the existing readout. Generic `CognixPipeline` independently refines node predictions and then invokes a belief fuser. It also uses the generic `[1−fused,fused]` class arrangement. Reusing that generic orchestration without an explicit adapter could both fuse twice and invert the registered official target. Generic COGNIX is unchanged. Any future wrapper must reproduce the existing scalar q and explicitly map its class order.

Per-agent canonical E/A remains available. H_B(q) is a collective predictive-entropy diagnostic, not an established collective E/A decomposition. Graph-seed disagreement is descriptive initialization sensitivity, not a replacement canonical epistemic estimate.

# Calibration Dependency Audit

The five existing CAL scenarios are Town01/scenario-1, scenario-4, scenario-8, scenario-9 and Town02/scenario-3. They fitted the upstream probability mappings. That mapping is part of the scoring function, even though CAL did not train the graph. Reusing these examples for ordinary split conformal fails its independent-calibration rank argument.

All 15 FIT scenarios fitted the upstream normality representations; the 12 GRAPH_TRAIN scenarios additionally trained graph models. GRAPH_VALIDATION consists of Town01/scenario-12, Town01/scenario-14 and Town02/scenario-2. Those scenarios controlled checkpoint/early-stopping selection and per-method/seed F1 thresholds, and upstream representations already saw them. They cannot automatically become independent final conformal data. Freezing an outcome-dependent scorer now does not retroactively remove its dependence on these examples.

Threshold selection alone would not change a conformal score that never uses the threshold; however, graph checkpoint selection already does. Existing thresholds can be frozen for future independent descriptive hard-label evaluation. No further threshold selection uses final calibration/evaluation data.

Conditioning on all completed development, fitting and model selection is compatible with future **disjoint, exchangeable official-task calibration/evaluation**. Poor probability transfer can harm set efficiency; it does not itself defeat the rank construction when its independence and exchangeability assumptions hold. CAL pseudo targets do not establish exchangeability with official anomalies. Temporal dependence and scenario sampling/provenance are separate problems, audited below. `dependency_audit.json` records dependencies, exact existing role lists and guarantee limits.

# Historical TEST Exposure

**Amendment 001 — pre-TEST-access historical-evidence correction.** The prior report incorrectly left the town unresolved and omitted partial scenario-10 exposure. Recovered preserved evidence supplied by the user establishes exactly:

- **`test/anomaly/Town01/change-weather/scenario-1` — complete historical exposure.** The main probe captured the scenario completely; earlier 64 KB / 8 MB probes also inspected its beginning.
- **`test/anomaly/Town01/change-weather/scenario-10` — partial historical exposure.** The same main probe continued into this scenario and reached **93 RGB frames**.

Both came from the official **`carlanomaly-base-test.tar.gz`** archive. The main probe requested inclusive bytes **0–125,829,119**, received **HTTP 206**, and used **sequential gzip/tar parsing**. Temporary probe files were deleted afterward. These are historical facts; no probe, payload or metadata inventory was reopened or executed in this amendment.

Both exact IDs are **SCHEMA_EXPOSED_DIAGNOSTIC_ONLY** and excluded as **entire scenarios from both FINAL_CONFORMAL_CAL and FINAL_EVALUATION**, regardless of complete versus partial exposure. These exact exclusions replace the provisional cross-town quarantine. No exposed ticks are used to allocate or rehabilitate remaining ticks within a scenario.

The **historical file-level ledger remains incomplete**. No evidence presently establishes any additional official TEST scenario IDs; none are inferred or claimed. The prior source documentation and synthetic fixtures are retained as contextual evidence, while `recovered_historical_evidence.json` records the authoritative recovered facts. `test_exposure_record.json` and `exposure_evidence.md` distinguish the two exposure extents and the remaining file-level limitation. Official metadata counts, provenance and actual final membership remain unaudited; no partition has occurred.

# Proposed FINAL_CONFORMAL_CAL / FINAL_EVALUATION Split

The algorithm is defined only; no official TEST inventory or payload was traversed and no actual IDs/counts were allocated. Start with approximately **20%/80%**, deterministic **PCG64 seed 2028**, distinct from earlier split/training seeds. Use complete scenarios and full split-qualified identities; never share ticks/windows across roles.

After separate metadata authorization and the exposure/provenance gates, stratify by **town × official normal/anomaly directory condition × anomaly type (NORMAL for normal)**. These metadata conditions are not timestep labels. For each nonempty stratum containing N_h unseen complete scenarios, assign **ceil(N_h/5)** to calibration and the remainder to evaluation. Sort strata and IDs lexicographically, use one PCG64(2028) generator across strata in order, permute each sorted ID list once, take its first calibration quota and sort the final lists. No redraws, performance-based selection or balancing by tick outcomes. A one-scenario stratum goes to CAL with zero evaluation scenarios; retain and disclose that composition limitation.

Before final payloads/predictions, seal the authorized metadata inventory, exact historical exclusion supplement, family/provenance audit, realized counts/composition, role manifest, ordered permutations and pinned NumPy/RNG version. If linked drives/variants require larger groups, stop for a group-level metadata-only amendment rather than split linked scenarios.

At α=.05, a finite conformal rank requires **at least 19 calibration scenarios per stratum**. With n=19 the cutoff is the maximum calibration score. With n=39 the rank is 38 of 40 augmented entries and resolution is .025, which offers more rank granularity; it is not a precision or power theorem. Under ceil(N/5), those counts first occur at **N=91** and **N=191** eligible scenarios per stratum respectively. Sparse official type/town strata may make this design uninformative. Thousands of ticks do not increase n.

Counts are unknown, so 20% is a sealed starting rule with an explicit adequacy gate, not a claim of sufficient calibration or evaluation precision. If counts require changing the ratio/group definition, seal a metadata-only amendment **before any predictions, fitting or sensor content**. Do not enlarge calibration, pool strata or change α after seeing sets. `test_partition_protocol.json` freezes the exact algorithm and gates.

# Scenario / Temporal Exchangeability

Individual ticks and overlapping IMU windows are correlated. They are not independent conformal samples. The selected calibration unit is an entire labeled finite scenario. Applying a fixed maximum across its ticks requires no within-scenario tick permutation or independence assumption.

Between-scenario score exchangeability within each registered stratum remains required, conditional on frozen development. Different directories do not prove it. Audit simulator seeds, route/driving repeats, normal/anomaly variants and shared family identities using provenance; town/type stratification alone cannot fix cross-scenario links. A random partition does not make arbitrary future drives exchangeable. Uniform assignment within metadata strata can support a finite-population randomized interpretation for fixed scores, distinct from generalization to newly generated drives.

Hierarchical conformal needs its own sampling and symmetry assumptions; ordered scenario ticks cannot automatically be treated as exchangeable members of a group. That distinction is supported by [Lee, Barber and Willett](https://arxiv.org/abs/2306.06342v4). Scenario aggregation or one-random-tick subsampling can answer different coverage targets, but neither automatically covers the full official label sequence. Scientific validity, rather than narrower sets, determines the selected construction.

# Conformal Construction

For each of the **15 fixed method/seed scorers**, define candidate nonconformity **s_j,t(y)=1−P_j,t(y)**. For each final calibration scenario j define **S_j=max_t s_j,t(Y_j,t)**, counting one S per scenario. Use all synchronized eligible ticks **t≥1** through that scenario's actual end, with the frozen causal trailing window [max(0,t−11),t]. The existing feature contract excludes tick 0 because it needs at least two temporal samples. Do not generate final pseudo examples, subsample/balance ticks or filter by predictions.

Within stratum h, sort its n_h calibration scores augmented by +∞, and use rank **k_h=ceil((n_h+1)·.95)**. Compute it with exact integers; include ties. For n_h=0 or a rank beyond n_h, **Q_h=+∞**. The registered prediction cutoff is the conservative envelope **Q=max_h Q_h**, including sparse strata. It avoids an oracle normal/anomaly/type selector at prediction time. Define **C_j,t={y∈{normal,anomaly}:1−P_j,t(y)≤Q}**. An infinite cutoff produces both labels everywhere and must be recorded as trivial coverage/full abstention.

For an exchangeable new scenario in stratum h, the augmented rank bound gives Pr(S_new≤Q_h)≥.95. Since Q≥Q_h, inclusion of every true tick label follows whenever S_new≤Q_h. Thus **Pr(all eligible tick labels in a new scenario belong to their sets)≥.95**, marginal over calibration/new scenarios, conditional on frozen development and stratum assumptions. This is our direct block/envelope derivation from the standard rank construction described by [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511v6). It is not a claim that CarlAnomaly's provenance has met the assumptions.

The bound applies per method/seed, not jointly over 15 scorers, all future scenarios, each realized calibration sample, each feature value, or selected singleton cases. It gives no guarantee for arbitrary live shift or physical actions. Finite n is necessary for an informative cutoff, not sufficient for small sets or precise empirical coverage.

Generic `ConformalPredictor.fit` computes 1−probability_of_true_class per input row and `predict` uses the correct augmented rank and inclusive comparison. It has no groups, block maxima, hierarchical weighting or cluster provenance. Its guarantee is example-level marginal coverage under independent fitting/exchangeable scores. `TwoStageConformalPredictor` merely fits separate individual/collective predictors and supplies no hierarchical or simultaneous guarantee.

A **minimal adapter-level extension** can compute block scores, stratum ranks and the global envelope, store their complete provenance, and apply the existing candidate-score comparison. Ordinary tick-level fitting must not be relabeled scenario fitting. Generic score-state/persistence lacks the stratum/envelope metadata; keep adapter state explicit rather than manufacture probability rows to imitate block scores. A reusable generic grouped API would require broader contracts but is unnecessary here. No code extension or fitting occurred.

# Final Target Semantics

Development labels meant clean versus controlled TRAIN-derived pseudo corruption, under artificial paired prevalence. Final labels mean **official observation normal versus official observation anomaly**, read and aligned from the official timestep annotations in a later authorized stage. Directory-level normal/anomaly/type remains separate provenance; an anomaly directory can contain all-False tick labels. Missing annotations/alignment must halt evaluation; never assign all ticks positive from a directory name.

q remains frozen pseudo-trained **normality evidence**, and 1−q anomaly evidence. Its probability scale is not an established official-task posterior. Official-label conformal predicts candidate labels under the stated sampling assumptions; it does not recalibrate upstream probabilities or make pseudo calibration transfer automatically. Neither scores nor sets mean physical driving safety, a certified safe action or P(safe).

# Final Metrics

`final_metrics_spec.json` freezes formulas, class order, denominators and units. Retain scenario-macro **AUROC (primary), AUPRC as stepwise AP, F1, accuracy, balanced accuracy, Brier, BCE and 15-bin probability-reliability ECE**, plus pooled and per-scenario descriptive versions. Positive class is official anomaly; ranking score is 1−q. Hard predictions use each exact already selected development threshold, with ≥ as recorded. BCE clipping stays 1e−7; ECE edges are k/15 with a closed final endpoint and empty-bin zero contribution. No final threshold search.

Preserve the existing **strict undefined-value convention**: any undefined within-scenario metric makes its all-scenario macro null with explicit IDs/reasons. Normal-only scenarios can therefore make the primary strict macro AUROC and its contrast undefined. Do not relabel directory conditions as tick targets or silently remove these scenarios. Freeze one separately named **secondary two-class-eligible scenario-macro AUROC**, whose membership depends only on whether official labels contain both classes; report its denominator/excluded IDs, use the same subset for every method/seed and never promote it after results. Pooled AUROC also remains secondary.

Conformal diagnostics per scorer: simultaneous scenario coverage (all eligible tick labels covered); equal-scenario mean tick coverage; pooled tick coverage; signed coverage−.95 and nonnegative undercoverage gap; equal-scenario mean set size; abstention rate |C|≠1; separate empty/ambiguous-set rates; and counted per-stratum diagnostics. The scenario coverage metric matches the bound. Ticks and 5×J seed/scenario cells are never independent coverage trials. Undefined/unfitted results stay null; valid infinite cutoffs must report 100% coverage, size 2 and abstention 1 as an uninformative result.

All assigned scenarios, labels, model/seed results and failure reasons remain in the ledger. Missing modalities/labels, unreadable scenarios or nonfinite predictions make the milestone incomplete; no convenient replacement or performance-based removal.

# Seed Aggregation Policy

Primary analysis retains **five paired seed results**, taking the arithmetic mean of Epi−Standard differences when all primary endpoints are defined. NoGraph is the secondary graph/no-graph comparison. **No prediction ensemble is introduced.** This preserves the existing methodological comparison and avoids a new scorer/conformal distribution.

Retain each difference, mean, median, sample SD, min/max, the original t4 conditional seed interval and exact 32-pattern two-sided sign-flip summary. Five pairs have minimum attainable two-sided p=.0625; they cannot demonstrate p<.05 with that test. The seed interval describes conditional initialization variation, not population generalization. Missing/undefined pairs are not replaced.

Calibrate each method/seed separately using the same final scenario roles. Average their coverage metrics descriptively only; no joint 95% coverage claim, ensemble, set-intersection or consensus guarantee. Any descriptive scenario bootstrap uses 10000 PCG64(606) whole-scenario draws within frozen metadata strata, shared across methods/seeds, retaining stratum counts and equal-scenario weighting. It does not refit conformal or thresholds. Tick bootstraps and new post-result inferential endpoints are excluded. `seed_aggregation_spec.json` records the policy.

# Decision / Attribution Semantics

M5 describes normality/anomaly evidence, input uncertainty, conformal support, abstention and a review request. Empty sets mean inconsistent support; two-label sets mean ambiguity; both lead to abstention. An anomaly singleton requests review. A normal singleton reports normal-label support without vehicle-control authorization. Invalid inputs, unfitted conformal, missing modalities or support/hard-threshold disagreement require abstention/review. These are structural rules, not tuned numerical risk thresholds.

Display all 15 results. Only unanimous normal singletons with consistent hard labels yield a common normal-evidence summary; ambiguity, anomaly support, failure or disagreement requests review. This operational agreement summary is not a newly guaranteed conformal set and does not certify selected-case correctness. Additional E/risk thresholds would need separate development data/protocol excluding both final sets; none are chosen here.

Record per-agent p/E/A, incoming attention at both layers, and node probabilities when faithfully exposed. Contributions v_i/3 sum exactly to q, but graph node i already mixes senders, so these are readout summands rather than raw-modality causal influence. Attention describes routing, not causal importance. Generic EpistemicShapley evaluates its own belief coalition game, which cannot automatically explain the frozen graph q. Graph-consistent intervention/coalition baselines and absent-node semantics require a separately preregistered attribution study. No fabricated Shapley or counterfactual result is introduced. `decision_semantics.json` freezes these boundaries.

# GNSS Disposition

GNSS stays outside the primary graph/conformal/decision inputs. Its structurally separated calibration remains invalid; calibrated probability/UQ is null, with no recalibration or new transformation. The previously contemplated raw-score auxiliary ablation might answer a separate contextual-information question, but cannot rescue the negative epistemic result or test calibrated GNSS anomaly detection. It adds a new learned head and analysis burden. **Defer it to future work** after the primary official evaluation rather than run it now. Integrity-only byte hashing of existing TRAIN GNSS files is not a GNSS scientific experiment.

# Live CARLA Demonstration Plan

The eventual loop is live front RGB/semantic segmentation/IMU → identical frozen features → existing agent members/calibrators → p/E/A → all fixed graph models/readouts → exported conformal support/decision layer → dashboard. Reset causal buffers per finite declared scenario; validate tick synchronization, schema and simulator compatibility. Do not infer sampling frequency, replace features or use an unregistered best seed.

After final pipeline freeze, export locally the immutable feature/agent/model source/schema, fitted one-class members, accepted probability mappings, all 15 selected checkpoints/thresholds, graph order/adjacency/dtypes/prior, full conformal block scores/counts/strata/cutoffs/envelopes/alpha, exclusion and split manifests, target/decision protocol hashes, environment pins and golden replay evidence. Restore without fit; do not bundle raw TEST frames.

The dashboard must identify LIVE SIMULATION and show evidence, UQ, support/abstention/review and routing diagnostics with measured latency and coverage status. No control actuation or certified-safety claim is proposed. Infinite streams/new conditions have no registered scenario-exchangeability guarantee; show experimental status and require review rather than silently refresh calibration. The demonstration is not a substitute for held-out evaluation. `live_demo_plan.md` gives the export/compatibility plan; no live loop or export was implemented.

# Protocol Hashes

**Amendment history:** the complete original 26-file directory is preserved byte-for-byte in `amendment_history/pre_amendment_001/`, including its 24-artifact manifest and detached hash. Previous manifest SHA-256: `2e7c20eb2d1a40367c6568027ddaea9e6650b80bee7d57770d94a644772fd6ab`. `amendment_record.json` documents this pre-TEST-access correction; `amendment_integrity.json` records the exact amended/new files and preservation checks. The current manifest also seals the archived original files, including their original seal files.

`SHA256SUMS` hashes every file in this protocol directory except itself and its detached `SHA256SUMS.sha256`; the detached file hashes the manifest. This avoids self-reference. All JSON, this report, plans, evidence, audit records and generator/verifier sources are sealed. Regeneration changes the seal and must not be done after final access without a recorded amendment. Hash sealing is byte-integrity evidence, not an external timestamp/preregistration registry.

Authoritative frozen bindings include graph scientific artifact `baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237`, graph protocol `68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203`, trainer bundle `9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b`, environment lock `0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d`, and N20 dataset manifest `5f401224e65ab581af7d7558c65b64e149af863bfbe21023968906d3f1066ddd`. Source/file-content distinctions are preserved in binding/audit JSON.

Preservation verification compares 3764 pre-existing workspace files plus exact recorded upstream/cache/graph artifacts against the inherited snapshot, verifies prior result/protocol/environment seals, and checks all 120100 raw TRAIN files against the frozen 20-scenario manifest. The current-turn content check and final verification results are separate artifacts. Generic COGNIX, upstream calibration/features/recipes, the completed 15 runs, N20/cache, graph/protocol/trainer bundle/environment and frozen synthetic benchmark remain unchanged. The inherited snapshot predates this turn; current commands reverify it rather than overwrite it. Git HEAD/tracked diff and status outside this directory remain unchanged.

Absence-of-action statements describe this milestone's executed tools/scripts, not an OS-wide access trace. Hashes alone cannot prove that a file was never read or that no historical push ever happened. No claim of rerunning the old ML test suite is made; this design-only milestone uses integrity and static protocol consistency checks, with no model/scientific imports or conformal fitting.

# Ready / Not Ready for Official TEST Partition

**NOT READY to execute a partition.** The deterministic design is reviewable and sealed, but the historical file-level ledger remains incomplete despite the two recovered exact exclusions, authorized metadata counts/strata are unavailable, and linked-scenario provenance must be audited. Ratio adequacy and any group-level amendment must be sealed using metadata alone before payload access. No untouched membership is certified by the incomplete exposure record.

# Ready / Not Ready for Conformal Fitting

**NOT READY to fit conformal.** Resolve the exposure/provenance/count gates, publish the role manifest before any predictions, implement and verify the minimal adapter on synthetic scenario fixtures without altering generic COGNIX or the frozen scorer, lock official label/alignment semantics and inference portability, and separately authorize calibration-set-only access/fitting. Evaluation payloads/predictions/labels remain inaccessible during calibration. Sparse-stratum infinite cutoffs are valid but uninformative; retain them or record a metadata-only preregistration amendment before any final payload access, never optimize against set sizes.

# Recommended Next Step

Stop after resealing this protocol correction. Retain the two exact known exclusions; any later work must reconcile the historical file-level ledger without guessing additional IDs or reopening TEST. Official metadata/provenance/count auditing remains separately authorized and unexecuted. Scientific methods and all other design choices are unchanged. No TEST access, metadata inventory, partition, prediction, fitting, training, tuning, GNSS experiment, commit or push occurred in this amendment.

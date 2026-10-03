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

The **historical file-level ledger remains incomplete**. No evidence presently establishes any additional official TEST scenario IDs; none are inferred or claimed. The prior source documentation and synthetic fixtures are retained as contextual evidence, while `recovered_historical_evidence.json` records the authoritative recovered facts. `test_exposure_record.json` and `exposure_evidence.md` distinguish the two exposure extents and the remaining file-level limitation. Sealed metadata/count and public-provenance audits now establish publisher aggregates and unavailable family mapping. The exact complete catalogue and actual final membership remain unavailable; no partition has occurred.

# Proposed FINAL_CONFORMAL_CAL / FINAL_EVALUATION Split

**Amendment 002 — final pre-prediction partition/conformal amendment only.** The original rule stratified by **town × directory condition × anomaly type**, with **n_CAL,h=ceil(N_h/5)** and **Q=max_h Q_h**. At **alpha=.05**, finite Q_h requires at least **19** calibration scenarios, hence N_h≥91 under that allocation. Publisher totals establish **107 normal TEST scenarios across six towns**. At most one such stratum can reach 91; at least five necessarily fail finite rank. Consequently at least one Q_h=+inf, Q=+inf and all conformal sets are **{normal, anomaly}**. This is metadata/count mathematics alone, independent of model outcomes. The sealed public audit found no recoverable released-scenario → seed/route/base-drive/family mapping; scenario-N has no supported family semantics. Exact family allocation is unavailable from public information.

Replace quotas with **one uniform whole-scenario split over the complete eligible TEST catalogue**. Before randomization exclude the two exact historically exposed scenarios above, plus any future independently authenticated historical exposure documented and sealed before partition execution. Seal all exclusion decisions. Use the **lexicographically sorted complete eligible full scenario IDs**, **PCG64 seed 2028**, and **exactly one RNG permutation**. Set **n_cal=ceil(N_eligible/5)**. The first n_cal permuted IDs become **FINAL_CONFORMAL_CAL**; the remainder become **FINAL_EVALUATION**. All ticks/windows inherit the complete scenario's role. No stratification by town, directory condition or anomaly type; no redraws or balancing. Preserve ordered permutation and identical membership for all 15 frozen scorers.

The exact 627-scenario list remains unavailable publicly. `inventory_access_protocol.json` and `inventory_access_protocol.md` design a later separately authorized, bounded structure-only header traversal; no extractor is implemented or executed here. Sequential gzip traversal may carry payload bytes through decompression, discarded without interpretation or persistence; it is not zero payload-byte access. Output only unique scenario IDs and path-derived town/condition/type, plus a separate exact access ledger.

Before any later RNG creation require complete unique inventory, full traversal/integrity validation, exactly **107 normal and 520 anomaly (627 total) before exclusions**, both known excluded IDs found and removed, no duplicate full scenario IDs, every ID path-parseable, all exclusion decisions sealed, and the inventory hash sealed. Changed publisher counts or failed reconciliation mean **STOP** and a new amendment. Membership/complete payload alignment cannot be inferred from aggregate counts. If only the two known IDs are removed, 625/125/500 are conditional eligible/calibration/evaluation count arithmetic, not executed assignments. Seal role manifest, exclusions, ordered permutation, realized descriptive composition, RNG/version and protocol hashes before future scoring.

# Scenario / Temporal Exchangeability

The complete scenario is the indivisible assignment and calibration unit. Ticks and overlapping windows are dependent; a fixed maximum over all frozen eligible ticks needs no within-scenario independence or permutation symmetry.

The **registered primary interpretation is finite-catalogue marginal randomization/exchangeability**: conditional on the frozen scorer, complete eligible released TEST inventory, uniform scenario assignment, and exclusions fixed before randomization, calibration and a uniformly selected held-out scenario have symmetric score positions under the registered split. Arbitrarily dependent catalogue members can be treated as fixed units for this random-assignment claim. PCG64(2028) is the committed reproducible realization; the guarantee is over the registered assignment mechanism and held-out scenario selection, not conditional on that realized seed/partition or each calibration sample.

The sealed public-provenance audit recovered no released-scenario family/route mapping, and scenario-N has no supported family semantics. Related drives/routes or normal/anomaly variants may cross roles. **Family/route dependence remains an external-generalization limitation**, even though IDs are unavailable. No independent-family coverage, new-route population coverage, iid drive generation, arbitrary future CARLA coverage, live-distribution coverage, physical-safety certification, conditional coverage by town/type/condition, simultaneous coverage of every evaluation scenario, or joint coverage across all 15 scorers is claimed. Subgroup results are descriptive only. Hierarchical or population coverage would require additional assumptions and a separately sealed design.

# Conformal Construction

Retain all **15 frozen method/seed scorers** and **s_j,t(y)=1−P_j,t(y)**. For each FINAL_CONFORMAL_CAL scenario, **S_j=max_t s_j,t(Y_j,t)** over all eligible synchronized ticks under the unchanged feature/label protocol: t≥1 through the actual final tick; causal trailing window [max(0,t−11),t]; tick 0 excluded. Each scenario contributes exactly one score. No pseudo corruption, tick balancing, outcome filtering or probability recalibration.

Pool all n_cal scenario scores into **one global calibration set per scorer**. Augment with +infinity, set **k=ceil((n_cal+1)·0.95)** (exact integer form `(19*(n_cal+1)+19)//20`), and take the augmented kth order statistic **Q**. Inclusive comparison preserves ties; rank beyond n_cal gives +infinity. Define **C_j,t={y : 1−P_j,t(y)≤Q}**. Fit a separate global Q for each of the 15 scorers with **identical calibration membership** and **no prediction ensemble**. Remove per-town, per-condition and per-anomaly-type cutoffs, stratum-specific Q_h and Q=max_h Q_h. Finite global rank does not promise small sets; retain infinite or inefficient outcomes.

For the registered uniform split and a uniformly selected held-out catalogue scenario, conditional on the fixed eligible catalogue and scorer, the n_cal+1 score positions are symmetric. The augmented-rank argument gives **Pr_randomization(S_new≤Q)≥k/(n_cal+1)≥.95**, conservatively with ties. Since S_new is a maximum, **S_new≤Q implies simultaneous inclusion of all eligible true tick labels for that one scenario**. This is a marginal finite-catalogue statement under the conditions above; it does not cover every evaluation scenario jointly or all 15 scorers jointly. Family, route, future-drive, conditional-subgroup, live-distribution and safety nonclaims remain explicit in `conformal_protocol.json`.

Generic ConformalPredictor remains unchanged: its ordinary row fit has no scenario grouping. A future report-side adapter must reduce one block score per scenario, pool scores and store scenario membership, scores, n_cal, rank, alpha and the single global Q per scorer. Do not disguise block scores as probability rows or ordinary tick fitting. No adapter implementation or conformal fitting occurred in this milestone.

# Final Target Semantics

Development labels meant clean versus controlled TRAIN-derived pseudo corruption, under artificial paired prevalence. Final labels mean **official observation normal versus official observation anomaly**, read and aligned from the official timestep annotations in a later authorized stage. Directory-level normal/anomaly/type remains separate provenance; an anomaly directory can contain all-False tick labels. Missing annotations/alignment must halt evaluation; never assign all ticks positive from a directory name.

q remains frozen pseudo-trained **normality evidence**, and 1−q anomaly evidence. Its probability scale is not an established official-task posterior. Official-label conformal predicts candidate labels under the stated sampling assumptions; it does not recalibrate upstream probabilities or make pseudo calibration transfer automatically. Neither scores nor sets mean physical driving safety, a certified safe action or P(safe).

# Final Metrics

`final_metrics_spec.json` freezes formulas, class order, denominators and units. Retain scenario-macro **AUROC (primary), AUPRC as stepwise AP, F1, accuracy, balanced accuracy, Brier, BCE and 15-bin probability-reliability ECE**, plus pooled and per-scenario descriptive versions. Positive class is official anomaly; ranking score is 1−q. Hard predictions use each exact already selected development threshold, with ≥ as recorded. BCE clipping stays 1e−7; ECE edges are k/15 with a closed final endpoint and empty-bin zero contribution. No final threshold search.

Preserve the existing **strict undefined-value convention**: any undefined within-scenario metric makes its all-scenario macro null with explicit IDs/reasons. Normal-only scenarios can therefore make the primary strict macro AUROC and its contrast undefined. Do not relabel directory conditions as tick targets or silently remove these scenarios. Freeze one separately named **secondary two-class-eligible scenario-macro AUROC**, whose membership depends only on whether official labels contain both classes; report its denominator/excluded IDs, use the same subset for every method/seed and never promote it after results. Pooled AUROC also remains secondary.

Conformal diagnostics per scorer: simultaneous scenario coverage (all eligible tick labels covered); equal-scenario mean tick coverage; pooled tick coverage; signed coverage−.95 and nonnegative undercoverage gap; equal-scenario mean set size; abstention rate |C|≠1; separate empty/ambiguous-set rates; and counted descriptive subgroup diagnostics by town, anomaly type and directory condition where meaningful, with sparse results null or qualified; no subgroup-specific cutoff or guarantee. The scenario coverage metric matches the bound. Ticks and 5×J seed/scenario cells are never independent coverage trials. Undefined/unfitted results stay null; valid infinite cutoffs must report 100% coverage, size 2 and abstention 1 as an uninformative result.

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

After final pipeline freeze, export locally the immutable feature/agent/model source/schema, fitted one-class members, accepted probability mappings, all 15 selected checkpoints/thresholds, graph order/adjacency/dtypes/prior, full pooled conformal block scores/scenario membership/counts/ranks/global per-scorer cutoffs/alpha and descriptive subgroup metadata, exclusion and split manifests, target/decision protocol hashes, environment pins and golden replay evidence. Restore without fit; do not bundle raw TEST frames.

The dashboard must identify LIVE SIMULATION and show evidence, UQ, support/abstention/review and routing diagnostics with measured latency and coverage status. No control actuation or certified-safety claim is proposed. Infinite streams/new conditions have no registered scenario-exchangeability guarantee; show experimental status and require review rather than silently refresh calibration. The demonstration is not a substitute for held-out evaluation. `live_demo_plan.md` gives the export/compatibility plan; no live loop or export was implemented.

# Protocol Hashes

**Amendment 002:** complete pre-amendment protocol (including Amendment 001 history and both prior seal files) is preserved byte-for-byte in `amendment_history/pre_amendment_002/`. Old protocol manifest: `e03da4019c35d53420e42e02d4c0ebe2dd539a24ff5b645fdf5427078122ade1`. Public provenance manifest: `88242128bd2d4b3564e9c58ef12445b7c44c146f0a7f79b30e14371ed9700c50`. `amendment_002_record.json`, `amendment_002_changelog.md` and `amendment_002_integrity.json` define the current change and preservation checks. Earlier audit baselines refer to the archived pre-amendment bytes; do not rewrite the sealed audits.

**Amendment history:** the complete original 26-file directory is preserved byte-for-byte in `amendment_history/pre_amendment_001/`, including its 24-artifact manifest and detached hash. Previous manifest SHA-256: `2e7c20eb2d1a40367c6568027ddaea9e6650b80bee7d57770d94a644772fd6ab`. `amendment_record.json` documents this pre-TEST-access correction; `amendment_integrity.json` records the exact amended/new files and preservation checks. The current manifest also seals the archived original files, including their original seal files.

`SHA256SUMS` hashes every file in this protocol directory except itself and its detached `SHA256SUMS.sha256`; the detached file hashes the manifest. This avoids self-reference. All JSON, this report, plans, evidence, audit records and generator/verifier sources are sealed. Regeneration changes the seal and must not be done after final access without a recorded amendment. Hash sealing is byte-integrity evidence, not an external timestamp/preregistration registry.

Authoritative frozen bindings include graph scientific artifact `baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237`, graph protocol `68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203`, trainer bundle `9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b`, environment lock `0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d`, and N20 dataset manifest `5f401224e65ab581af7d7558c65b64e149af863bfbe21023968906d3f1066ddd`. Source/file-content distinctions are preserved in binding/audit JSON.

Historical preregistration preservation verification compared 3764 pre-existing workspace files plus exact recorded upstream/cache/graph artifacts against the inherited snapshot, verified prior result/protocol/environment seals, and checked all 120100 raw TRAIN files against the frozen 20-scenario manifest. Those historical content checks remain preserved as separate artifacts; they were not rerun for Amendment 002. Generic COGNIX, upstream calibration/features/recipes, the completed 15 runs, N20/cache, graph/protocol/trainer bundle/environment and frozen synthetic benchmark remain unchanged. The inherited snapshot predates this turn; its historical results are retained; Amendment 002 verifies only protocol/audit/source metadata bytes without reopening TRAIN or model outputs. Git HEAD remains unchanged; tracked diff and status outside this protocol directory remain unchanged. Changes inside this directory are listed in Amendment 002 integrity.

Absence-of-action statements describe this milestone's executed tools/scripts, not an OS-wide access trace. Hashes alone cannot prove that a file was never read or that no historical push ever happened. No claim of rerunning the old ML test suite is made; this design-only milestone uses integrity and static protocol consistency checks, with no model/scientific imports or conformal fitting.

# Ready / Not Ready for Official TEST Partition

**NOT READY to execute a partition.** Amendment 002 is design only. The exact complete authenticated scenario catalogue remains unavailable. Later separately authorized structure-only extraction must pass every inventory gate and seal all exclusions and inventory hashes before RNG creation. Both historical exclusions remain fixed; the incomplete historical file-level ledger remains disclosed, with no additional IDs inferred. Family IDs are unavailable; the registered guarantee is finite-catalogue only.

# Ready / Not Ready for Conformal Fitting

**NOT READY to fit conformal.** Obtain and seal the reconciled eligible inventory and role manifest in later authorized stages; implement and verify a minimal pooled scenario adapter without altering the frozen scorer or generic COGNIX; preserve official label/alignment semantics. Calibration-only payload access and fitting need separate authorization. Evaluation payloads/predictions/labels remain inaccessible during calibration. No score-based partition, construction, metric or threshold changes.

# Recommended Next Step

Stop after sealing Amendment 002. This milestone performed no TEST access, TEST scenario enumeration, sensor or label access, inventory execution, partition, inference, conformal fitting, training, tuning, commit or push. Upstream M1, graph architecture/checkpoints/readout/fusion/seeds, probability calibration, node features, pseudo recipes, threshold selection, target semantics, GNSS exclusion, frozen performance metrics (except necessary conformal stratification rewording), and generic COGNIX remain unchanged. Future inventory work requires separate authorization and must stop on failed reconciliation.

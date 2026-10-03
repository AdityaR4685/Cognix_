"""Write design artifacts from recorded development evidence only.

No dataset loader, ML library, inference, fitting, or TEST directory access.
The TEST partition algorithm below is prose; it is never executed here.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = "carla_final_evaluation_preregistration_v1"
SEEDS = [101, 202, 303, 404, 505]
METHODS = ["nograph", "standard_gat", "epistemic_gat"]


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def base():
    return {"protocol_version": VERSION, "status": "SEALED_DESIGN_ONLY_WITH_EXPLICIT_READINESS_GATES",
            "client_date": "2026-10-02", "client_timezone": "Asia/Calcutta",
            "TEST_payload_access_authorized": False, "executed": False}


def main():
    if (HERE / 'amendment_record.json').exists():
        raise RuntimeError('Amendment 001 is sealed; original generator cannot overwrite amended protocols. Use a documented amendment.')
    protocol = read("reports/carla_gat_preregistration_v1/protocol.json")
    splits = read("reports/carla_gat_preregistration_v1/split_manifest.json")
    per_seed = read("reports/carla_gat_paired_runs_v1/per_seed_metrics.json")["runs"]
    comparison = read("reports/carla_gat_paired_runs_v1/paired_comparison.json")
    ledger = read("reports/carla_gat_paired_runs_v1/run_ledger.json")
    assert ledger["completed_runs"] == 15 and ledger["complete"]
    bindings = []
    for r in ledger["runs"]:
        name = f"{r['method']}_{r['seed']}"
        root = ROOT / "reports/carla_gat_paired_raw_runs_v1" / name / r["method"] / f"seed_{r['seed']}"
        checkpoint = root / f"checkpoint_epoch_{r['selected_epoch']:03d}.pt"
        assert sha(checkpoint) == r["checkpoint_file_sha256"]
        bindings.append({"method": r["method"], "seed": r["seed"], "selected_epoch": r["selected_epoch"],
                         "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                         "file_sha256": r["checkpoint_file_sha256"], "content_sha256": r["checkpoint_content_sha256"],
                         "frozen_anomaly_threshold": per_seed[name]["threshold_selection"]["threshold"]})
    assert {(r['method'], r['seed']) for r in bindings} == {(m, s) for m in METHODS for s in SEEDS}
    write("frozen_method_bindings.json", {**base(), "methods": METHODS, "seeds": SEEDS, "runs": bindings,
        "graph_protocol_sha256": sha(ROOT / "reports/carla_gat_preregistration_v1/protocol.json"),
        "development_comparison": comparison["primary_summary"],
        "graph_scientific_sha256": "baa7deb4f20698d3ed681d0676a14208ae1e9f9a88c0542a1024d5b14d416237",
        "trainer_bundle_sha256": "9012174611be989f5305f5cbc40f05527831c9f704e49d8823c52fed88ccea3b",
        "environment_lock_sha256": "0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d",
        "development_findings": ["Both graph methods descriptively exceed NoGraph AUROC in all five seeds",
            "StandardGAT and EpistemicGAT are extremely close; no consistent epistemic superiority",
            "Canonical E is generally small; prior remains weak; preserve negative result"],
        "prohibited": ["best seed selection", "new graph experiment", "refit", "prior amplification",
                       "feature/recipe/architecture/metric tuning from completed outcomes"]})

    write("pipeline_map.json", {**base(), "source_files": {
        "M1": "cognix/adapters/carla/real_agents.py", "UQ": "cognix/adapters/carla/normality.py::ensemble_to_uncertainty",
        "M4_M2": "cognix/adapters/carla/graph_training_models.py", "M3": "cognix/calibration/conformal.py",
        "generic_orchestrator": "cognix/engine/pipeline.py"},
        "node_order": ["Camera", "IMU", "Seg"], "input_order": ["p_normal", "E", "A"],
        "equations": [
            "x_m,t = frozen feature extractor(raw observation and trailing window [max(0,t-11),t])",
            "u_m,k,t = sigmoid(a_m*s_m,k(x_m,t)+b_m), k=1..5, frozen one-class members and upstream calibrator",
            "p_m,t = mean_k u_m,k,t; A_m,t = mean_k H_B(u_m,k,t); E_m,t = max(0,H_B(p_m,t)-A_m,t)",
            "H_B(p) = -p log p -(1-p) log(1-p), production clipping unchanged",
            "h_m,t^(0) = [p_m,t,E_m,t,A_m,t], computed float64 then model input float32; raw E retained float64",
            "l_ij^(r) = LeakyReLU(a_r^T[W_r h_i^(r-1)||W_r h_j^(r-1)])",
            "w_j=1 for StandardGAT; w_j=1/(1+raw E_j) for EpistemicGAT, at both layers",
            "alpha_ij^(r) = exp(l_ij^(r))*w_j / sum_{k != i} exp(l_ik^(r))*w_k, j != i; alpha_ii=0",
            "h_i^(1)=ELU(sum_{j != i}alpha_ij^(1) W_1 h_j^(0)); z_i=sum_{j != i}alpha_ij^(2) W_2 h_j^(1)",
            "q_method,seed,t = (1/3) sum_i sigmoid(z_i,t); anomaly evidence score r=1-q",
            "NoGraph: z_i=Linear_12to1(ELU(Linear_3to12(h_i^(0)))); same mean of node sigmoids",
            "P_t(candidate y)=[q_t,1-q_t] in final protocol class order [normal=0,anomaly=1]",
            "M3: frozen candidate score 1-P_t(y), scenario-block cutoff; M5 consumes q,r,p/E/A,sets and attention diagnostics"],
        "inference": "eval mode; dropout disabled; frozen float32 execution and prior rounding, no AMP/TF32; no new scaling",
        "architecture": protocol["GAT_architecture"], "NoGraph": protocol["NoGraph"],
        "graph_readout_is_collective_fusion": True,
        "M4_M2_relationship": "M4 message passing plus its fixed mean(node_sigmoid) readout implements M2 collective fusion in this experiment",
        "extra_probability_fusion": None,
        "separate_fusion_justified_when": "Only when a graph implementation returns distinct unfused node beliefs; the fixed mean readout is that fusion here. No epistemic weighted re-fusion, logit averaging, post-graph probability calibration, or extra learned head is permitted.",
        "generic_engine_difference": "Generic pipeline refines node predictions then calls an independent belief fuser, and uses [1-fused,fused] at alpha=.05. Its generic semantics/order are not the adapter's final [normal,anomaly] contract. Do not route an already pooled q through that node-level path or change generic code.",
        "collective_UQ": "Input canonical E/A is retained per agent. H_B(q) is predictive entropy, not a justified decomposition into collective E/A. Seed disagreement is descriptive, not canonical epistemic UQ.",
        "implementation_status": "M1 and graph/readout exist; final official-target adapter, scenario M3 adapter and M5/dashboard wiring are future work; not an implemented end-to-end final evaluator"})

    dependencies = [
        {"set": "FIT_NORMAL", "scenarios": splits["sorted_FIT_ids"], "dependence": "Upstream one-class representations fitted on all 15 FIT scenarios; graph supervised models fitted on 12 GRAPH_TRAIN scenarios",
         "reuse_for_final_CP": "Violates ordinary split-conformal score symmetry with new scenarios because these observations fitted the scorer. Freezing it now does not remove that dependence."},
        {"set": "AGENT_CAL_ONLY", "scenarios": splits["AGENT_CAL_ONLY"], "dependence": "All five scenarios fitted upstream ensemble-predictive probability calibrators using clean and modality-matched pseudo targets",
         "reuse_for_final_CP": "Probability mapping is part of the scorer: this is training dependence even though graph optimization did not use CAL. No ordinary split-conformal guarantee on reuse."},
        {"set": "GRAPH_VALIDATION", "scenarios": splits["GRAPH_VALIDATION"], "dependence": "Upstream representations saw these FIT scenarios; graph early stopping/checkpoint selection and per-method/seed F1 thresholds used them",
         "reuse_for_final_CP": "Model selection dependence invalidates the naive independent split rank argument. Threshold reuse alone would not alter a CP score if the threshold did not enter it; here checkpoint selection already does."}]
    write("dependency_audit.json", {**base(), "existing_sets": dependencies,
        "conditioning": "Condition on all completed development data, upstream fits, selected checkpoints, fixed scoring functions, this protocol, and independent partition randomization. Reusing development for fitting does not invalidate future independent official-task CP; reusing it as CP calibration does.",
        "training_independence_vs_calibration_quality": "Poor transfer/probability calibration can widen sets but does not by itself invalidate conformal with independent, exchangeable official-task calibration/evaluation scores. Pseudo-target calibration cannot establish exchangeability with official labels.",
        "temporal": "Arbitrary dependence inside a scenario is allowed by the selected block maximum score. Tick count is never n_cal.",
        "scenario": "Whole scenarios/scores must be exchangeable within design strata, conditional on frozen development. Town alone is not a dependence block. Shared routes, simulator seeds, repeated drives or paired variant scenarios require a provenance audit; if scenarios are linked, halt and preregister larger independent groups before partition. No independence is inferred from different directory names.",
        "historical_schema_effect": "Prior TEST schema observations conditioned feature/loader design. New disjoint scenarios can still be evaluated conditional on that design if representative and exchangeable, but touched scenarios are not untouched and selection bias remains a limitation.",
        "guarantee_status": "Conditional mathematical proposition, not an established guarantee for CarlAnomaly or live CARLA"})

    exposure_sources = [
        {"path": "cognix/adapters/carla/carlanomaly_loader.py", "lines": [26, 35, 49],
         "record": "Base TEST probe verifies segmentation-front and GNSS/IMU/anomaly-observation schemas; labels documented all-False for change-weather/scenario-1; no town or complete member access ledger recorded"},
        {"path": "cognix/adapters/carla/real_features.py", "lines": [10],
         "record": "Base TEST byte-range schema probe for gnss.feather and imu.feather; no scenario IDs"},
        {"path": "tests/unit/test_carlanomaly_loader.py", "lines": [175, 192, 580],
         "record": "Synthetic temporary fixtures include Town01/change-weather/scenario-1 and other TEST-shaped IDs. These are not official payload-access logs and cannot prove the historical town or additional exposure."}]
    for s in exposure_sources:
        s["sha256"] = sha(ROOT / s["path"])
    import runpy
    exposure_amendment = runpy.run_path(str(HERE / 'amend_protocol.py'))
    write('test_exposure_record.json', exposure_amendment['corrected_exposure_record']())

    write("test_partition_protocol.json", {**base(), "seed": 2028,
        "seed_distinct_from": [0, 42, 101, 202, 303, 404, 505, 606, 2026, 2027],
        "unit": "whole complete scenario; all ticks/windows inherit its role",
        "candidate_fraction": {"FINAL_CONFORMAL_CAL": 0.2, "FINAL_EVALUATION": 0.8},
        "ratio_status": "20% is the frozen starting algorithm, conditional on count/provenance audit; no actual counts/IDs available. If inadequacy requires a changed ratio, seal a metadata-only amendment before any predictions or sensor payload access; never adjust after fitting.",
        "metadata_allowed_later": "Separately authorized official metadata-only inventory: split, town, scenario ID, directory normal/anomaly condition, anomaly type, complete-scenario inventory and any known family/run/route identity. No feather tables, images, masks, model predictions or metric inspection to allocate.",
        "scenario_identity": "test/normal/<town>/<scenario> or test/anomaly/<town>/<anomaly-type>/<scenario>; never use town/scenario alone",
        "stratum": "(town, official directory condition normal/anomaly, anomaly-type or NORMAL)",
        "composition": "Approximately 20% of each nonempty official metadata stratum; preserve type and town composition to whole-scenario rounding. Directory condition is not a tick label.",
        "algorithm_only": [
            "Require complete historical exclusion ledger, verified official metadata inventory, and scenario-family exchangeability audit; otherwise STOP without allocating",
            "Canonicalize UTF-8 POSIX IDs; reject duplicates, malformed IDs, non-TEST records and uncertain completion; never silently replace missing scenarios",
            "Remove all SCHEMA_EXPOSED_DIAGNOSTIC_ONLY scenarios and unresolved/linked exposures from both candidate roles",
            "If linked scenario families are present, STOP for a separately sealed group-level amendment; do not treat variants as separate independent scenarios",
            "Form sorted strata and lexicographically sorted complete IDs inside each stratum",
            "For each stratum h use n_cal,h = ceil(N_h/5), n_eval,h=N_h-n_cal,h; if N_h=1 it goes to CAL and stratum evaluation count is zero, explicitly report it",
            "Create ONE NumPy Generator(PCG64(2028)), with NumPy version recorded/pinned before execution; iterate strata lexicographically and permute their sorted IDs once in sequence",
            "First n_cal,h positions become FINAL_CONFORMAL_CAL; remaining positions FINAL_EVALUATION; sort output lists; no redraws, score-based stratification or rebalancing",
            "Publish/hash metadata inventory, exclusion supplement, ordered permutations, realized composition/counts, role manifest, RNG/version and protocol hashes before loading any final payloads or running inference",
            "The same role manifest applies to every method and seed; ticks/windows never cross sets; evaluation remains inaccessible during final conformal fitting"],
        "count_gate": {"alpha": 0.05, "minimum_calibration_scenarios_per_required_stratum_for_finite_cutoff": 19,
            "minimum_eligible_total_per_stratum_under_ceil_20pct": 91,
            "preferred_calibration_scenarios_per_stratum": 39,
            "preferred_total_per_stratum_under_ceil_20pct": 191,
            "explanation": "19 is algebraic nontriviality, not coverage precision/power. At n=19 the 95% cutoff is the maximum. n=39 allows rank 38/40 and resolution .025; this is a transparent design preference, not a sufficiency theorem.",
            "failure": "n_h<19 implies +infinity cutoff; selected global envelope becomes +infinity if any required stratum is sparse. Retain this outcome; do not pool strata or raise alpha to obtain smaller sets.",
            "evaluation_precision": "Counts also need enough independent evaluation scenarios; coverage resolution is 1/J. Neither .8N nor .2N proves useful power without metadata counts; no numeric scenario counts guessed."},
        "stratification_and_validity": "Stratified fixed quotas generally do not make pooled calibration/test scores exchangeable. Selected construction uses within-stratum rank calibration plus a conservative global envelope, not naive pooled CP.",
        "partition_performed": False, "counts_known": False, "readiness": False})

    write("conformal_protocol.json", {**base(), "construction": "Scenario-block maximum split conformal, stratum-specific cutoffs with a global conservative envelope",
        "alpha": 0.05, "coverage_target": 0.95, "methods": METHODS, "seeds": SEEDS,
        "class_order": ["official normal observation (0)", "official anomaly observation (1)"],
        "score": "s_method,seed,j,t(y)=1-P_method,seed,j,t(y); P(0)=q_normal, P(1)=1-q_normal",
        "scenario_score": "S_method,seed,j = max_{t in frozen eligible ticks of j} s_method,seed,j,t(Y_j,t)",
        "eligible_ticks": "All synchronized ticks t>=1 through the final tick, each exactly once; trailing window [max(0,t-11),t]; tick0 has no prediction by existing feature policy. Never require 3000 ticks when actual length differs. No pseudo corruption, resampling, balancing or outcome-based skips.",
        "calibration": "For each method/seed and each metadata stratum h, collect ONE S_j per FINAL_CONFORMAL_CAL scenario. Sort {S_1,...,S_n,+infinity}; k_h=ceil((n_h+1)*.95); Q_h=augmented k_h order statistic, inclusive <=. For n_h=0 use +infinity.",
        "numerics": "Implement .95 rank with exact integers k=ceil(19*(n+1)/20)=(19*(n+1)+19)//20; preserve ties. Serialized +infinity uses null plus quantile_is_infinite=true; no nonstandard JSON Infinity.",
        "inference_cutoff": "Q_method,seed=max_h Q_method,seed,h over all eligible metadata strata, including sparse strata. Never select a cutoff using the true test directory label/type or predicted anomaly type.",
        "prediction_set": "C_j,t = {y in {0,1}: 1-P_j,t(y)<=Q}; when Q=+infinity C={0,1} for every tick",
        "proof": [
            "Condition on the frozen scorer/development and stratum counts; assume calibration scenarios and a new scenario within h yield exchangeable S_j",
            "Whole-scenario maximum is a measurable fixed function, preserving between-scenario score exchangeability even when ordered ticks inside scenarios are arbitrarily dependent",
            "Augmented rank implies Pr(S_new<=Q_h | new stratum h, frozen development) >= .95, marginal over calibration and new scenario; ties are conservative",
            "Q>=Q_h implies Pr(for ALL eligible ticks t: Y_new,t in C_new,t | h, frozen development) >= .95",
            "Mixtures across the registered strata retain >=.95 by total probability. No within-scenario permutation symmetry is required"],
        "scope_limits": ["For each fixed method/seed separately; not simultaneous over 15 predictors or all future scenarios",
            "Not conditional on each realized calibration set, feature value, individual anomaly class or singleton selection",
            "Scenario exchangeability/provenance remains unverified; deterministic seed alone does not create exchangeability",
            "Random partition gives finite-population randomized coverage interpretation if uniform within strata and scores are fixed; does not prove generalization to newly generated drives",
            "Not arbitrary OOD/time-uniform live coverage; not a driving safety or action probability guarantee",
            "No scores can be used to alter alpha, partitions, features, thresholds, checkpoints or construction; inefficiency/trivial sets must be retained"],
        "generic_module_audit": {"module": "cognix/calibration/conformal.py::ConformalPredictor",
            "fit": "1-cal_outputs[row,cal_labels[row]]; sorted row scores; no groups or input/group validation",
            "predict": "ceil((n+1)*(1-alpha)) augmented +infinity order statistic; inclusive class score comparison",
            "exchangeability_unit": "one input row; under ordinary use this is one example/tick",
            "actual_guarantee": "Finite-sample marginal class coverage if the fitted scorer is independent of calibration and future scores are exchangeable; neither per-class nor temporal/scenario simultaneous coverage is implemented",
            "cluster_handling": "No group IDs, block reduction, family provenance, hierarchical weights or cluster guarantee",
            "TwoStageConformalPredictor": "Fits separate individual and collective predictors only; no joint or hierarchical validity, no error budget allocation",
            "extension_decision": "A minimal CARLA/report-side block-score adapter suffices mathematically. It groups scores, computes exact ranks per stratum and global envelope, and applies the same inclusive score rule. Do not call ordinary tick-level fit and label it scenario CP. Generic module remains unchanged.",
            "reuse_boundary": "Generic prediction rule can be reused after a transparent adapter imports grouped score state with n_cal counting scenarios, but its generic save format loses group/envelope metadata. Prefer separately sealed adapter state; do not manufacture pseudo probability rows to disguise block scores.",
            "generic_extension_needed_for": "A reusable domain-neutral clustered/hierarchical API would require new generic score/group fitting and serialization contracts; out of scope and not necessary for this adapter design."},
        "alternatives": [
            {"name": "tick pooling", "decision": "Rejected: temporal dependence and variable scenario size do not justify n_cal=tick count"},
            {"name": "hierarchical conformal", "decision": "Not selected: standard hierarchical group/within-group permutation assumptions are not established for ordered CARLA ticks; genuine implementation/assumption work would be required"},
            {"name": "scenario scalar aggregation", "decision": "Mathematically compatible with generic CP after one fixed score per scenario, but predicts a scenario target and loses observation label localization. Mean/max r is not automatically a calibrated scenario posterior; directory anomaly condition and any-positive-tick target differ."},
            {"name": "random one-tick-per-scenario", "decision": "Can give marginal coverage for a randomly sampled tick in a new scenario under suitable sampling assumptions, but not all ticks; unnecessary target change here"},
            {"name": "block maximum", "decision": "Selected for transparent whole-scenario guarantee, arbitrary temporal dependence, matched tick target, and adapter-only implementation; not selected for narrow sets"}],
        "target": {"development": "clean vs controlled TRAIN-derived pseudo corruption at artificial paired prevalence",
            "final": "official per-timestep anomaly-observation label, mapped normal=0/anomaly=1; directory normal/anomaly and anomaly type retained as metadata, not substituted for labels",
            "semantics": "q stays a frozen pseudo-trained normality evidence score. Its complement is anomaly evidence; neither is an established official-task posterior. Conformal supports candidate official labels; it does not recalibrate q or validate a safe action.",
            "labels": "Read/alignment audit only in separately authorized future stage. No blanket labels from anomaly directory; a historical change-weather scenario has all-False observation labels. Missing/misaligned labels halt evaluation; never fabricate or zero-fill."},
        "sources": [{"url": "https://arxiv.org/abs/2107.07511v6", "use": "Split-conformal augmented rank and exchangeability basis; block/envelope proof above is this protocol's direct derivation"},
                    {"url": "https://arxiv.org/abs/2306.06342v4", "use": "Hierarchical conformal requires a separately defined symmetry/sampling target; not assumed for ordered scenario ticks"}],
        "fitted": False, "adapter_implemented": False, "ready_for_fitting": False})

    metrics = {"AUROC": "Pr(r_anomaly>r_normal)+.5 Pr(tie), within each scenario; requires both tick classes",
        "AUPRC": "Stepwise average precision by descending tied anomaly score groups, not PR trapezoids; null if no positive ticks",
        "F1": "2TP/(2TP+FP+FN); frozen convention 0 if denominator=0",
        "accuracy": "(TP+TN)/n", "balanced_accuracy": ".5*(TP/(TP+FN)+TN/(TN+FP)); null when either true class absent",
        "Brier": "mean((q-(1-Y_anomaly))^2)",
        "BCE": "mean(-(1-Y)*log(clip(q,1e-7,1-1e-7))-Y*log(clip(1-q,1e-7,1-1e-7)))",
        "ECE": "15 equal probability bins with edges k/15: [lower,upper), final includes 1; sum_b n_b/n*abs(mean(q_b)-mean(1-Y_b)); empty bins contribute 0; binary reliability, not max-confidence ECE"}
    write("final_metrics_spec.json", {**base(), "target_and_unit": "Official observation labels and one score per eligible tick; equal scenario weights for macro metrics; pooled metrics explicitly tick weighted",
        "primary_endpoint": "Strict scenario-macro AUROC over ALL FINAL_EVALUATION scenarios; primary contrast EpistemicGAT minus StandardGAT per seed then mean of all five pairs",
        "undefined_primary_policy": "Preserve the development strict-null convention: if any evaluation scenario's AUROC is undefined, strict macro and primary contrast are null with exact reasons/denominators. Official normal scenarios commonly make this unavoidable. Do not silently discard them or replace the primary endpoint.",
        "preregistered_descriptive_alternative": "Secondary two-class-eligible scenario-macro AUROC over the explicit label-defined subset with both tick classes. Freeze this subset rule now, retain excluded IDs/reasons/counts, same subset across all methods/seeds; null if empty. Never promote it to primary after TEST.",
        "binary_metrics": metrics, "aggregation": "Compute per-scenario metrics first, unweighted arithmetic macro; any undefined per-scenario value makes strict macro null; retain pooled metrics and all per-scenario evidence. Explicit eligible AUROC subset is the sole named alternative; no undeclared partial macros.",
        "hard_labels": "Y_hat_anomaly=1 iff r>=the exact development GRAPH_VALIDATION threshold for that method/seed in frozen_method_bindings.json; no final threshold search; thresholds retain pseudo-trained semantics and transfer may be poor",
        "conformal_metrics": {
            "scenario_simultaneous_coverage": "J^-1 sum_j 1[ALL eligible ticks in j have Y_j,t in C_j,t], primary CP diagnostic matching the guarantee",
            "scenario_macro_tick_coverage": "J^-1 sum_j (1/n_j) sum_t 1[Y_j,t in C_j,t]",
            "pooled_tick_coverage": "sum_j,t covered / sum_j n_j; descriptive, not independent tick coverage trials",
            "signed_coverage_gap": "empirical coverage minus .95, separately for simultaneous scenario, scenario-macro tick and pooled tick coverage",
            "undercoverage_gap": "max(0,.95-empirical coverage), same units",
            "prediction_set_size": "per-scenario mean |C| then equal-scenario mean; also pooled descriptive mean",
            "abstention_rate": "per-scenario mean 1[|C|!=1] then equal-scenario mean; empty and two-label sets separately retained",
            "empty_and_ambiguous_rate": "equal-scenario fractions of |C|=0 and |C|=2",
            "per_stratum_coverage": "Same simultaneous coverage/size diagnostics with counts; descriptive, no new class-conditional claim",
            "missing_calibration": "null plus reason; do not substitute uncalibrated sets; infinite valid cutoff produces full sets and 100% coverage, size2, abstention1 and must be reported as trivial"},
        "scenario_condition_target": "No new directory-condition classifier or scenario detection AUROC is primary. Directory strata only define partition/provenance and coverage diagnostics.",
        "reporting": "15 method/seed results and paired contrasts retained; secondary NoGraph comparisons, official-type/town diagnostics predeclared; no superiority selection across metrics/strata",
        "uncertainty": "Do not treat ticks or 5*J seed-scenario cells as independent. Seed t/sign-flip summaries are conditional initialization summaries. Optional descriptive whole-scenario bootstrap is frozen in seed_aggregation_spec.json; no independent-population confidence claim absent scenario sampling assumptions.",
        "failure": "Retain all assigned scenarios and seeds in ledger. Missing modalities, nonfinite data/scores or unreadable labels mark protocol incomplete; no replacement or dropping to improve results; tick0 exclusion is the fixed feature contract, not failure-based filtering."})

    write("seed_aggregation_spec.json", {**base(), "seeds": SEEDS, "methods": METHODS,
        "primary": "Five paired method/seed results separately; arithmetic mean of the five EpistemicGAT-StandardGAT primary endpoint differences",
        "ensemble_used": False, "reason": "Match completed paired-seed design, retain initialization variability and avoid introducing a new scoring distribution requiring separate ensemble calibration",
        "prediction_ensemble_rule": None, "no_best_seed": True,
        "conformal": "15 independent frozen scoring functions calibrated separately on the SAME role manifest; 5-member upstream bootstrap agent is unchanged and distinct from graph training seeds",
        "primary_summary": "all individual differences, mean, median, sample SD ddof1, min/max; null/incomplete if any planned pair or endpoint undefined",
        "conditional_seed_interval": "mean(delta) +/- 2.776445105*SD(delta)/sqrt(5); initialization variability conditional on fixed final evaluation data, not scenario population uncertainty",
        "exact_sign_flip": "All 32 sign patterns, two-sided fraction abs(mean(sign*delta))>=abs(mean(delta))-1e-12; symmetry assumption explicit, minimum p=.0625; no .05 superiority inference possible with five pairs",
        "coverage_reporting": "Per method/seed CP metrics, then descriptive arithmetic mean over seeds. Shared calibration/evaluation creates dependence. No .95 joint coverage across seeds, no coverage claim for averaging/union/intersection/consensus of their sets.",
        "descriptive_scenario_bootstrap": {"draws": 10000, "seed": 606, "rng": "PCG64", "unit": "whole evaluation scenario",
            "algorithm": "Use common resampled IDs for every method/seed. For defined equal-scenario endpoints compute mean over resampled scenarios, then paired-seed average; percentile2.5/97.5. Resample inside frozen metadata strata retaining original stratum evaluation counts, weight scenario means by realized evaluation stratum counts/J. For secondary eligible AUROC use its explicitly reported two-class subset and fixed counts. Undefined endpoints stay null.",
            "interpretation": "descriptive scenario fragility only; no tick bootstrap, no reselecting thresholds, no calibrator refits, no replacement seeds"}})

    write("decision_semantics.json", {**base(), "labels": ["normality evidence", "anomaly evidence", "uncertainty diagnostics", "conformal support", "abstention", "review request"],
        "forbidden_interpretations": ["certified driving safety", "P(safe action)", "physical risk bound", "causal anomaly attribution"],
        "inputs": "15 q/r scores, input modality p/E/A, candidate sets and provenance/status; attention and node sigmoid readout diagnostics when exposed faithfully",
        "per_predictor_logic": {"invalid_or_missing_input_or_unfitted_conformal": "abstain / require review; no numerical fallback",
            "C_empty": "abstain / inconsistent conformal support / require review",
            "C_normal_and_anomaly": "abstain / ambiguous support / request more observations or review",
            "C_anomaly_only": "official anomaly-label support / require review",
            "C_normal_only": "official normal-label support; report as evidence only, no vehicle-control authorization",
            "hard_threshold_disagrees_with_singleton": "flag disagreement / abstain operationally; retain both raw outputs"},
        "multi_seed_method_logic": "Display every frozen result. A common normal singleton with consistent hard labels across all 15 is a candidate normal-evidence summary. Any anomaly support, ambiguity, failure or disagreement requests review. This operational agreement rule is not a new conformal set and has no selected-case .95 correctness claim.",
        "threshold_selection": "Use the 15 recorded development thresholds for descriptive hard classifications. No new numeric E, score, set-size or risk threshold is tuned. Set cardinality rule is structural. Additional operational thresholds would require separate development data/protocol excluding FINAL_CONFORMAL_CAL and FINAL_EVALUATION; not authorized here.",
        "attribution": {"available": "Per-modality p/E/A; two-layer incoming attention alpha and final node probabilities v_i=sigmoid(z_i). q=sum_i v_i/3; v_i/3 are exact readout summands, not independent raw-sensor influence because each graph node already mixes senders.",
            "meaning": "Attention is routing, not causal importance; readout summands are bookkeeping. No Shapley sign/importance score is invented.",
            "generic_attribution_limit": "Generic EpistemicShapley uses its own coalition belief evaluation, which is not automatically the frozen learned graph/readout. Do not label those values explanations of q without a separate frozen graph-consistent coalition game and baseline/missing-node semantics.",
            "future": "Adapter may expose existing attention/node logits without changing q, checked against frozen forward results before TEST. A new intervention or Shapley baseline would need a separately preregistered attribution study; no TEST tuning."},
        "GNSS": "Excluded from primary p/E/A and decisions: structurally invalid calibrated probability/UQ remains null; raw-score auxiliary study deferred to future work"})

    (HERE / "live_demo_plan.md").write_text("""# Live CARLA Demonstration Plan

Scientific evaluation and demonstration are separate deliverables. The demo is not an official held-out benchmark and cannot establish coverage under live distribution shift.

Live front RGB / semantic segmentation / IMU -> frozen deterministic compact feature extraction (Camera18, Seg29, IMU10) -> the five existing bootstrap one-class members and frozen probability mappings per primary agent -> canonical p/E/A -> each of the 15 fixed graph/NoGraph checkpoints -> existing mean(node sigmoid) collective readout -> separately exported scenario conformal state -> conservative evidence/abstention/review logic -> dashboard. GNSS is excluded. Do not substitute pretrained embeddings or a newly trained model.

The IMU buffer contains the trailing 12 ticks, resets at the declared start of each finite scenario, and emits from tick1 after at least two samples. RGB and segmentation tick identifiers must align; use the frozen segmentation palette/channel semantics and IMU column semantics. A live CARLA adapter must explicitly validate coordinate fields, sensor synchronization and compatibility; different simulator sensor semantics cannot be silently coerced. Do not infer a sample rate from dataset row counts.

Export locally after final freeze: immutable extractor/agent/model source and feature schema, one-class fitted member parameters, frozen accepted upstream calibrators and validity audit, node order/dtypes/adjacency/prior, all 15 ledger-bound selected checkpoints and their content/file hashes, all 15 development hard thresholds, conformal scenario scores/counts/strata/alpha/cutoffs/global envelopes and infinite-cutoff status, final partition/exclusion/target manifests and protocol seals, decision semantics, pinned inference environment, and row-keyed golden replay evidence. Artifact restoration must reproduce q without any fit/refit. Do not export raw TEST frames to a demo bundle.

Keep each seed/method identifiable; no best-seed selection or unregistered prediction ensemble. Dashboard declares LIVE SIMULATION and scientific coverage status, shows normality/anomaly evidence, per-agent UQ, support sets, failure/abstention and review flags, and labels attention as routing. It never claims a safe action probability. Display real recorded inference/feature/transport latency when available; no real-time claim before measurement. No control actuation is proposed.

The conservative conformal envelope requires no oracle official normal/anomaly/type metadata at inference. Its mathematical guarantee still assumes a finite scenario generated within registered exchangeable conditions. An indefinite live stream or a new town/type is unsupported: show experimental sets with guarantee unavailable and request review. No automatic online adaptation, threshold optimization, conformal refresh or calibration accumulation. Validate separately on simulator-generated development scenarios, preserving official final sets.

This milestone exports no checkpoints, fits no conformal state, builds no dashboard, and implements no live loop.
""", encoding="utf-8")

    (HERE / 'exposure_evidence.md').write_text(exposure_amendment['exposure_markdown'](), encoding='utf-8')

    # Inventory of already-read local evidence, without any payload paths.
    evidence_paths = ["reports/carla_gat_preregistration_v1/protocol.json", "reports/carla_gat_preregistration_v1/split_manifest.json",
        "reports/carla_gat_preregistration_v1/metrics_spec.json", "reports/carla_gat_paired_execution_v1/delivery_report.md",
        "reports/carla_gat_paired_runs_v1/run_ledger.json", "reports/carla_gat_paired_runs_v1/per_seed_metrics.json",
        "reports/carla_gat_paired_runs_v1/paired_comparison.json", "cognix/calibration/conformal.py",
        "cognix/engine/pipeline.py", "cognix/adapters/carla/graph_training_models.py",
        "cognix/adapters/carla/normality.py", "cognix/adapters/carla/real_agents.py",
        "cognix/adapters/carla/real_features.py", "cognix/adapters/carla/cache_builder.py"]
    write("source_evidence.json", {"local_files": {p: sha(ROOT / p) for p in evidence_paths},
        "literature": ["https://arxiv.org/abs/2107.07511v6", "https://arxiv.org/abs/2306.06342v4"],
        "no_official_TEST_web_or_payload_request": True})
    print(json.dumps({"protocol_files_written": 12, "frozen_runs_bound": len(bindings), "partition_executed": False, "conformal_fitted": False}))


if __name__ == "__main__":
    main()

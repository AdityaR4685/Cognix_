"""Apply Amendment 002 once, using local sealed protocol/audit text only.

No archive reader, inventory extractor, RNG, scientific import, fitting or
inference is implemented here. Run the separate metadata-only verifier to seal.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE / "amendment_history/pre_amendment_002"
OLD_HASH = "e03da4019c35d53420e42e02d4c0ebe2dd539a24ff5b645fdf5427078122ade1"
META_HASH = "888a63a0f0bb190eb899cfe900674bea6804b354017993b45c235b41b1c26839"
PROV_HASH = "88242128bd2d4b3564e9c58ef12445b7c44c146f0a7f79b30e14371ed9700c50"
ID = "002_pre_prediction_uniform_catalogue_pooled_scenario_conformal"
EXCLUSIONS = ["test/anomaly/Town01/change-weather/scenario-1",
              "test/anomaly/Town01/change-weather/scenario-10"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")


def verify_manifest(directory, expected):
    manifest = directory / "SHA256SUMS"
    assert sha(manifest) == expected, directory
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        path = (directory / rel).resolve()
        assert path.is_relative_to(directory.resolve()) and rel not in seen
        assert sha(path) == digest, path
        seen.add(rel)
    return seen


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def main():
    assert not OLD.exists(), "Already archived; do not rerun or overwrite history"
    assert not (HERE / "amendment_002_record.json").exists()
    prior = verify_manifest(HERE, OLD_HASH)
    verify_manifest(ROOT / "reports/carla_test_metadata_audit_v1", META_HASH)
    verify_manifest(ROOT / "reports/carla_public_provenance_audit_v1", PROV_HASH)
    assert (HERE / "SHA256SUMS.sha256").read_text().split()[0] == OLD_HASH
    prior.update(["SHA256SUMS", "SHA256SUMS.sha256"])
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()}
    assert actual == prior | {Path(__file__).name}, "Unexpected protocol files"
    prior_hashes = {rel: sha(HERE / rel) for rel in sorted(prior)}
    # Fixed source/config scope; no reports' checkpoints, predictions or raw data.
    source_paths = [p for area in ("cognix", "tests", "configs")
                    for p in (ROOT / area).rglob("*") if p.is_file()
                    and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
                    and "__pycache__" not in p.parts]
    source_paths += [ROOT / name for name in
                     ("pyproject.toml", "README.md", "CHANGELOG.md", ".gitignore")]
    baseline = {
        "old_protocol_manifest_sha256": OLD_HASH,
        "metadata_audit_manifest_sha256": META_HASH,
        "public_provenance_manifest_sha256": PROV_HASH,
        "prior_protocol_files": prior_hashes,
        "protected_source_config_files": {p.relative_to(ROOT).as_posix(): sha(p)
                                          for p in sorted(source_paths)},
        "git_head": git("rev-parse", "HEAD").strip(),
        "git_tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary", "--", ".", ":(exclude)reports/carla_final_evaluation_preregistration_v1/**").encode()).hexdigest(),
        "git_tracked_diff_scope": "Outside amended protocol directory only; prior full tracked diff was empty at capture",
        "git_status_outside_protocol": [line for line in git("status", "--porcelain=v1").splitlines()
                                        if "reports/carla_final_evaluation_preregistration_v1/" not in line],
        "scope": "Protocol/audit text and source/config bytes only. No payload/model-output rehash; no full untracked-content checksum."
    }
    for rel in sorted(prior):
        target = OLD / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / rel, target)
        assert sha(target) == prior_hashes[rel]
    write("amendment_002_initial_state.json", baseline)

    partition = read(OLD / "test_partition_protocol.json")
    for key in ("stratum", "stratification_and_validity", "count_gate", "ratio_status"):
        partition.pop(key)
    partition.update({
        "amendment_id": ID,
        "status": "SEALED_AMENDMENT_002_DESIGN_ONLY",
        "counts_known": False,
        "published_aggregate_counts_known": True,
        "complete_eligible_inventory_known": False,
        "composition": "Uniform whole-scenario allocation; town, directory condition and anomaly type are descriptive metadata only. No composition quotas or balancing.",
        "algorithm_only": [
            "Later authorized structure-only inventory must finish all gates in inventory_access_protocol.json. STOP on any failed reconciliation. No archive access is authorized by this amendment.",
            "Derive complete UTF-8 POSIX full scenario IDs from paths only; retain town, directory condition and anomaly type (NORMAL for normal). Reject malformed IDs, duplicate full inventory IDs, non-TEST entries, incomplete traversal and uncertain completion.",
            "Before randomization remove both exact historical exclusions in full and any future independently authenticated historical exposure documented and sealed before partition execution. Resolve all exclusion decisions; do not infer families from scenario-N.",
            "Seal complete source inventory, its hash, all exclusion decisions and the eligible inventory hash before creating an RNG. Pin and record the NumPy version and the UTF-8 bytewise lexicographic ID ordering.",
            "Use the single lexicographically sorted complete eligible ID list as input. Let N_eligible be its length and n_cal=ceil(N_eligible/5)=(N_eligible+4)//5.",
            "Create exactly one NumPy Generator(PCG64(2028)); call permutation exactly once on the complete eligible list (or its ordered integer indices). No other RNG draws, stratum loops, quota assignments, redraws or balancing.",
            "First n_cal permuted IDs become FINAL_CONFORMAL_CAL; remainder become FINAL_EVALUATION. Preserve ordered permutation and assignment positions; sorted role views are display-only.",
            "Seal role manifest, ordered permutation, source/eligible hashes, exclusions, RNG/version/protocol hashes and realized descriptive composition before loading any payloads for scoring. Same complete-scenario membership for all 15 scorers; all ticks/windows inherit the scenario role.",
            "Evaluation payloads/predictions/labels remain inaccessible during calibration fitting. Failed later label/alignment/feature checks make the evaluation incomplete; never replace assigned scenarios or redraw."
        ],
        "allocation": {"rng": "PCG64", "seed": 2028, "permutation_calls": 1,
                       "input": "lexicographically sorted complete eligible full scenario IDs",
                       "n_cal": "ceil(N_eligible/5)", "stratify_assignment": False,
                       "redraws_or_balancing": False},
        "historical_exclusion_supplement": "Any future independently authenticated exposure must be documented and its whole-scenario exclusion sealed before partition execution; no outcome-based exclusions.",
        "inventory_gates_reference": "inventory_access_protocol.json",
        "conditional_count_arithmetic_only": {"before_exclusions": 627,
            "normal": 107, "anomaly": 520, "after_only_two_known_exclusions": 625,
            "n_cal_if_625_eligible": 125, "n_evaluation_if_625_eligible": 500,
            "qualification": "Publisher totals and two known IDs only; no enumerated inventory, partition or certified eligible count. Additional authenticated historical exclusions would change these arithmetic counts."},
        "metadata_allowed_later": "Structure-only complete archive traversal under separate authorization: member header paths and path-derived town/condition/type only; inventory_access_protocol.json defines bounds and byte-transit disclosure.",
        "family_provenance": "Sealed public audit found no recoverable released-scenario to seed/route/base-drive/family mapping; scenario-N has no supported family semantics. Exact family allocation unavailable. Family/route dependence limits external generalization, not finite-catalogue random assignment.",
        "randomization_interpretation": "Registered assignment mechanism is uniform over whole-scenario permutations; PCG64(2028) fixes reproducible implementation. Probability statements are over the registered randomization and a uniformly selected held-out scenario, not conditional on the realized seed/partition/calibration scores; a fixed seed alone proves no population exchangeability."
    })
    write("test_partition_protocol.json", partition)

    cp = read(OLD / "conformal_protocol.json")
    cp.update({
        "amendment_id": ID, "status": "SEALED_AMENDMENT_002_DESIGN_ONLY",
        "construction": "Pooled scenario-block maximum split conformal with one global cutoff per frozen method/seed",
        "calibration": "For each fixed method/seed, collect exactly ONE S_j=max_t(1-P_j,t(Y_j,t)) from each FINAL_CONFORMAL_CAL scenario over all frozen eligible synchronized ticks. Pool all n_cal scenario scores globally; sort {S_1,...,S_n_cal,+infinity}; k=ceil((n_cal+1)*0.95); Q is the augmented kth order statistic. Inclusive <= retains ties. For n_cal=0 or k>n_cal use +infinity. No category-specific cutoffs or envelope.",
        "inference_cutoff": "One global Q_method,seed; fit separately for all 15 frozen scorers using identical FINAL_CONFORMAL_CAL membership. No prediction ensemble or category-dependent selector.",
        "numerics": "Exact integer rank k=(19*(n_cal+1)+19)//20; +infinity augmentation and inclusive comparison. Serialize +infinity as null with quantile_is_infinite=true; no nonstandard JSON Infinity.",
        "primary_guarantee": "Marginal scenario-level randomization/exchangeability coverage >=0.95 for a held-out scenario chosen uniformly from the remainder of the frozen eligible released TEST catalogue under the registered uniform whole-scenario split, conditional on frozen scorer, complete eligible inventory, uniform scenario assignment and exclusion rules fixed before randomization. S_new<=Q implies simultaneous inclusion of all eligible true tick labels for that scenario.",
        "proof": [
            "Condition on the frozen scorer, complete fixed eligible released catalogue and pre-randomization exclusions; each scenario's hypothetical true-label maximum score is fixed. No scores are computed in this milestone.",
            "Uniformly assign n_cal scenarios without replacement, then select a held-out scenario uniformly from the remainder. Conditional on their union, calibration/new positions are symmetric; the n_cal+1 score positions are exchangeable under this assignment even if catalogue scenarios share routes or families.",
            "The augmented rank k=ceil((n_cal+1)*.95) gives Pr_randomization(S_new<=Q)>=k/(n_cal+1)>=.95, with conservative ties; if k=n_cal+1, Q=+infinity and coverage is trivial.",
            "Because S_new=max_t(1-P_new,t(Y_new,t)), S_new<=Q entails Y_new,t in C_new,t for ALL frozen eligible ticks of that one scenario; arbitrary temporal dependence is allowed.",
            "This is a marginal statement over registered assignment and held-out scenario selection. It is not conditional on the realized calibration sample, cutoff, seed, partition, town/type, features or singleton selection; no new-route or future-drive sampling model is established."
        ],
        "scope_limits": [
            "No independent-family coverage, new-route population coverage or iid drive-generation claim.",
            "No arbitrary future CARLA coverage, live-distribution coverage or physical-safety certification.",
            "No conditional coverage by town, directory condition or anomaly type; those are descriptive diagnostics only.",
            "No simultaneous coverage of every evaluation scenario and no joint coverage across all 15 scorers.",
            "Family/route dependence remains an external-generalization limitation despite unavailable family IDs.",
            "No guarantee conditional on each realized calibration sample, partition, feature value or singleton selection; a reproducibility seed is not a population-exchangeability proof.",
            "Never use outcomes to change alpha, partition, exclusion rules, scorer, features, probabilities, thresholds, checkpoints or construction; retain inefficient/trivial sets."
        ],
        "removed_construction": ["per-town cutoffs", "per-condition cutoffs", "per-anomaly-type cutoffs", "stratum-specific Q_h", "Q=max_h Q_h"],
        "global_cutoffs": 15,
        "prediction_ensemble": False,
        "identical_calibration_membership": True,
        "sources_note": "Prior sealed references preserved; finite-catalogue random-assignment proof is the direct rank derivation above, not a claim that publisher drive generation is iid."
    })
    cp["generic_module_audit"]["extension_decision"] = "Future minimal CARLA/report-side block-score adapter groups each scenario once, pools global scenario scores and computes one exact augmented rank per method/seed. Do not use ordinary tick-level fit as scenario CP. Generic module remains unchanged."
    cp["generic_module_audit"]["reuse_boundary"] = "Generic inclusive prediction rule may be reused with explicit grouped score state; separately seal scenario membership, pooled scores, n_cal, rank, alpha and global Q per scorer. Never manufacture pseudo probability rows to disguise block scores."
    cp["sources"][0]["use"] = "Existing sealed split-conformal augmented-rank reference; pooled scenario and finite-catalogue random-assignment proof is this protocol's direct derivation."
    write("conformal_protocol.json", cp)

    metrics = read(OLD / "final_metrics_spec.json")
    metrics["conformal_metrics"]["per_subgroup_coverage"] = metrics["conformal_metrics"].pop("per_stratum_coverage").replace("descriptive, no new class-conditional claim", "descriptive only by town, anomaly type and directory condition where meaningful; no subgroup-specific cutoff or coverage guarantee; sparse/undefined results null or qualified with counts/reasons")
    metrics["scenario_condition_target"] = "No new directory-condition classifier or scenario detection AUROC is primary. Town/type/directory condition remain inventory and evaluation-ledger metadata for descriptive subgroup reporting only; never define assignment quotas or conformal cutoffs."
    write("final_metrics_spec.json", metrics)
    dependency = read(OLD / "dependency_audit.json")
    dependency["scenario"] = "Amendment 002 primary guarantee is marginal finite-catalogue randomization/exchangeability under uniform whole-scenario assignment, conditional on frozen scorer, complete eligible inventory and exclusions fixed before randomization. Public audit found no released-scenario family mapping; scenario-N has no supported family semantics. Dependence may cross roles and remains an external-generalization limitation. No independent-family, iid-drive or new-route claim."
    dependency["guarantee_status"] = "Registered conditional finite-catalogue randomization proposition only; no established future CARLA/live-distribution guarantee."
    write("dependency_audit.json", dependency)

    inventory = {
        "amendment_id": ID, "status": "DESIGN_ONLY_NOT_IMPLEMENTED_OR_EXECUTED",
        "authorization": "Later separate authorization required before any TEST archive request, acquisition or traversal. This design authorizes none.",
        "public_inventory": "Exact complete 627-scenario list unavailable publicly; no standalone public Base TEST manifest recovered in sealed audits.",
        "input": "Official Base TEST gzip-compressed tar, with publisher release, exact URL, advertised checksum/size and access limits sealed before opening. No supplementary sensor archives, sample archives or local TEST directory enumeration.",
        "byte_transit": "Reaching later tar headers may require sequential gzip decompression. Payload bytes may transit the decompressor in bounded volatile buffers but are discarded without interpretation or persistence. This is archive byte access, not zero payload-byte reads; semantic sensor/label inspection remains forbidden.",
        "mechanical_bounds": {"compressed_bytes_max": "Exact advertised archive byte size from authenticated public release record; absent size is a STOP gate before access", "uncompressed_bytes_max": 1099511627776, "tar_header_records_max": 10000000, "member_body_bytes_max": 68719476736, "volatile_chunk_bytes_max": 1048576, "unique_scenarios_max": 627, "wall_clock_seconds_max": 86400, "policy": "Hard caps include discarded bodies, padding and all compressed requests/retries. No automatic retries/restarts, expanded budgets or fallback. Exceeding any bound marks inventory incomplete and stops; revised bounds/source records need a separately sealed access amendment before another attempt."},
        "path_parser": {
            "header_only": True,
            "accepted": "512-byte ordinary POSIX/USTAR headers with checksum/size validation; name and prefix fields only for member path. Optional single documented carlanomaly/ wrapper or ./ prefix canonicalized, recording the transformation.",
            "normal_grammar": "test/normal/<town>/scenario-<positive decimal ID>",
            "anomaly_grammar": "test/anomaly/<town>/<anomaly-type>/scenario-<positive decimal ID>",
            "normal_type": "NORMAL metadata sentinel; never an observation label",
            "metadata": "Full scenario prefix, town, directory condition normal/anomaly and anomaly type derived only from header path components; scenario suffix has no family semantics.",
            "rejection": "Reject absolute/drive/UNC paths, backslashes, .. traversal, NUL ambiguity, invalid encoding, malformed TEST prefixes, unsupported town/type, conflicting canonical identities, links/devices/sparse entries. STOP on PAX/GNU long-name/link or other extensions requiring body inspection to resolve paths; do not decode extension bodies. Support changes require a later sealed header-access amendment.",
            "scope": "After scenario prefix is derived, never interpret child filename/tick numbers or suffixes as annotation/timestep data; no tick enumeration emitted. Non-scenario structural root directories may be accepted; unclassifiable data-bearing members halt."
        },
        "procedure_only": [
            "Preflight seal source/version/checksum/size, read budgets, explicit authorization, parser/tool source and output schemas; do not instantiate model, simulator, generic devkit index or dataset loaders.",
            "Stream gzip to a bounded tar-header state machine. Validate each 512-byte header checksum/size/path; count compressed/decompressed offsets exactly. Never unpack archive members to disk or cache compressed/decompressed bodies.",
            "Derive scenario ID and town/condition/type from header path only. Add a unique scenario-level record; repeated headers for files in one scenario deduplicate to one ID, while duplicate member paths, duplicate explicit scenario directory entries, duplicate inventory rows or conflicting canonical aliases halt.",
            "Consume member bodies and tar padding in bounded volatile chunks only to advance to next header, discard immediately. Never decode JPEG/PNG, parse Feather, inspect timestep annotations, retain sensor bodies, compute sensor statistics or call model inference. Do not hash or log per-sensor content.",
            "Reach valid full tar termination and consume/check gzip trailer/EOF under the caps; reject truncation, checksum errors, unexpected concatenated streams or later nonpadding tar content. Validate streamed compressed-byte digest against authenticated archive checksum; digest serves byte-integrity only, not sensor statistics. Stop immediately on failure, with incomplete ledger and no eligible inventory seal.",
            "Emit sorted unique scenario IDs with path-derived town/condition/type only. Emit access ledger separately with source/access counters, tool/protocol hashes and completion/failure state. No sensor contents, tick annotations, predictions, scores or labels.",
            "Apply all inventory gates below and seal source inventory, exclusion decisions and eligible inventory hash before RNG creation. Full archive traversal proves structure traversal only; modality/label/synchronization completeness is a later separately authorized feature/label check, never a claim from headers alone."
        ],
        "access_ledger": {
            "exact_fields": ["authorization_reference", "tool_source_sha256", "protocol_sha256", "source_URL", "publisher_release", "advertised_checksum_and_size", "request_ranges_HTTP_status_bytes_received", "start_stop_UTC", "compressed_bytes_consumed", "decompressed_bytes_consumed", "header_bytes_consumed", "member_body_bytes_discarded", "padding_bytes_discarded", "gzip_trailer_and_EOF_status", "compressed_stream_digest", "headers_visited", "unique_scenario_count", "canonicalization_rule", "budget_usage", "failure_reason_or_success"],
            "header_event_fields": ["sequence_number", "decompressed_header_offset", "compressed_input_counter", "header_type", "declared_body_size", "discarded_body_and_padding_counts", "scenario_ID_or_structural_root", "parse_or_failure_code"],
            "limits": "Do not persist complete child paths or timestep identifiers. Exact byte counters and parser decisions suffice for access accounting; ledger is structure/access metadata separate from the scenario-only output. Uninterpreted buffers never logged. No OS-wide access-trace claim."
        },
        "gates_before_randomization": [
            "Complete unique scenario list from authenticated full traversal; every scenario ID path-parseable into town, directory condition and anomaly type; structural EOF/integrity/budget checks passed.",
            "Reconcile published total: exactly 627 full scenario IDs, 107 normal and 520 anomaly before historical exclusions. If publisher records change, STOP for a sealed amendment rather than silently adopt new totals.",
            "No duplicate full inventory scenario IDs; distinguish multiple member headers under one scenario from duplicate inventory rows or duplicate archive entries.",
            "Both known historical IDs found exactly in the pre-exclusion catalogue and removed in full from both final roles.",
            "All independently authenticated historical exposure decisions documented and sealed before execution; incomplete historical file-level ledger remains disclosed; no guessed additional IDs or families.",
            "Seal source inventory hash, exact exclusion supplement and eligible inventory hash before creating PCG64(2028). Reconciliation failure or unresolved exclusion decision means STOP, no partition."
        ],
        "prohibited": ["JPEG/PNG decoding", "Feather parsing", "timestep annotation inspection", "sensor body persistence", "sensor statistics", "model inference", "partition execution in inventory tool", "family inference from scenario-N"],
        "executed": False, "implemented": False
    }
    write("inventory_access_protocol.json", inventory)
    write("subgroup_reporting_protocol.json", {
        "amendment_id": ID,
        "inventory_and_evaluation_ledger_required_fields": ["full_scenario_ID", "town", "directory_condition", "anomaly_type"],
        "reports": ["composition before exclusions, eligible, calibration and evaluation", "metrics by town", "metrics by anomaly type", "metrics by directory condition where meaningful", "empirical scenario-level conformal coverage and set size by each subgroup"],
        "semantics": "Descriptive subgroup diagnostics only. Same per-scorer global Q in all subgroups; no subgroup-specific cutoff or conditional guarantee. Report counts, denominators and undefined reasons; sparse results remain null or explicitly qualified. No realized composition adjustments.",
        "metrics": "Use unchanged frozen metrics and strict undefined conventions. Strict all-scenario scenario-macro AUROC stays primary where defined; two-class-eligible macro separately secondary. Scenario simultaneous coverage stays primary CP diagnostic. No new metric selected by outcomes."
    })
    write("readiness.json", {
        "amendment_id": ID, "design_sealed": True, "official_TEST_partition_ready": False,
        "conformal_fitting_ready": False, "no_TEST_access": True,
        "no_inventory_execution": True, "no_partition_or_conformal_fit": True,
        "known_exact_exclusions": EXCLUSIONS,
        "exclusion_amendment_id": "001_pre_TEST_historical_exposure_correction",
        "blockers": ["Complete exact authenticated scenario inventory unavailable; later structure-only access requires separate authorization", "Inventory reconciliation, historical exclusion decisions and eligible-inventory hash must all be sealed before RNG creation", "Final official label/inference/scenario conformal adapter remains unimplemented and unverified; no fitting authorized"],
        "limitation": "Family/route mapping unavailable per sealed audit; exact family allocation impossible from public information. Finite-catalogue scope only; incomplete historical file-level exposure ledger disclosed.",
        "next_step": "Stop after sealing Amendment 002. Future authorized inventory acquisition must follow inventory_access_protocol.json and stop on failed reconciliation; no inventory, RNG or scores now."
    })
    write("milestone_scope.json", {
        "milestone": "FINAL PRE-PREDICTION CONFORMAL / PARTITION PROTOCOL AMENDMENT ONLY",
        "amendment_id": ID,
        "authorized_actions_executed": ["Read sealed protocol and audit text", "Verify metadata/protocol seals", "Archive complete prior protocol byte-for-byte", "Write protocol-only amendment artifacts", "Hash source/config text for preservation", "Read-only Git state checks", "Static protocol/algebra checks and sealing"],
        "prohibited": ["TEST archive access", "TEST scenario enumeration", "sensor or label access", "inventory execution", "partition execution", "predictions", "conformal fitting", "training", "tuning", "commit", "push"],
        "limits": "Executed-tool attestation, not an OS-wide access trace. Existing historical findings are quoted from sealed text; no payload/model-output rehash."
    })
    write("amendment_002_record.json", {
        "amendment_id": ID, "client_date": "2026-10-02", "client_timezone": "Asia/Calcutta",
        "authorization": "Explicit user request for this protocol amendment only",
        "previous_protocol_manifest_sha256": OLD_HASH,
        "public_provenance_audit_manifest_sha256": PROV_HASH,
        "metadata_audit_manifest_sha256": META_HASH,
        "previous_protocol_archive": "amendment_history/pre_amendment_002/",
        "prior_amendment_record": "amendment_record.json (Amendment 001; preserved unchanged)",
        "prior_files_archived": len(prior),
        "established_reasons": [
            "Original stratification: town x directory condition x anomaly type, n_CAL,h=ceil(N_h/5), Q=max_h Q_h.",
            "At alpha=.05 finite Q_h requires at least 19 calibration scenarios; under old ceil(N_h/5) allocation this requires N_h>=91.",
            "Publisher totals establish 107 normal TEST scenarios across six towns. At most one normal-town stratum can have N_h>=91; at least five fail finite rank. Hence at least one Q_h=+inf, Q=+inf and all conformal sets are {normal, anomaly}. This follows from metadata/count mathematics alone, not model outcomes.",
            "Public audit found no recoverable released-scenario to seed/route/base-drive/family mapping. scenario-N has no supported family semantics; exact family partitioning unavailable from public information."
        ],
        "changes": ["Uniform whole-scenario split with one PCG64(2028) permutation and n_cal=ceil(N_eligible/5)", "Pooled one-score-per-scenario global cutoff per scorer", "Finite-catalogue marginal guarantee and explicit nonclaims", "Bounded future structure-only inventory design and gates", "Descriptive subgroup preservation and removal/rewording of conformal stratification fields only"],
        "outcome_independent": "No replacement choice uses performance, TEST labels, predictions or fitted conformal set sizes.",
        "unchanged": ["upstream M1", "graph architecture", "checkpoints", "graph readout/fusion", "graph seeds", "probability calibration", "node features", "pseudo recipes", "threshold selection", "target semantics", "GNSS exclusion", "final performance metrics except conformal stratification-specific fields necessarily removed/reworded", "generic COGNIX"],
        "new_manifest_reference": "SHA256SUMS and SHA256SUMS.sha256 (detached digest avoids self-reference)",
        "current_integrity_reference": "amendment_002_integrity.json",
        "historical_records": "Earlier integrity/validation facts describe their recorded milestones. Prior audit baseline hashes continue to refer to pre-Amendment-002 bytes now archived; do not regenerate historical audits. Current protocol validation and Amendment 002 integrity supersede current-readiness statements.",
        "activity_attestation": {name: False for name in ["TEST_access", "TEST_scenario_enumeration", "sensor_or_label_access", "inventory_execution", "partition_execution", "inference", "conformal_fitting", "training", "tuning", "commit", "push"]}
    })
    # Update prose by named sections; preserve all unrelated frozen content.
    report = (OLD / "report.md").read_text(encoding="utf-8")
    def section(title, body):
        nonlocal report
        start = report.index("# " + title + "\n")
        end = report.find("\n# ", start + 1)
        if end < 0:
            end = len(report)
        report = report[:start] + "# " + title + "\n\n" + body.strip() + "\n" + report[end:]
    report = report.replace("Official metadata counts, provenance and actual final membership remain unaudited; no partition has occurred.", "Sealed metadata/count and public-provenance audits now establish publisher aggregates and unavailable family mapping. The exact complete catalogue and actual final membership remain unavailable; no partition has occurred.")
    section("Proposed FINAL_CONFORMAL_CAL / FINAL_EVALUATION Split", """
**Amendment 002 — final pre-prediction partition/conformal amendment only.** The original rule stratified by **town × directory condition × anomaly type**, with **n_CAL,h=ceil(N_h/5)** and **Q=max_h Q_h**. At **alpha=.05**, finite Q_h requires at least **19** calibration scenarios, hence N_h≥91 under that allocation. Publisher totals establish **107 normal TEST scenarios across six towns**. At most one such stratum can reach 91; at least five necessarily fail finite rank. Consequently at least one Q_h=+inf, Q=+inf and all conformal sets are **{normal, anomaly}**. This is metadata/count mathematics alone, independent of model outcomes. The sealed public audit found no recoverable released-scenario → seed/route/base-drive/family mapping; scenario-N has no supported family semantics. Exact family allocation is unavailable from public information.

Replace quotas with **one uniform whole-scenario split over the complete eligible TEST catalogue**. Before randomization exclude the two exact historically exposed scenarios above, plus any future independently authenticated historical exposure documented and sealed before partition execution. Seal all exclusion decisions. Use the **lexicographically sorted complete eligible full scenario IDs**, **PCG64 seed 2028**, and **exactly one RNG permutation**. Set **n_cal=ceil(N_eligible/5)**. The first n_cal permuted IDs become **FINAL_CONFORMAL_CAL**; the remainder become **FINAL_EVALUATION**. All ticks/windows inherit the complete scenario's role. No stratification by town, directory condition or anomaly type; no redraws or balancing. Preserve ordered permutation and identical membership for all 15 frozen scorers.

The exact 627-scenario list remains unavailable publicly. `inventory_access_protocol.json` and `inventory_access_protocol.md` design a later separately authorized, bounded structure-only header traversal; no extractor is implemented or executed here. Sequential gzip traversal may carry payload bytes through decompression, discarded without interpretation or persistence; it is not zero payload-byte access. Output only unique scenario IDs and path-derived town/condition/type, plus a separate exact access ledger.

Before any later RNG creation require complete unique inventory, full traversal/integrity validation, exactly **107 normal and 520 anomaly (627 total) before exclusions**, both known excluded IDs found and removed, no duplicate full scenario IDs, every ID path-parseable, all exclusion decisions sealed, and the inventory hash sealed. Changed publisher counts or failed reconciliation mean **STOP** and a new amendment. Membership/complete payload alignment cannot be inferred from aggregate counts. If only the two known IDs are removed, 625/125/500 are conditional eligible/calibration/evaluation count arithmetic, not executed assignments. Seal role manifest, exclusions, ordered permutation, realized descriptive composition, RNG/version and protocol hashes before future scoring.
"""
    )
    section("Scenario / Temporal Exchangeability", """
The complete scenario is the indivisible assignment and calibration unit. Ticks and overlapping windows are dependent; a fixed maximum over all frozen eligible ticks needs no within-scenario independence or permutation symmetry.

The **registered primary interpretation is finite-catalogue marginal randomization/exchangeability**: conditional on the frozen scorer, complete eligible released TEST inventory, uniform scenario assignment, and exclusions fixed before randomization, calibration and a uniformly selected held-out scenario have symmetric score positions under the registered split. Arbitrarily dependent catalogue members can be treated as fixed units for this random-assignment claim. PCG64(2028) is the committed reproducible realization; the guarantee is over the registered assignment mechanism and held-out scenario selection, not conditional on that realized seed/partition or each calibration sample.

The sealed public-provenance audit recovered no released-scenario family/route mapping, and scenario-N has no supported family semantics. Related drives/routes or normal/anomaly variants may cross roles. **Family/route dependence remains an external-generalization limitation**, even though IDs are unavailable. No independent-family coverage, new-route population coverage, iid drive generation, arbitrary future CARLA coverage, live-distribution coverage, physical-safety certification, conditional coverage by town/type/condition, simultaneous coverage of every evaluation scenario, or joint coverage across all 15 scorers is claimed. Subgroup results are descriptive only. Hierarchical or population coverage would require additional assumptions and a separately sealed design.
"""
    )
    section("Conformal Construction", """
Retain all **15 frozen method/seed scorers** and **s_j,t(y)=1−P_j,t(y)**. For each FINAL_CONFORMAL_CAL scenario, **S_j=max_t s_j,t(Y_j,t)** over all eligible synchronized ticks under the unchanged feature/label protocol: t≥1 through the actual final tick; causal trailing window [max(0,t−11),t]; tick 0 excluded. Each scenario contributes exactly one score. No pseudo corruption, tick balancing, outcome filtering or probability recalibration.

Pool all n_cal scenario scores into **one global calibration set per scorer**. Augment with +infinity, set **k=ceil((n_cal+1)·0.95)** (exact integer form `(19*(n_cal+1)+19)//20`), and take the augmented kth order statistic **Q**. Inclusive comparison preserves ties; rank beyond n_cal gives +infinity. Define **C_j,t={y : 1−P_j,t(y)≤Q}**. Fit a separate global Q for each of the 15 scorers with **identical calibration membership** and **no prediction ensemble**. Remove per-town, per-condition and per-anomaly-type cutoffs, stratum-specific Q_h and Q=max_h Q_h. Finite global rank does not promise small sets; retain infinite or inefficient outcomes.

For the registered uniform split and a uniformly selected held-out catalogue scenario, conditional on the fixed eligible catalogue and scorer, the n_cal+1 score positions are symmetric. The augmented-rank argument gives **Pr_randomization(S_new≤Q)≥k/(n_cal+1)≥.95**, conservatively with ties. Since S_new is a maximum, **S_new≤Q implies simultaneous inclusion of all eligible true tick labels for that one scenario**. This is a marginal finite-catalogue statement under the conditions above; it does not cover every evaluation scenario jointly or all 15 scorers jointly. Family, route, future-drive, conditional-subgroup, live-distribution and safety nonclaims remain explicit in `conformal_protocol.json`.

Generic ConformalPredictor remains unchanged: its ordinary row fit has no scenario grouping. A future report-side adapter must reduce one block score per scenario, pool scores and store scenario membership, scores, n_cal, rank, alpha and the single global Q per scorer. Do not disguise block scores as probability rows or ordinary tick fitting. No adapter implementation or conformal fitting occurred in this milestone.
"""
    )
    report = report.replace("and counted per-stratum diagnostics", "and counted descriptive subgroup diagnostics by town, anomaly type and directory condition where meaningful, with sparse results null or qualified; no subgroup-specific cutoff or guarantee")
    report = report.replace("full conformal block scores/counts/strata/cutoffs/envelopes/alpha", "full pooled conformal block scores/scenario membership/counts/ranks/global per-scorer cutoffs/alpha and descriptive subgroup metadata")
    report = report.replace("Preservation verification compares 3764", "Historical preregistration preservation verification compared 3764")
    report = report.replace("current commands reverify it rather than overwrite it", "its historical results are retained; Amendment 002 verifies only protocol/audit/source metadata bytes without reopening TRAIN or model outputs")
    report = report.replace("checks all 120100 raw TRAIN files", "checked all 120100 raw TRAIN files")
    report = report.replace("The current-turn content check and final verification results are separate artifacts.", "Those historical content checks remain preserved as separate artifacts; they were not rerun for Amendment 002.")
    report = report.replace("verifies prior result/protocol/environment seals", "verified prior result/protocol/environment seals")
    report = report.replace("Git HEAD/tracked diff and status outside this directory remain unchanged.", "Git HEAD remains unchanged; tracked diff and status outside this protocol directory remain unchanged. Changes inside this directory are listed in Amendment 002 integrity.")
    report = report.replace("compares 3764 pre-existing", "compared 3764 pre-existing")
    report = report.replace("# Protocol Hashes\n\n", "# Protocol Hashes\n\n**Amendment 002:** complete pre-amendment protocol (including Amendment 001 history and both prior seal files) is preserved byte-for-byte in `amendment_history/pre_amendment_002/`. Old protocol manifest: `" + OLD_HASH + "`. Public provenance manifest: `" + PROV_HASH + "`. `amendment_002_record.json`, `amendment_002_changelog.md` and `amendment_002_integrity.json` define the current change and preservation checks. Earlier audit baselines refer to the archived pre-amendment bytes; do not rewrite the sealed audits.\n\n", 1)
    section("Ready / Not Ready for Official TEST Partition", """
**NOT READY to execute a partition.** Amendment 002 is design only. The exact complete authenticated scenario catalogue remains unavailable. Later separately authorized structure-only extraction must pass every inventory gate and seal all exclusions and inventory hashes before RNG creation. Both historical exclusions remain fixed; the incomplete historical file-level ledger remains disclosed, with no additional IDs inferred. Family IDs are unavailable; the registered guarantee is finite-catalogue only.
"""
    )
    section("Ready / Not Ready for Conformal Fitting", """
**NOT READY to fit conformal.** Obtain and seal the reconciled eligible inventory and role manifest in later authorized stages; implement and verify a minimal pooled scenario adapter without altering the frozen scorer or generic COGNIX; preserve official label/alignment semantics. Calibration-only payload access and fitting need separate authorization. Evaluation payloads/predictions/labels remain inaccessible during calibration. No score-based partition, construction, metric or threshold changes.
"""
    )
    section("Recommended Next Step", """
Stop after sealing Amendment 002. This milestone performed no TEST access, TEST scenario enumeration, sensor or label access, inventory execution, partition, inference, conformal fitting, training, tuning, commit or push. Upstream M1, graph architecture/checkpoints/readout/fusion/seeds, probability calibration, node features, pseudo recipes, threshold selection, target semantics, GNSS exclusion, frozen performance metrics (except necessary conformal stratification rewording), and generic COGNIX remain unchanged. Future inventory work requires separate authorization and must stop on failed reconciliation.
"""
    )
    (HERE / "report.md").write_text(report, encoding="utf-8")
    demo = (OLD / "live_demo_plan.md").read_text(encoding="utf-8")
    demo = demo.replace("conformal scenario scores/counts/strata/alpha/cutoffs/global envelopes", "pooled conformal scenario scores/membership/counts/ranks/alpha/global per-scorer cutoffs and descriptive subgroup metadata")
    demo = demo.replace("The conservative conformal envelope requires no oracle official normal/anomaly/type metadata at inference. Its mathematical guarantee still assumes a finite scenario generated within registered exchangeable conditions.", "Each pooled global cutoff requires no oracle official normal/anomaly/type metadata at inference. Amendment 002 registers only marginal randomization coverage for a held-out scenario from the frozen eligible released catalogue; no live-generated scenario coverage is established.")
    (HERE / "live_demo_plan.md").write_text(demo, encoding="utf-8")
    print(json.dumps({"amendment_written": ID, "previous_manifest": OLD_HASH,
                      "archived_files": len(prior), "seal_pending": True}))


if __name__ == "__main__":
    main()

"""Amendment 002 verifier: metadata/source/config reads only; no science imports.
check/seal finalize validation and integrity before the new seal. verify is
read-only. No archives, raw data, labels, checkpoints or predictions are opened.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE / "amendment_history/pre_amendment_002"
OLD_HASH = "e03da4019c35d53420e42e02d4c0ebe2dd539a24ff5b645fdf5427078122ade1"
META_HASH = "888a63a0f0bb190eb899cfe900674bea6804b354017993b45c235b41b1c26839"
PROV_HASH = "88242128bd2d4b3564e9c58ef12445b7c44c146f0a7f79b30e14371ed9700c50"
ID = "002_pre_prediction_uniform_catalogue_pooled_scenario_conformal"
CHANGED = {"test_partition_protocol.json", "conformal_protocol.json", "final_metrics_spec.json",
           "dependency_audit.json", "readiness.json", "milestone_scope.json", "report.md",
           "live_demo_plan.md", "verify_protocol.py", "protocol_validation.json"}
NEW = {"amend_protocol_002.py", "amendment_002_initial_state.json", "amendment_002_record.json",
       "amendment_002_changelog.md", "amendment_002_integrity.json", "inventory_access_protocol.json",
       "inventory_access_protocol.md", "subgroup_reporting_protocol.json"}


def sha(path):
    assert path.suffix.lower() not in {".gz", ".tar", ".zip", ".pt", ".npz", ".feather", ".jpg", ".png"}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def manifest_files(directory):
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
            and p not in {directory / "SHA256SUMS", directory / "SHA256SUMS.sha256"}}


def verify_manifest(directory, expected):
    assert sha(directory / "SHA256SUMS") == expected
    seen = set()
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        path = (directory / rel).resolve()
        assert path.is_relative_to(directory.resolve()) and rel not in seen
        assert sha(path) == digest, path
        seen.add(rel)
    assert seen == manifest_files(directory), "Unsealed or missing file"
    detached = directory / "SHA256SUMS.sha256"
    if detached.exists():
        assert detached.read_text().split() == [expected, "SHA256SUMS"]
    return len(seen)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def sections(text):
    result = {}
    title = None
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:]
            result[title] = []
        elif title is not None:
            result[title].append(line)
    return {k: "\n".join(v).strip() for k, v in result.items()}


def static_checks():
    base = read(HERE / "amendment_002_initial_state.json")
    prior = base["prior_protocol_files"]
    assert base["old_protocol_manifest_sha256"] == OLD_HASH
    assert verify_manifest(OLD, OLD_HASH) == len(prior) - 2
    for rel, digest in prior.items():
        assert sha(OLD / rel) == digest, rel
        if rel not in CHANGED | {"SHA256SUMS", "SHA256SUMS.sha256"}:
            assert sha(HERE / rel) == digest, "Unapproved change: " + rel
    assert {p.relative_to(OLD).as_posix() for p in OLD.rglob("*") if p.is_file()} == set(prior)
    metadata_count = verify_manifest(ROOT / "reports/carla_test_metadata_audit_v1", META_HASH)
    provenance_count = verify_manifest(ROOT / "reports/carla_public_provenance_audit_v1", PROV_HASH)
    for rel, digest in base["protected_source_config_files"].items():
        path = (ROOT / rel).resolve()
        assert path.is_relative_to(ROOT)
        assert rel.split("/")[0] in {"cognix", "tests", "configs"} or "/" not in rel
        assert sha(path) == digest, "Source/config changed: " + rel
    assert git("rev-parse", "HEAD").strip() == base["git_head"]
    assert hashlib.sha256(git("diff", "HEAD", "--binary", "--", ".", ":(exclude)reports/carla_final_evaluation_preregistration_v1/**").encode()).hexdigest() == base["git_tracked_diff_sha256"]
    outside = [line for line in git("status", "--porcelain=v1").splitlines()
               if "reports/carla_final_evaluation_preregistration_v1/" not in line]
    assert outside == base["git_status_outside_protocol"]
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()}
    expected = set(prior) | NEW | {"amendment_history/pre_amendment_002/" + rel for rel in prior}
    assert actual <= expected, "Unexpected files: " + repr(actual - expected)
    assert expected - actual <= {"amendment_002_integrity.json"}, expected - actual
    for path in HERE.glob("*.json"):
        read(path)
    for name in ("amend_protocol_002.py", "verify_protocol.py"):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"), filename=name)
        modules = {n.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
        assert modules <= {"argparse", "ast", "hashlib", "json", "shutil", "subprocess"}
        assert {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} <= {"pathlib"}
    cp = read(HERE / "conformal_protocol.json")
    old_cp = read(OLD / "conformal_protocol.json")
    bindings = read(HERE / "frozen_method_bindings.json")
    assert cp["alpha"] == .05 and cp["coverage_target"] == .95
    assert cp["methods"] == bindings["methods"] == ["nograph", "standard_gat", "epistemic_gat"]
    assert cp["seeds"] == bindings["seeds"] == [101, 202, 303, 404, 505]
    assert len(bindings["runs"]) == cp["global_cutoffs"] == 15
    assert {(x["method"], x["seed"]) for x in bindings["runs"]} == {(m, s) for m in cp["methods"] for s in cp["seeds"]}
    for key in ("score", "scenario_score", "eligible_ticks", "class_order", "target", "alternatives"):
        assert cp[key] == old_cp[key], "Frozen scoring/target field changed: " + key
    assert not any(cp[key] for key in ("fitted", "executed", "adapter_implemented", "ready_for_fitting", "prediction_ensemble", "TEST_payload_access_authorized"))
    assert cp["identical_calibration_membership"]
    assert all(x in cp["calibration"] for x in ("ONE", "Pool", "ceil((n_cal+1)*0.95)", "+infinity", "Inclusive"))
    assert cp["removed_construction"] == ["per-town cutoffs", "per-condition cutoffs", "per-anomaly-type cutoffs", "stratum-specific Q_h", "Q=max_h Q_h"]
    assert all(x in cp["primary_guarantee"] for x in ("Marginal", "eligible", "catalogue", "uniform", "frozen", "exclusion"))
    partition = read(HERE / "test_partition_protocol.json")
    old_partition = read(OLD / "test_partition_protocol.json")
    ids = ["test/anomaly/Town01/change-weather/scenario-1", "test/anomaly/Town01/change-weather/scenario-10"]
    assert partition["exact_excluded_scenario_ids"] == ids
    for key in ("seed", "seed_distinct_from", "candidate_fraction", "unit", "scenario_identity", "exposure_record_sha256", "exclusion_classification", "excluded_from_both_final_roles", "exclusion_amendment_id", "historical_file_level_ledger_complete"):
        assert partition[key] == old_partition[key], key
    assert sha(HERE / "test_exposure_record.json") == partition["exposure_record_sha256"]
    assert partition["seed"] == 2028 and partition["allocation"]["permutation_calls"] == 1
    assert not partition["allocation"]["stratify_assignment"] and not partition["allocation"]["redraws_or_balancing"]
    assert not ({"stratum", "stratification_and_validity", "count_gate", "ratio_status"} & set(partition))
    assert not any(partition[k] for k in ("executed", "partition_performed", "readiness", "complete_eligible_inventory_known", "TEST_payload_access_authorized"))
    # Count/rank algebra only: no score arrays, RNG, partitions or fitting.
    rank = lambda n: (19 * (n + 1) + 19) // 20
    quota = lambda n: (n + 4) // 5
    assert rank(18) == 19 and rank(19) == 19
    assert quota(90) == 18 and quota(91) == 19 and 107 < 2 * 91
    assert 107 + 520 == 627 and 627 - 2 == 625
    assert quota(625) == 125 and 625 - quota(625) == 500 and rank(125) == 120
    metrics = read(HERE / "final_metrics_spec.json")
    old_metrics = read(OLD / "final_metrics_spec.json")
    for key in old_metrics.keys() - {"conformal_metrics", "scenario_condition_target"}:
        assert metrics[key] == old_metrics[key], "Metric changed: " + key
    old_cm = dict(old_metrics["conformal_metrics"])
    cm = dict(metrics["conformal_metrics"])
    old_cm.pop("per_stratum_coverage")
    cm.pop("per_subgroup_coverage")
    assert cm == old_cm, "Non-stratification conformal metric changed"
    assert "null" in metrics["undefined_primary_policy"] and "Secondary" in metrics["preregistered_descriptive_alternative"]
    dep = read(HERE / "dependency_audit.json")
    old_dep = read(OLD / "dependency_audit.json")
    assert {k: v for k, v in dep.items() if k not in {"scenario", "guarantee_status"}} == {k: v for k, v in old_dep.items() if k not in {"scenario", "guarantee_status"}}
    seeds = read(HERE / "seed_aggregation_spec.json")
    assert not seeds["ensemble_used"] and seeds["no_best_seed"]
    inventory = read(HERE / "inventory_access_protocol.json")
    assert not inventory["implemented"] and not inventory["executed"] and inventory["path_parser"]["header_only"]
    assert len(inventory["gates_before_randomization"]) == 6
    assert "Payload bytes may transit" in inventory["byte_transit"]
    assert inventory["mechanical_bounds"]["unique_scenarios_max"] == 627
    assert "PAX/GNU" in inventory["path_parser"]["rejection"]
    assert not any((HERE / n).exists() for n in ("test_scenario_inventory.json", "role_manifest.json", "predictions.npz", "conformal_state.json"))
    ready = read(HERE / "readiness.json")
    assert not ready["official_TEST_partition_ready"] and not ready["conformal_fitting_ready"]
    record = read(HERE / "amendment_002_record.json")
    assert record["amendment_id"] == ID and record["prior_files_archived"] == len(prior)
    assert all(not value for value in record["activity_attestation"].values())
    old_sections = sections((OLD / "report.md").read_text(encoding="utf-8"))
    current_sections = sections((HERE / "report.md").read_text(encoding="utf-8"))
    assert list(current_sections) == list(old_sections)
    for title in ("Frozen Development Findings", "End-to-End COGNIX Pipeline Map", "Graph/Fusion Relationship", "Calibration Dependency Audit", "Final Target Semantics", "Seed Aggregation Policy", "Decision / Attribution Semantics", "GNSS Disposition"):
        assert current_sections[title] == old_sections[title], "Frozen report section changed: " + title
    changed = sorted(rel for rel, digest in prior.items() if rel not in {"SHA256SUMS", "SHA256SUMS.sha256"} and sha(HERE / rel) != digest)
    assert set(changed) <= CHANGED
    validation = {"amendment_id": ID, "passed": True,
        "checks_scope": "Metadata/source/config consistency, count/rank algebra and seals only; no science imports, archives, payloads, labels, checkpoints or prediction arrays.",
        "JSON_syntax": "Current JSON parsed, nonfinite constants rejected", "method_seed_bindings": 15,
        "frozen_score_target_features_unchanged": True, "non_stratification_metrics_unchanged": True,
        "uniform_single_permutation_design_verified_without_execution": True,
        "pooled_scenario_rank_design_verified_without_scores_or_fitting": True,
        "finite_catalogue_guarantee_and_nonclaims_checked": True, "subgroup_diagnostics_preserved": True,
        "structure_only_inventory_design_checked_without_execution": True,
        "TEST_partition_ready": False, "conformal_fitting_ready": False,
        "prior_protocol_archive_files_verified": len(prior), "metadata_audit_entries_verified": metadata_count,
        "public_provenance_audit_entries_verified": provenance_count}
    integrity = {"amendment_id": ID, "passed": True, "old_protocol_manifest_sha256": OLD_HASH,
        "metadata_audit_manifest_sha256": META_HASH, "public_provenance_audit_manifest_sha256": PROV_HASH,
        "prior_protocol_archive": "amendment_history/pre_amendment_002/",
        "prior_files_archived_byte_identically": len(prior),
        "protected_source_config_files_verified": len(base["protected_source_config_files"]),
        "changed_prior_artifacts_excluding_seals": changed, "new_artifacts_excluding_history": sorted(NEW),
        "archive_new_file_count": len(prior), "git_head_before_and_after": base["git_head"],
        "git_HEAD_and_tracked_diff_and_status_outside_protocol_preserved": True,
        "all_prior_files_outside_authorized_change_list_unchanged": True,
        "unchanged_scientific_components": record["unchanged"],
        "binding_files_preserved_without_checkpoint_reads": True,
        "new_manifest_reference": "SHA256SUMS.sha256", "seal_changes": ["SHA256SUMS", "SHA256SUMS.sha256"],
        "change_details": {rel: {"old_sha256": prior[rel], "new_sha256": sha(HERE / rel)} for rel in changed if rel != "protocol_validation.json"},
        "validation_hash_location": "SHA256SUMS seals finalized protocol_validation.json and this record",
        "activity_attestation": record["activity_attestation"],
        "scope_limits": "No payload/model-output rehash or full untracked-content checksum. Beyond hashed metadata/source/config bytes, preservation rests on confined writes and unchanged Git state. Executed-action attestation, not OS-wide access trace. Historical scientific integrity evidence unchanged."}
    return validation, integrity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "seal", "verify"])
    mode = parser.parse_args().mode
    manifest = HERE / "SHA256SUMS"
    current = sha(manifest)
    if mode != "verify":
        assert current == OLD_HASH, "Already sealed: use read-only verify; new changes require explicit amendment"
    validation, integrity = static_checks()
    if mode == "verify":
        assert current != OLD_HASH
        count = verify_manifest(HERE, current)
        assert read(HERE / "protocol_validation.json") == validation
        assert read(HERE / "amendment_002_integrity.json") == integrity
        print(json.dumps({"passed": True, "sealed_files": count, "old_manifest_sha256": OLD_HASH,
                          "new_manifest_sha256": current, "no_TEST_inventory_partition_scoring_fitting": True}))
        return
    write("protocol_validation.json", validation)
    validation, integrity = static_checks()
    write("amendment_002_integrity.json", integrity)
    if mode == "seal":
        manifest.write_text("".join(f"{sha(HERE / rel)}  {rel}\n" for rel in sorted(manifest_files(HERE))), encoding="utf-8")
        current = sha(manifest)
        (HERE / "SHA256SUMS.sha256").write_text(f"{current}  SHA256SUMS\n", encoding="utf-8")
        count = verify_manifest(HERE, current)
        print(json.dumps({"passed": True, "sealed_files": count, "old_manifest_sha256": OLD_HASH, "new_manifest_sha256": current}))
    else:
        print(json.dumps(validation))


if __name__ == "__main__":
    if (HERE / "amendment_003_record.json").exists():
        import runpy
        runpy.run_path(str(HERE / "verify_protocol_003.py"), run_name="__main__")
    else:
        main()

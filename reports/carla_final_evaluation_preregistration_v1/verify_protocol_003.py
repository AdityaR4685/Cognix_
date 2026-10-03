"""Metadata-only Amendment 003 validator/sealer; never generates randomness.

Reads protocol/audit/source/config metadata only. verify is read-only.
check/seal finalize validation/integrity before the new seal; resealing is
forbidden once the new manifest exists. No inventory/partition/science imports.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE / "amendment_history/pre_amendment_003"
OLD_HASH = "c6516d507a7b0189c4fc11d0ad429a7dc080ebb49ea127623a87e1c4ed39482b"
RECEIPT_HASH = "2a6cc338637a6a2b6ce661f6711ac4ea3e45ff6337346d27ff1b4753e3377368"
ID = "003_partition_randomization_clarification_only"
CHANGED = {"test_partition_protocol.json", "conformal_protocol.json", "inventory_access_protocol.json",
           "inventory_access_protocol.md", "readiness.json", "report.md", "verify_protocol.py", "protocol_validation.json"}
NEW = {"amend_protocol_003.py", "verify_protocol_003.py", "amendment_003_initial_state.json",
       "partition_randomization_receipt.json", "partition_randomization_receipt.sha256",
       "amendment_003_record.json", "amendment_003_changelog.md", "amendment_003_integrity.json"}


def sha(path):
    assert path.suffix.lower() not in {".gz", ".tar", ".zip", ".pt", ".npz", ".feather", ".jpg", ".png"}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def files(directory):
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
    assert seen == files(directory), "Unsealed or missing file"
    if (directory / "SHA256SUMS.sha256").exists():
        assert (directory / "SHA256SUMS.sha256").read_text().split() == [expected, "SHA256SUMS"]
    return len(seen)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def diffs(a, b, path=()):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        result = []
        for key in a.keys() | b.keys():
            result += diffs(a[key], b[key], path + (key,)) if key in a and key in b else [path + (key,)]
        return result
    if isinstance(a, list):
        if len(a) != len(b):
            return [path]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in diffs(x, y, path + (i,))]
    return [] if a == b else [path]


def changed_fields(name, allowed):
    delta = diffs(read(OLD / name), read(HERE / name))
    assert all(any(p[:len(prefix)] == prefix for prefix in allowed) for p in delta), (name, delta)
    return ["/".join(map(str, p)) for p in sorted(delta, key=str)]


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


def checks():
    base = read(HERE / "amendment_003_initial_state.json")
    prior = base["prior_files"]
    assert base["old_manifest_sha256"] == OLD_HASH
    assert verify_manifest(OLD, OLD_HASH) == len(prior) - 2 == 120
    for rel, digest in prior.items():
        assert sha(OLD / rel) == digest
        if rel not in CHANGED | {"SHA256SUMS", "SHA256SUMS.sha256"}:
            assert sha(HERE / rel) == digest, "Unapproved change: " + rel
    assert {p.relative_to(OLD).as_posix() for p in OLD.rglob("*") if p.is_file()} == set(prior)
    metadata_entries = verify_manifest(ROOT / "reports/carla_test_metadata_audit_v1", "888a63a0f0bb190eb899cfe900674bea6804b354017993b45c235b41b1c26839")
    provenance_entries = verify_manifest(ROOT / "reports/carla_public_provenance_audit_v1", "88242128bd2d4b3564e9c58ef12445b7c44c146f0a7f79b30e14371ed9700c50")
    for rel, digest in base["protected_source_config_files"].items():
        path = (ROOT / rel).resolve()
        assert path.is_relative_to(ROOT)
        assert rel.split("/")[0] in {"cognix", "tests", "configs"} or "/" not in rel
        assert sha(path) == digest
    assert git("rev-parse", "HEAD").strip() == base["git_head"]
    assert hashlib.sha256(git("diff", "HEAD", "--binary", "--", ".", ":(exclude)reports/carla_final_evaluation_preregistration_v1/**").encode()).hexdigest() == base["git_diff_outside_protocol_sha256"]
    assert [line for line in git("status", "--porcelain=v1").splitlines() if "reports/carla_final_evaluation_preregistration_v1/" not in line] == base["git_status_outside_protocol"]
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()}
    expected = set(prior) | NEW | {"amendment_history/pre_amendment_003/" + rel for rel in prior}
    assert actual <= expected, actual - expected
    assert expected - actual <= {"amendment_003_integrity.json"}, expected - actual
    for path in HERE.glob("*.json"):
        read(path)
    receipt = read(HERE / "partition_randomization_receipt.json")
    assert sha(HERE / "partition_randomization_receipt.json") == RECEIPT_HASH
    assert (HERE / "partition_randomization_receipt.sha256").read_text().split() == [RECEIPT_HASH, "partition_randomization_receipt.json"]
    seed = receipt["seed"]
    assert type(seed) is int and 0 <= seed < 2**128 and str(seed) == receipt["seed_decimal"]
    assert receipt["draw_count"] == 1 and receipt["bits"] == 128
    assert receipt["python_version"] == "3.12.14" and receipt["generation_method"].startswith("Python secrets.randbits(128)")
    assert datetime.fromisoformat(receipt["generated_at_utc"]) == datetime.fromisoformat(receipt["generated_at_Asia_Calcutta"])
    assert receipt["before_any_TEST_inventory_access_in_this_milestone"]
    assert not any(receipt[k] for k in ("inventory_accessed", "inventory_executed", "partition_executed", "PCG64_instantiated", "permutations_executed"))
    tree = ast.parse((HERE / "amend_protocol_003.py").read_text(encoding="utf-8"))
    draws = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "secrets" and n.func.attr == "randbits"]
    assert len(draws) == 1 and draws[0].args[0].value == 128
    for name in ("amend_protocol_003.py", "verify_protocol_003.py"):
        parsed = ast.parse((HERE / name).read_text(encoding="utf-8"), filename=name)
        mods = {n.name.split(".")[0] for node in ast.walk(parsed) if isinstance(node, ast.Import) for n in node.names}
        assert mods <= {"argparse", "ast", "hashlib", "json", "subprocess", "os", "platform", "secrets"}
        assert {n.module for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom)} <= {"pathlib", "datetime"}

    allowed_partition = {(k,) for k in ("amendment_id", "status", "seed", "seed_distinct_from", "seed_receipt", "seed_collision_policy", "randomization_interpretation", "seed_draw_timing")}
    allowed_partition |= {("allocation", "seed"), ("algorithm_only", 3), ("algorithm_only", 5), ("algorithm_only", 7)}
    field_changes = {"test_partition_protocol.json": changed_fields("test_partition_protocol.json", allowed_partition),
        "conformal_protocol.json": changed_fields("conformal_protocol.json", {(k,) for k in ("amendment_id", "status", "primary_guarantee", "population_interpretation", "randomization_receipt", "randomization_assumption")} | {("proof", 4)}),
        "inventory_access_protocol.json": changed_fields("inventory_access_protocol.json", {("amendment_id",), ("gates_before_randomization", 5), ("procedure_only", 6)}),
        "readiness.json": changed_fields("readiness.json", {("amendment_id",), ("next_step",), ("partition_seed_receipt",), ("blockers", 1)})}
    partition = read(HERE / "test_partition_protocol.json")
    assert partition["seed"] == partition["allocation"]["seed"] == seed
    assert partition["seed_receipt"] == {"path": "partition_randomization_receipt.json", "sha256": RECEIPT_HASH}
    assert partition["candidate_fraction"] == {"FINAL_CONFORMAL_CAL": .2, "FINAL_EVALUATION": .8}
    assert partition["allocation"]["permutation_calls"] == 1 and partition["allocation"]["n_cal"] == "ceil(N_eligible/5)"
    assert "SEALED_RANDOM_SEED" in partition["algorithm_only"][5]
    assert not any(partition[k] for k in ("executed", "partition_performed", "readiness", "complete_eligible_inventory_known", "TEST_payload_access_authorized"))
    cp = read(HERE / "conformal_protocol.json")
    assert cp["alpha"] == .05 and cp["global_cutoffs"] == 15
    assert cp["randomization_receipt"] == partition["seed_receipt"]
    assert "marginal over" in cp["primary_guarantee"] and "uniform scenario assignment" in cp["primary_guarantee"]
    assert "does not establish" in cp["population_interpretation"]
    assert not cp["fitted"] and not cp["executed"]
    old_sections = sections((OLD / "report.md").read_text(encoding="utf-8"))
    current_sections = sections((HERE / "report.md").read_text(encoding="utf-8"))
    assert list(old_sections) == list(current_sections)
    affected_sections = {"Proposed FINAL_CONFORMAL_CAL / FINAL_EVALUATION Split", "Scenario / Temporal Exchangeability", "Protocol Hashes", "Recommended Next Step"}
    for title in old_sections.keys() - affected_sections:
        assert old_sections[title] == current_sections[title], title
    for name in ("report.md", "inventory_access_protocol.md"):
        assert "PCG64(2028)" not in (HERE / name).read_text(encoding="utf-8")
    assert not any((HERE / n).exists() for n in ("test_scenario_inventory.json", "role_manifest.json", "predictions.npz", "conformal_state.json"))
    ready = read(HERE / "readiness.json")
    assert not ready["official_TEST_partition_ready"] and not ready["conformal_fitting_ready"]
    record = read(HERE / "amendment_003_record.json")
    assert record["amendment_id"] == ID and record["receipt"]["sha256"] == RECEIPT_HASH
    assert record["receipt"]["seed"] == seed and record["seed_draw_count"] == 1
    assert all(not v for v in record["activity_attestation"].values())
    changed = sorted(rel for rel, digest in prior.items() if rel not in {"SHA256SUMS", "SHA256SUMS.sha256"} and sha(HERE / rel) != digest)
    assert set(changed) <= CHANGED
    validation = {"amendment_id": ID, "passed": True, "partition_seed_decimal": str(seed),
        "receipt_sha256": RECEIPT_HASH, "single_OS_draw_recorded_and_receipt_unchanged": True,
        "archive_files_verified": len(prior), "all_Amendment_002_science_metrics_exclusions_subgroups_unchanged": True,
        "ratio_unit_score_conformal_alpha_model_seeds_unchanged": True,
        "inventory_access_design_unchanged_except_randomization_timing": True,
        "finite_catalogue_and_population_interpretations_separated": True,
        "TEST_partition_ready": False, "conformal_fitting_ready": False,
        "metadata_audit_entries_verified": metadata_entries, "provenance_audit_entries_verified": provenance_entries,
        "verification_scope": "Protocol/audit/source/config metadata and static source checks only; receipt inspected without any redraw; no NumPy import, PCG64 creation, permutation, inventory, sensor/label/prediction/checkpoint reads or scientific fitting."}
    integrity = {"amendment_id": ID, "passed": True, "old_manifest_sha256": OLD_HASH,
        "new_manifest_reference": "SHA256SUMS.sha256", "receipt_sha256": RECEIPT_HASH,
        "partition_seed_decimal": str(seed), "previous_archive": "amendment_history/pre_amendment_003/",
        "previous_files_preserved_byte_identically": len(prior),
        "protected_source_config_files_verified": len(base["protected_source_config_files"]),
        "git_head_before_and_after": base["git_head"], "git_state_outside_protocol_preserved": True,
        "changed_prior_files_excluding_seals": changed, "new_files_excluding_history": sorted(NEW),
        "changed_JSON_fields": field_changes,
        "change_hashes": {rel: {"old": prior[rel], "new": sha(HERE / rel)} for rel in changed if rel != "protocol_validation.json"},
        "unchanged_components": record["unchanged"], "activity_attestation": record["activity_attestation"],
        "seed_generation_executed": True, "seed_draws": 1, "partition_permutations_executed": 0,
        "limitations": "Executed-tool attestation and scoped hashes; no OS-wide access trace, raw-data/checkpoint/prediction rehash or full untracked-content checksum. Prior scientific integrity evidence preserved unchanged."}
    return validation, integrity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "seal", "verify"])
    mode = parser.parse_args().mode
    current = sha(HERE / "SHA256SUMS")
    if mode != "verify":
        assert current == OLD_HASH, "Amendment 003 already sealed; verify read-only, never regenerate"
    validation, integrity = checks()
    if mode == "verify":
        assert current != OLD_HASH
        count = verify_manifest(HERE, current)
        assert read(HERE / "protocol_validation.json") == validation
        assert read(HERE / "amendment_003_integrity.json") == integrity
        print(json.dumps({"passed": True, "sealed_files": count, "seed_decimal": validation["partition_seed_decimal"], "old_manifest_sha256": OLD_HASH, "new_manifest_sha256": current}))
        return
    write("protocol_validation.json", validation)
    validation, integrity = checks()
    write("amendment_003_integrity.json", integrity)
    if mode == "seal":
        (HERE / "SHA256SUMS").write_text("".join(f"{sha(HERE / rel)}  {rel}\n" for rel in sorted(files(HERE))), encoding="utf-8")
        current = sha(HERE / "SHA256SUMS")
        (HERE / "SHA256SUMS.sha256").write_text(f"{current}  SHA256SUMS\n", encoding="utf-8")
        count = verify_manifest(HERE, current)
        print(json.dumps({"passed": True, "sealed_files": count, "seed_decimal": validation["partition_seed_decimal"], "old_manifest_sha256": OLD_HASH, "new_manifest_sha256": current}))
    else:
        print(json.dumps(validation))


if __name__ == "__main__":
    main()

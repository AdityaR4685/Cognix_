"""Amendment 001: local historical-evidence correction only; no TEST operations."""
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "amendment_history/pre_amendment_001"
IDS = ["test/anomaly/Town01/change-weather/scenario-1",
       "test/anomaly/Town01/change-weather/scenario-10"]
OLD_MANIFEST = "2e7c20eb2d1a40367c6568027ddaea9e6650b80bee7d57770d94a644772fd6ab"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def corrected_exposure_record():
    evidence_path = HERE / "recovered_historical_evidence.json"
    evidence = read(evidence_path)
    record = read(OLD / "test_exposure_record.json")
    for name in ("empty_list_means", "exact_partial_official_reference", "provisional_quarantine_predicate"):
        record.pop(name, None)
    record.update({
        "status": "KNOWN_EXACT_EXPOSURES_CORRECTED_FILE_LEVEL_LEDGER_INCOMPLETE",
        "amendment_id": "001_pre_TEST_historical_exposure_correction",
        "town": "Town01",
        "exact_fully_qualified_official_scenario_ids_recovered": IDS,
        "scenario_exposures": evidence["scenario_exposures"],
        "main_probe": evidence["main_probe"],
        "earlier_probes": evidence["earlier_probes"],
        "temporary_probe_files_deleted_afterward": True,
        "classification": "SCHEMA_EXPOSED_DIAGNOSTIC_ONLY",
        "exact_exclusions": {"FINAL_CONFORMAL_CAL": IDS, "FINAL_EVALUATION": IDS},
        "exclusion_policy": "Exclude the ENTIRE scenario for both complete and partial historical exposure. Both exact known scenarios are SCHEMA_EXPOSED_DIAGNOSTIC_ONLY and excluded from BOTH FINAL_CONFORMAL_CAL and FINAL_EVALUATION; no cross-town quarantine remains and no exposed metrics merge into untouched evaluation.",
        "historical_file_level_ledger_complete": False,
        "historical_scenario_list_claimed_exhaustive": False,
        "additional_official_TEST_scenario_ids_established": [],
        "unknown_exposures": "No evidence presently establishes any additional official TEST scenario IDs. The historical file-level ledger remains incomplete; no additional IDs, member paths, frame numbers or cross-town exclusions are inferred.",
        "next_requirement": "Retain these two exact exclusions and reconcile the incomplete historical file-level ledger from archival evidence without reopening TEST. Official metadata/provenance/count auditing remains unexecuted and separately authorized; no partition is executed.",
        "recovered_evidence": {"path": evidence_path.name, "sha256": sha(evidence_path),
            "provenance": "User-supplied preserved historical evidence in the amendment request; accepted as the authoritative historical correction. No original TEST probe or payload reopened."}
    })
    return record


def exposure_markdown():
    return """# Historical TEST exposure evidence — Amendment 001

This is a documented pre-TEST-access protocol correction based on recovered historical evidence supplied by the user. No official TEST payload, archive, directory, metadata inventory, image, feather table, mask or sensor sample was accessed in this amendment. Temporary probe files had been deleted afterward; they were not recreated or reopened.

The recovered evidence establishes exactly these known official TEST exposures:

| Exact official scenario ID | Historical exposure extent | Classification | Final role exclusion |
|---|---|---|---|
| `test/anomaly/Town01/change-weather/scenario-1` | Complete: the main sequential probe captured the scenario completely; earlier 64 KB / 8 MB probes also inspected its beginning | SCHEMA_EXPOSED_DIAGNOSTIC_ONLY | Both FINAL_CONFORMAL_CAL and FINAL_EVALUATION, entire scenario |
| `test/anomaly/Town01/change-weather/scenario-10` | Partial: the same main probe continued into this scenario and reached 93 RGB frames | SCHEMA_EXPOSED_DIAGNOSTIC_ONLY | Both FINAL_CONFORMAL_CAL and FINAL_EVALUATION, entire scenario |

Main probe: official `carlanomaly-base-test.tar.gz`; inclusive byte range **0–125,829,119** (125,829,120 bytes); **HTTP 206**; sequential gzip/tar parsing. Scenario-1 was complete; scenario-10 was partial. Do not infer specific frame filenames, additional modalities for the partial capture, earlier probe byte ranges, or other scenario IDs from those counts.

`recovered_historical_evidence.json` records the recovered facts and their provenance as human-provided historical evidence. Existing source documentation independently records TEST schema inspection for GNSS/IMU/labels/segmentation and the all-False change-weather/scenario-1 observation-label pattern. The earlier report relied on those incomplete source summaries and incorrectly left the town unresolved and omitted scenario-10. The recovered evidence supersedes that conclusion. Synthetic unit fixtures remain synthetic and establish no additional official exposures.

The historical **file-level ledger remains incomplete**. No evidence presently establishes any additional official TEST scenario IDs; none are claimed or inferred. Exactly the two known IDs above replace the provisional cross-town quarantine. Both complete and partial exposure disqualify the whole scenario from both final roles; touched ticks are never split off to rehabilitate the remaining ticks as untouched.

The full original preregistration, including its manifest and detached hash, is preserved in `amendment_history/pre_amendment_001/`. Its manifest SHA-256 is `2e7c20eb2d1a40367c6568027ddaea9e6650b80bee7d57770d94a644772fd6ab`. `amendment_record.json` documents this correction and `amendment_integrity.json` records exact byte changes. Scientific methods, seeds, metrics, thresholds, conformal construction, partition seed/ratio/stratification, model/checkpoint bindings and generic COGNIX remain unchanged. No metadata inventory, partition, prediction, training, fitting or tuning occurred. This amendment stops after resealing.
"""


def apply_correction():
    assert sha(OLD / "SHA256SUMS") == OLD_MANIFEST
    assert not (HERE / "amendment_record.json").exists(), "Amendment already applied; do not rewrite a seal"
    evidence = {
        "amendment_id": "001_pre_TEST_historical_exposure_correction",
        "source": "User amendment request describing preserved historical evidence",
        "evidence_scope": "Historical facts supplied in text; no new official TEST or metadata access",
        "archive": "carlanomaly-base-test.tar.gz",
        "official_TEST_archive": True,
        "main_probe": {"byte_range_inclusive": [0, 125829119], "captured_bytes": 125829120,
            "HTTP_status": 206, "parsing": "sequential gzip/tar",
            "scenario_1_complete": True, "scenario_10_partial_RGB_frames": 93},
        "earlier_probes": [{"reported_size": "64 KB", "inspected": "beginning of " + IDS[0]},
                           {"reported_size": "8 MB", "inspected": "beginning of " + IDS[0]}],
        "scenario_exposures": [
            {"scenario_id": IDS[0], "official_TEST": True, "historically_exposed": True,
             "exposure_extent": "complete", "main_probe_captured_scenario_completely": True,
             "earlier_probes_inspected_beginning": True, "classification": "SCHEMA_EXPOSED_DIAGNOSTIC_ONLY",
             "excluded_from": ["FINAL_CONFORMAL_CAL", "FINAL_EVALUATION"]},
            {"scenario_id": IDS[1], "official_TEST": True, "historically_exposed": True,
             "exposure_extent": "partial", "main_probe_captured_RGB_frame_count": 93,
             "classification": "SCHEMA_EXPOSED_DIAGNOSTIC_ONLY",
             "excluded_from": ["FINAL_CONFORMAL_CAL", "FINAL_EVALUATION"]}],
        "temporary_probe_files_deleted_afterward": True,
        "historical_file_level_ledger_complete": False,
        "additional_official_TEST_scenario_ids_established": [],
        "do_not_infer": "Additional IDs, file-level completeness, specific RGB tick filenames, or earlier probe byte ranges"
    }
    write("recovered_historical_evidence.json", evidence)
    write("test_exposure_record.json", corrected_exposure_record())
    (HERE / "exposure_evidence.md").write_text(exposure_markdown(), encoding="utf-8")

    partition = read(OLD / "test_partition_protocol.json")
    partition["exact_excluded_scenario_ids"] = IDS
    partition["exclusion_classification"] = "SCHEMA_EXPOSED_DIAGNOSTIC_ONLY"
    partition["excluded_from_both_final_roles"] = True
    partition["exposure_record_sha256"] = sha(HERE / "test_exposure_record.json")
    partition["exclusion_amendment_id"] = evidence["amendment_id"]
    partition["historical_file_level_ledger_complete"] = False
    partition["algorithm_only"][0] = "Require historical file-level exposure reconciliation, verified official metadata inventory, and scenario-family exchangeability audit; the two known exact exclusions are fixed by Amendment 001. Otherwise STOP without allocating."
    partition["algorithm_only"][2] = "Exclude ENTIRE test/anomaly/Town01/change-weather/scenario-1 (complete historical exposure) and test/anomaly/Town01/change-weather/scenario-10 (partial historical exposure, 93 RGB frames) from BOTH candidate roles as SCHEMA_EXPOSED_DIAGNOSTIC_ONLY. No cross-town quarantine or additional IDs are inferred; reconcile unresolved/linked exposure evidence before allocation."
    write("test_partition_protocol.json", partition)
    readiness = read(OLD / "readiness.json")
    readiness["blockers"][0] = "Two exact known TEST scenario exclusions are established; historical file-level ledger remains incomplete and must be reconciled without inferring additional IDs"
    readiness["next_step"] = "Retain the two sealed exact exclusions; reconcile the incomplete historical file-level ledger from archival evidence without TEST access. Metadata/provenance/count auditing remains separately authorized and unexecuted. Current amendment stops after resealing."
    readiness["known_exact_exclusions"] = IDS
    readiness["exclusion_amendment_id"] = evidence["amendment_id"]
    write("readiness.json", readiness)

    report = (OLD / "report.md").read_text(encoding="utf-8")
    start, end = report.index("# Historical TEST Exposure\n"), report.index("# Proposed FINAL_CONFORMAL_CAL / FINAL_EVALUATION Split\n")
    section = """# Historical TEST Exposure

**Amendment 001 — pre-TEST-access historical-evidence correction.** The prior report incorrectly left the town unresolved and omitted partial scenario-10 exposure. Recovered preserved evidence supplied by the user establishes exactly:

- **`test/anomaly/Town01/change-weather/scenario-1` — complete historical exposure.** The main probe captured the scenario completely; earlier 64 KB / 8 MB probes also inspected its beginning.
- **`test/anomaly/Town01/change-weather/scenario-10` — partial historical exposure.** The same main probe continued into this scenario and reached **93 RGB frames**.

Both came from the official **`carlanomaly-base-test.tar.gz`** archive. The main probe requested inclusive bytes **0–125,829,119**, received **HTTP 206**, and used **sequential gzip/tar parsing**. Temporary probe files were deleted afterward. These are historical facts; no probe, payload or metadata inventory was reopened or executed in this amendment.

Both exact IDs are **SCHEMA_EXPOSED_DIAGNOSTIC_ONLY** and excluded as **entire scenarios from both FINAL_CONFORMAL_CAL and FINAL_EVALUATION**, regardless of complete versus partial exposure. These exact exclusions replace the provisional cross-town quarantine. No exposed ticks are used to allocate or rehabilitate remaining ticks within a scenario.

The **historical file-level ledger remains incomplete**. No evidence presently establishes any additional official TEST scenario IDs; none are inferred or claimed. The prior source documentation and synthetic fixtures are retained as contextual evidence, while `recovered_historical_evidence.json` records the authoritative recovered facts. `test_exposure_record.json` and `exposure_evidence.md` distinguish the two exposure extents and the remaining file-level limitation. Official metadata counts, provenance and actual final membership remain unaudited; no partition has occurred.

"""
    report = report[:start] + section + report[end:]
    anchor = "# Protocol Hashes\n\n"
    report = report.replace(anchor, anchor + "**Amendment history:** the complete original 26-file directory is preserved byte-for-byte in `amendment_history/pre_amendment_001/`, including its 24-artifact manifest and detached hash. Previous manifest SHA-256: `" + OLD_MANIFEST + "`. `amendment_record.json` documents this pre-TEST-access correction; `amendment_integrity.json` records the exact amended/new files and preservation checks. The current manifest also seals the archived original files, including their original seal files.\n\n", 1)
    report = report.replace("exact historical TEST exposure identities/completeness are unresolved", "the historical file-level ledger remains incomplete despite the two recovered exact exclusions")
    start = report.index("# Recommended Next Step\n")
    report = report[:start] + """# Recommended Next Step

Stop after resealing this protocol correction. Retain the two exact known exclusions; any later work must reconcile the historical file-level ledger without guessing additional IDs or reopening TEST. Official metadata/provenance/count auditing remains separately authorized and unexecuted. Scientific methods and all other design choices are unchanged. No TEST access, metadata inventory, partition, prediction, fitting, training, tuning, GNSS experiment, commit or push occurred in this amendment.
"""
    (HERE / "report.md").write_text(report, encoding="utf-8")

    # Make the report generator preserve the corrected evidence, and prevent
    # regeneration of the original scientific protocol after amendment sealing.
    generator = (OLD / "build_protocol.py").read_text(encoding="utf-8")
    generator = generator.replace("def main():\n", "def main():\n    if (HERE / 'amendment_record.json').exists():\n        raise RuntimeError('Amendment 001 is sealed; original generator cannot overwrite amended protocols. Use a documented amendment.')\n", 1)
    start = generator.index('    write("test_exposure_record.json",')
    end = generator.index('\n\n    write("test_partition_protocol.json",', start)
    generator = generator[:start] + "    import runpy\n    exposure_amendment = runpy.run_path(str(HERE / 'amend_protocol.py'))\n    write('test_exposure_record.json', exposure_amendment['corrected_exposure_record']())" + generator[end:]
    start = generator.index('    (HERE / "exposure_evidence.md").write_text(')
    end = generator.index('\n\n    # Inventory', start)
    generator = generator[:start] + "    (HERE / 'exposure_evidence.md').write_text(exposure_amendment['exposure_markdown'](), encoding='utf-8')" + generator[end:]
    (HERE / "build_protocol.py").write_text(generator, encoding="utf-8")

    write("amendment_record.json", {
        "amendment_id": evidence["amendment_id"], "type": "PRE_TEST_ACCESS_PROTOCOL_CORRECTION",
        "authorization": "Explicit user request; recovered historical evidence supplied by user",
        "previous_manifest_sha256": OLD_MANIFEST,
        "previous_preregistration_archive": "amendment_history/pre_amendment_001/",
        "original_files_archived": 26, "previous_sealed_artifacts": 24,
        "corrected_exposures": evidence["scenario_exposures"],
        "file_level_ledger_complete": False, "additional_official_TEST_scenario_ids_claimed": [],
        "scientific_methods_changed": False,
        "changes": "Historical exposure provenance/extent and exact whole-scenario exclusions; corresponding report/readiness/reproduction guard/static checks and seals only",
        "activity_attestation": {"official_TEST_payload_access": False, "official_metadata_inventory": False,
            "partition_execution": False, "prediction": False, "training_or_refitting": False,
            "conformal_fitting": False, "tuning": False, "secondary_GNSS": False,
            "generic_COGNIX_modification": False, "commit_or_push": False},
        "attestation_limit": "Executed tools/scripts in this amendment, not an OS-wide access trace; recovered probe facts describe earlier historical activity",
        "new_manifest_reference": "SHA256SUMS and SHA256SUMS.sha256; final digest recorded there and reported to user, avoiding a self-referential sealed hash",
        "exact_change_inventory_reference": "amendment_integrity.json"})


def verify_preservation():
    initial = read(HERE / "amendment_initial_state.json")
    audit = runpy.run_path(str(HERE / "audit_integrity.py"))
    current = audit["workspace"]()
    assert current == initial["workspace_sha256"], "Changes outside protocol directory"
    external = audit["external"]()
    assert external == initial["external_artifact_hashes"], "Frozen dependency changed"
    git = audit["git"]()
    assert git["HEAD"] == initial["git"]["HEAD"] and git["tracked_diff_sha256"] == initial["git"]["tracked_diff_sha256"]
    outside = lambda s: {line for line in s.splitlines() if "reports/carla_final_evaluation_preregistration_v1/" not in line}
    assert outside(git["status"]) == outside(initial["git"]["status"])
    assert all(sha(OLD / n) == h for n, h in initial["original_protocol_file_sha256"].items())
    changed = sorted(n for n, h in initial["original_protocol_file_sha256"].items()
                     if n not in {"SHA256SUMS", "SHA256SUMS.sha256"} and sha(HERE / n) != h)
    expected = ["build_protocol.py", "exposure_evidence.md", "protocol_validation.json", "readiness.json",
                "report.md", "test_exposure_record.json", "test_partition_protocol.json", "verify_protocol.py"]
    assert changed == sorted(expected), changed
    scientific = ["pipeline_map.json", "dependency_audit.json", "conformal_protocol.json", "final_metrics_spec.json",
                  "seed_aggregation_spec.json", "decision_semantics.json", "frozen_method_bindings.json", "live_demo_plan.md"]
    assert all(sha(HERE / n) == sha(OLD / n) for n in scientific)
    partition = read(HERE / "test_partition_protocol.json")
    old_partition = read(OLD / "test_partition_protocol.json")
    for k, v in old_partition.items():
        if k != "algorithm_only":
            assert partition[k] == v, k
    assert all(a == b for i, (a, b) in enumerate(zip(partition["algorithm_only"], old_partition["algorithm_only"])) if i not in (0, 2))
    new = sorted(p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()
                 and not p.is_relative_to(OLD) and p.relative_to(HERE).as_posix() not in initial["original_protocol_file_sha256"])
    write("amendment_integrity.json", {"passed": True, "old_manifest_sha256": OLD_MANIFEST,
        "original_archive_files_verified": 26, "workspace_files_unchanged_outside_protocol_directory": len(current),
        "frozen_external_artifacts_unchanged": True, "git_HEAD_and_tracked_diff_preserved": True,
        "git_status_outside_protocol_directory_preserved": True,
        "amended_original_artifacts": changed,
        "seal_files_regenerated": ["SHA256SUMS", "SHA256SUMS.sha256"],
        "new_artifacts": sorted(set(new + ["amendment_integrity.json"])),
        "history_archive": "amendment_history/pre_amendment_001/ (26 original files)",
        "scientific_artifacts_byte_identical": scientific,
        "partition_scientific_rule_unchanged": "Only exposure-reconciliation/exclusion prose and exact exclusion/hash fields changed; original seed, ratio, stratum, quotas, RNG, count gates and remaining algorithm steps preserved",
        "activity_attestation": read(HERE / "amendment_record.json")["activity_attestation"]})
    print(json.dumps({"amendment_preservation_passed": True, "amended_original_artifacts": changed,
                      "workspace_files_unchanged": len(current)}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["apply", "verify-preservation"])
    mode = parser.parse_args().mode
    if mode == "apply":
        apply_correction()
        print("Historical exposure correction applied; no metadata/TEST/scientific execution")
    else:
        verify_preservation()

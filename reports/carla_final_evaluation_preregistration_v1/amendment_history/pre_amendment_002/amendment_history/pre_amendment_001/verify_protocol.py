"""Static design checks and SHA256 sealing; no scientific imports or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REQUIRED = ["pipeline_map.json", "dependency_audit.json", "test_exposure_record.json",
            "test_partition_protocol.json", "conformal_protocol.json", "final_metrics_spec.json",
            "seed_aggregation_spec.json", "decision_semantics.json", "live_demo_plan.md", "report.md"]
HEADINGS = ["Frozen Development Findings", "End-to-End COGNIX Pipeline Map", "Graph/Fusion Relationship",
            "Calibration Dependency Audit", "Historical TEST Exposure",
            "Proposed FINAL_CONFORMAL_CAL / FINAL_EVALUATION Split", "Scenario / Temporal Exchangeability",
            "Conformal Construction", "Final Target Semantics", "Final Metrics", "Seed Aggregation Policy",
            "Decision / Attribution Semantics", "GNSS Disposition", "Live CARLA Demonstration Plan",
            "Protocol Hashes", "Ready / Not Ready for Official TEST Partition",
            "Ready / Not Ready for Conformal Fitting", "Recommended Next Step"]


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def files():
    return sorted(p for p in HERE.rglob("*") if p.is_file()
                  and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})


def static_checks():
    for name in REQUIRED:
        assert (HERE / name).is_file(), name
    for p in HERE.glob("*.json"):
        json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    bindings = read("frozen_method_bindings.json")
    seeds = [101, 202, 303, 404, 505]
    methods = ["nograph", "standard_gat", "epistemic_gat"]
    assert bindings["seeds"] == seeds and bindings["methods"] == methods
    assert {(x["method"], x["seed"]) for x in bindings["runs"]} == {(m, s) for m in methods for s in seeds}
    for run in bindings["runs"]:
        p = (ROOT / run["checkpoint"]).resolve()
        assert p.is_relative_to((ROOT / "reports/carla_gat_paired_raw_runs_v1").resolve())
        assert digest(p) == run["file_sha256"]
        assert 0 <= run["frozen_anomaly_threshold"] <= 1
    assert bindings["development_comparison"]["exact_two_sided_sign_flip_p"] == 0.875
    assert bindings["development_comparison"]["conditional_seed_CI"][0] < 0 < bindings["development_comparison"]["conditional_seed_CI"][1]
    cp = read("conformal_protocol.json")
    partition = read("test_partition_protocol.json")
    seed = read("seed_aggregation_spec.json")
    assert cp["alpha"] == 0.05 and cp["seeds"] == seeds and cp["methods"] == methods
    assert not cp["fitted"] and not cp["adapter_implemented"] and not cp["ready_for_fitting"]
    assert not partition["partition_performed"] and not partition["counts_known"] and not partition["readiness"]
    assert partition["seed"] == 2028 and 2028 not in partition["seed_distinct_from"]
    assert partition["candidate_fraction"] == {"FINAL_CONFORMAL_CAL": .2, "FINAL_EVALUATION": .8}
    # Pure integer rank/count algebra. No sample scores or conformal object exists.
    rank = lambda n: (19 * (n + 1) + 19) // 20
    assert rank(18) == 19 and rank(19) == 19 and rank(39) == 38
    assert (90 + 4) // 5 == 18 and (91 + 4) // 5 == 19
    assert (190 + 4) // 5 == 38 and (191 + 4) // 5 == 39
    assert not seed["ensemble_used"] and seed["no_best_seed"]
    exposure = read("test_exposure_record.json")
    assert exposure["town"] is None and exposure["exact_fully_qualified_official_scenario_ids_recovered"] == []
    assert "INCOMPLETE" in exposure["status"] and exposure["exact_partial_official_reference"] == "change-weather/scenario-1"
    assert read("pipeline_map.json")["graph_readout_is_collective_fusion"]
    assert read("pipeline_map.json")["extra_probability_fusion"] is None
    integrity = read("integrity_results.json")
    train = read("current_train_content_verification.json")
    assert integrity["passed"] and integrity["workspace_files_verified"] == 3764
    assert not integrity["changed_preexisting_files"] and not integrity["added_outside_protocol_directory"]
    assert not integrity["git_status_changes_outside_protocol_directory"]
    assert train["scenarios"] == 20 and train["files"] == 120100 and train["full_sha256_verified"] and not train["failures"]
    assert all(not v for v in integrity["activity_attestation"].values())
    report = (HERE / "report.md").read_text(encoding="utf-8")
    assert [line[2:] for line in report.splitlines() if line.startswith("# ")] == HEADINGS
    metrics = read("final_metrics_spec.json")
    assert set(metrics["binary_metrics"]) == {"AUROC", "AUPRC", "F1", "accuracy", "balanced_accuracy", "Brier", "BCE", "ECE"}
    assert "null" in metrics["undefined_primary_policy"] and "Secondary" in metrics["preregistered_descriptive_alternative"]
    for name, expected in read("source_evidence.json")["local_files"].items():
        assert digest(ROOT / name) == expected, name
    return {"passed": True, "required_artifacts": len(REQUIRED), "JSON_syntax": "all JSON parsed with nonfinite constants rejected",
            "method_seed_checkpoint_threshold_bindings": 15, "report_sections": 18,
            "rank_and_quota_algebra": "passed without scores/fitting/partition execution",
            "preservation_and_current_TRAIN_hash_evidence": "passed",
            "TEST_partition_ready": False, "conformal_fitting_ready": False,
            "checks_scope": "Static protocol/evidence consistency only; no science module imports, model load, inference, training, conformal fit or TEST payload reads"}


def verify_seal():
    manifest = HERE / "SHA256SUMS"
    assert digest(manifest) == (HERE / "SHA256SUMS.sha256").read_text().split()[0]
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        p = (HERE / rel).resolve()
        assert p.is_relative_to(HERE.resolve()) and rel not in seen
        assert digest(p) == expected, rel
        seen.add(rel)
    assert seen == {p.relative_to(HERE).as_posix() for p in files()}
    return {"sealed_files": len(seen), "manifest_sha256": digest(manifest), "all_seal_hashes_passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "seal", "verify"])
    mode = parser.parse_args().mode
    result = static_checks()
    if mode in {"check", "seal"}:
        (HERE / "protocol_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mode == "seal":
        manifest = HERE / "SHA256SUMS"
        manifest.write_text("".join(f"{digest(p)}  {p.relative_to(HERE).as_posix()}\n" for p in files()), encoding="utf-8")
        (HERE / "SHA256SUMS.sha256").write_text(f"{digest(manifest)}  SHA256SUMS\n", encoding="utf-8")
    if mode in {"seal", "verify"}:
        result.update(verify_seal())
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Separate preservation evidence for the runtime-calibration engineering task."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from audit_dataset_integration import digest, reports_inventory

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "docs/research/carla_runtime_calibration_preservation.json"
AUTHORIZED_UPDATES = {
    "cognix/engine/pipeline.py",
    "docs/research/CARLA_DATASET_PIPELINE_INTEGRATION.md",
    "docs/research/RESEARCH_DASHBOARD.md",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def audit():
    """Inspect metadata and restore existing states; never fit or decode source data."""
    import ast
    import importlib.metadata
    import platform
    import re
    sys.path.insert(0, str(ROOT))
    from cognix.adapters.carla.runtime_artifacts import (
        ORDER, DIMS, load_runtime_artifacts, authenticated, verify_unit,
    )

    ledger = []

    def read_json(relative):
        path = ROOT / relative
        ledger.append({"path": relative, "sha256": digest(path), "access": "metadata_only"})
        return json.loads(path.read_text(encoding="utf-8"))

    report = "reports/carla_train_full_v1/"
    graph_partition = read_json(report + "graph_partition_freeze_v1/partition.json")
    partition_path = report + "partition_freeze_v1/partition.json"
    upstream_bindings = graph_partition["upstream_bindings"]
    authenticated(ROOT / partition_path, upstream_bindings["original_partition_sha256"])
    verify_unit(ROOT / (report + "partition_freeze_v1"), upstream_bindings["original_partition_runtime_seal"])
    partition = read_json(partition_path)
    fit, cal = set(partition["FIT_NORMAL"]), set(partition["CAL_NORMAL"])
    assert len(fit) == 76 and len(cal) == 25 and not fit & cal and len(fit | cal) == 101
    assert set(graph_partition["GRAPH_TRAIN"]) | set(graph_partition["GRAPH_VAL"]) == fit
    assert set(graph_partition["excluded_CAL_NORMAL"]) == cal

    health = read_json(report + "gate2_train_health_v3/FINAL.json")
    agent_rows = []
    for name, dim in zip(ORDER, DIMS):
        prefix = report + "gate2_train_health_v3/units/"
        for role in ("fit-", "audit-"):
            verify_unit(ROOT / (prefix + role + name), health["committed_unit_seals"][role + name])
        evidence = read_json(prefix + "audit-" + name + "/audit.json")
        parents = set(evidence["CAL_parent_scenarios"])
        assert evidence["status"] == "VALID" and evidence["optimization_info"]["finite_optimum"] is True
        assert parents == cal and set(evidence["FIT_binding"]["FIT_NORMAL"]) == fit
        references = []
        for i in range(5):
            relative = prefix + f"fit-{name}/member_{i}.npz"
            references.append({"path": relative, "sha256": digest(ROOT / relative)})
        mapping = prefix + f"audit-{name}/audit_arrays.npz"
        agent_rows.append({"agent": name, "feature_dimension": dim,
                           "fit_unit_seal": health["committed_unit_seals"]["fit-" + name],
                           "probability_mapping_unit_seal": health["committed_unit_seals"]["audit-" + name],
                           "member_artifacts": references,
                           "probability_mapping_parameters_source": {"path": mapping, "sha256": digest(ROOT / mapping)},
                           "probability_mapping_class": evidence["calibrator_runtime_class"],
                           "probability_mapping_fitted_on_CAL": True,
                           "CAL_parent_scenarios": sorted(parents),
                           "CAL_overlap_count": len(parents & cal),
                           "untouched_CAL_parent_count": len(cal - parents),
                           "sample_counts": evidence["sample_counts"]})

    archive_path = Path(upstream_bindings["original_TRAIN_archive_local_identity"]["path"])
    assert archive_path.is_file(), "TRAIN archive missing"
    stat = archive_path.stat()
    fingerprint = {"bytes": stat.st_size, "ctime_ns": stat.st_ctime_ns, "mtime_ns": stat.st_mtime_ns,
                   "device": stat.st_dev, "inode": stat.st_ino}
    expected_fingerprint = upstream_bindings["archive_current_fingerprint"]
    assert fingerprint == expected_fingerprint, "TRAIN archive fingerprint differs from authenticated provenance"

    historical = []
    for version in (1, 2, 3, 4):
        prefix = f"reports/carla_final_conformal_cal_bundle_v{version}/"
        spec = read_json(prefix + "conformal_specification.json")
        registry = read_json(prefix + "model_registry.json")
        old_upstream = read_json(prefix + "upstream/manifest.json")
        runner_path = ROOT / (prefix + "runner.py")
        runner = runner_path.read_text(encoding="utf-8")
        ledger.append({"path": prefix + "runner.py", "sha256": digest(runner_path), "access": "code_only_not_executed"})
        old_graph = next(row for row in registry["scorers"] if row["method"] == "nograph" and row["seed"] == 101)
        assert "carlanomaly-base-test.tar.gz" in runner
        historical.append({"bundle": prefix.rstrip("/"), "compatible": False,
                           "source_archive_split": "TEST", "n_cal_scenarios": spec["n_cal"],
                           "score": spec["score"], "unit": "scenario maximum over synchronized tick scores",
                           "upstream_protocol": old_upstream["protocol_version"],
                           "upstream_package_agent_order": old_upstream["agent_order"],
                           "NoGraph_101_checkpoint": {k: old_graph[k] for k in ("checkpoint", "file_sha256", "selected_epoch")},
                           "rejection_reasons": ["TEST-derived official-label calibration, not TRAIN-only",
                                                 "different upstream artifacts and graph checkpoint",
                                                 "benchmark pooling, not current runtime epistemic-weighted fusion",
                                                 "class 0 normal/class 1 anomaly, not current constructed target ordering",
                                                 "scenario-max score, not current row-wise ConformalPredictor"]})

    _, _, provenance = load_runtime_artifacts(ROOT, "NoGraph", 101)
    critical_sources = ["cognix/engine/pipeline.py", "cognix/engine/decision_engine.py",
                        "cognix/engine/result.py", "cognix/core/types.py", "cognix/core/interfaces.py",
                        "cognix/config/schema.py", "cognix/runtime/bridge.py",
                        "cognix/adapters/carla/runtime_artifacts.py", "cognix/adapters/carla/real_agents.py",
                        "cognix/adapters/carla/normality.py", "cognix/adapters/carla/real_features.py",
                        "cognix/adapters/carla/replay_source.py", "cognix/belief/fusion.py",
                        "cognix/calibration/conformal.py", "cognix/artifacts/persistence.py",
                        "reports/carla_train_full_v1/kaggle_graph_training_bundle_v1/kaggle_models.py"]
    source_hashes = {name: digest(ROOT / name) for name in critical_sources}
    ledger.extend({"path": name, "sha256": sha, "access": "code_only"} for name, sha in source_hashes.items())
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "scipy", "scikit-learn", "pandas", "pyarrow", "pytest", "fastapi")}
    contract = {
        "schema_version": 1, "contract_id": "carla-runtime-NoGraph-seed101-audit-v1",
        "status": "BLOCKED_NO_INDEPENDENT_TRAIN_CALIBRATION_SOURCE",
        "executable_calibration_artifact": False, "configuration_role": "existing predeclared demonstration default",
        "agent_order": list(ORDER), "feature_dimensions": dict(zip(ORDER, DIMS)),
        "upstream_artifacts": agent_rows,
        "graph": {"strategy": "NoGraph", "seed": 101, "checkpoint_epoch": provenance["checkpoint_epoch"],
                  "checkpoint_sha256": provenance["checkpoint_sha256"], "artifact_id": provenance["artifact_id"],
                  "node_feature_order": ["prob_normal", "epistemic", "aleatoric"],
                  "node_dtype": "float32", "original_epistemic_dtype": "float64",
                  "node_probability": "clip(sigmoid(node_outputs[:, 0]), 1e-7, 1-1e-7)"},
        "fusion": {"implementation": "cognix.belief.fusion.EpistemicWeightedFusion", "epsilon": 1e-6,
                   "weight": "r_i/(e_i+epsilon); normalized by sum(raw_weights)+epsilon",
                   "collective_score": "clip(sum_i(normalized_weight_i * refined_p_i), 0, 1)",
                   "reliability": {name: 1.0 for name in ORDER},
                   "reliability_source": "explicit uniform runtime assumption; no compatible fitted artifact identified"},
        "conformal": {"implementation": "cognix.calibration.conformal.ConformalPredictor",
                      "input_shape": ["N", 2], "class_order": ["constructed_pseudo_or_non_normal", "constructed_normal"],
                      "input": "numpy.array([[1 - fused_confidence, fused_confidence]])",
                      "dtype": "float64", "nonconformity": "1 - input_probability[true_class]",
                      "alpha": 0.05, "target_coverage": 0.95,
                      "quantile": "ceil((n_cal+1)*(1-alpha))-th sorted score of {cal_scores, +infinity}",
                      "inclusive_comparison": True, "serialization": "existing sorted cal_scores .npy; n_cal restored from length",
                      "empty_sets_possible": True, "probability_magnitude_transformed": False,
                      "calibrated_confidence_field": "original fused confidence unchanged, with conformal metadata",
                      "n_cal": None, "fitted_quantile": None, "artifact_id": None, "artifact_seal": None,
                      "coverage_validity": "Not established: no independent CAL source; temporal/scenario exchangeability also needs a specified protocol"},
        "target_semantics": "P(target=normal) under TRAIN-derived constructed clean-vs-pseudo task",
        "partition": {"path": partition_path, "sha256": digest(ROOT / partition_path),
                      "FIT_count": len(fit), "CAL_count": len(cal), "CAL_membership": sorted(cal),
                      "untouched_CAL_scenarios": [], "upstream_fit_or_mapping_union_count": len(fit | cal)},
        "archive_identity": {"path": str(archive_path), "historical_sha256": upstream_bindings["original_TRAIN_archive_local_identity"]["sha256"],
                             "current_fingerprint": fingerprint, "fingerprint_matches_authenticated_history": True,
                             "archive_contents_opened_in_this_task": False, "fresh_full_archive_hash": False},
        "software": {"python": platform.python_version(), "interpreter": sys.executable, "platform": platform.platform(), **versions},
        "source_hashes": source_hashes,
        "separate_calibrator_required_per_configuration": True,
        "TEST_used_for_calibration": False, "TEST_labels_decoded": False,
    }
    contract_path = ROOT / "docs/research/carla_runtime_calibration_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    audit_path = ROOT / "docs/research/carla_runtime_calibration_audit.json"
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    current_ast = ast.parse((ROOT / "cognix/engine/pipeline.py").read_text(encoding="utf-8"))
    base_ast = ast.parse(subprocess.check_output(["git", "show", "HEAD:cognix/engine/pipeline.py"], cwd=ROOT, text=True))
    def methods(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CognixPipeline")
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
    current_methods, base_methods = methods(current_ast), methods(base_ast)
    changed = [name for name in current_methods if current_methods[name] != base_methods.get(name)]
    assert changed == ["run", "_estimate_uncertainties"]
    imported = [n.module for n in ast.walk(current_ast) if isinstance(n, ast.ImportFrom)]
    assert not any("carla" in str(name).lower() for name in imported)
    vocabulary = [word for word in ("CARLA", "CarlAnomaly", "Camera", "IMU", "Seg", "Town01", "anomaly_front")
                  if re.search(r"\b" + re.escape(word) + r"\b", (ROOT / "cognix/engine/pipeline.py").read_text())]
    assert not vocabulary
    record.update({"status": contract["status"], "current_pipeline_sha256": source_hashes["cognix/engine/pipeline.py"],
                   "changed_pipeline_methods_vs_base": changed, "CARLA_vocabulary_in_pipeline": vocabulary,
                   "core_fix_in_this_task": "Isolate optional method metadata errors from successful numeric UQ/graph execution",
                   "contract_path": contract_path.relative_to(ROOT).as_posix(), "contract_sha256": digest(contract_path),
                   "agent_calibration_parent_overlap": agent_rows, "historical_conformal_bundle_audit": historical,
                   "inputs_access_ledger": ledger,
                   "calibration_generation_started": False, "TRAIN_features_or_labels_decoded": False,
                   "TEST_sensor_or_label_payloads_decoded": False, "TEST_metrics_computed": False,
                   "new_model_fitting_or_threshold_tuning": False, "new_artifact_seal": None,
                   "stop_reason": "All 25 CAL scenarios fitted frozen upstream probability mappings; all remaining 76 TRAIN scenarios fitted upstream models. No independent scenario remains in the frozen 101-scenario TRAIN catalogue. Historical conformal bundles are TEST-derived and incompatible.",
                   "next_separate_task": "Specify untouched TRAIN-only calibration scenarios and a scenario-aware constructed-task protocol, or separately authorize a new runtime artifact training/partition task. Preserve the closed Experiment 2B chain."})
    audit_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "CAL_parent_overlaps": {r["agent"]: r["CAL_overlap_count"] for r in agent_rows},
                      "historical_bundles_rejected": len(historical), "contract_sha256": record["contract_sha256"],
                      "calibration_generation_started": False}))


def main():
    action = sys.argv[1]
    if action == "baseline":
        if RECEIPT.exists():
            raise RuntimeError("Refusing to overwrite the task baseline")
        modified = git("diff", "--name-only").splitlines()
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
        untracked.remove(Path(__file__).relative_to(ROOT).as_posix())
        files = {name: {"sha256": digest(ROOT / name), "bytes": (ROOT / name).stat().st_size}
                 for name in modified + untracked if not name.startswith("reports/")}
        value = {"interpreter": sys.executable, "git_status_before": git("status", "--short"),
                 "preexisting_modified_files": modified, "preexisting_untracked_files": untracked,
                 "preexisting_nonreport_files": files, "protected_reports_before": reports_inventory(),
                 "status": "BASELINE_RECORDED"}
        RECEIPT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": value["status"], "reports": value["protected_reports_before"],
                          "modified_count": len(modified), "untracked_count": len(untracked)}))
    elif action == "read":
        path = ROOT / sys.argv[2]
        lines = path.read_text(encoding="utf-8").splitlines()
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        end = int(sys.argv[4]) if len(sys.argv) > 4 else len(lines)
        print(f"{path} ({len(lines)} lines)")
        for i in range(start - 1, min(end, len(lines))):
            print(f"{i+1}: {lines[i]}")
    elif action == "json":
        value = json.loads((ROOT / sys.argv[2]).read_text(encoding="utf-8"))
        if len(sys.argv) > 3:
            for key in sys.argv[3].split("."):
                value = value[int(key)] if isinstance(value, list) else value[key]
        print(json.dumps(value, indent=2)[:18000])
    elif action == "keys":
        value = json.loads((ROOT / sys.argv[2]).read_text(encoding="utf-8"))
        print(json.dumps({k: (list(v) if isinstance(v, dict) else
                             {"count": len(v), "first": v[:1]} if isinstance(v, list) else v)
                          for k, v in value.items()}, indent=2)[:9000])
    elif action == "capture_core":
        path = ROOT / "docs/research/carla_runtime_calibration_audit.json"
        if path.exists():
            raise RuntimeError("Refusing to overwrite initial core audit evidence")
        base = subprocess.check_output(["git", "show", "HEAD:cognix/engine/pipeline.py"], cwd=ROOT)
        import hashlib
        value = {"status": "CORE_DIFF_CAPTURED_BEFORE_FURTHER_CHANGES", "interpreter": sys.executable,
                 "base_commit": git("rev-parse", "HEAD").strip(),
                 "base_pipeline_sha256": hashlib.sha256(base).hexdigest(),
                 "pre_task_pipeline_sha256": digest(ROOT / "cognix/engine/pipeline.py"),
                 "pre_task_pipeline_diff": git("diff", "--", "cognix/engine/pipeline.py")}
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in value.items() if not k.endswith("diff")}))
    elif action == "audit":
        audit()
    elif action == "verify":
        value = json.loads(RECEIPT.read_text(encoding="utf-8"))
        unchanged, authorized = [], []
        for name, expected in value["preexisting_nonreport_files"].items():
            current = digest(ROOT / name)
            if name in AUTHORIZED_UPDATES and current != expected["sha256"]:
                authorized.append({"path": name, "before_sha256": expected["sha256"], "after_sha256": current})
            else:
                assert current == expected["sha256"], f"Pre-existing work unexpectedly changed: {name}"
                unchanged.append(name)
        current_reports = reports_inventory()
        assert current_reports == value["protected_reports_before"], "Protected evidence changed"
        value.update({"protected_reports_after": current_reports, "authorized_integration_updates": authorized,
                      "preexisting_nonreport_files_unchanged": unchanged,
                      "status": "PASS_PROTECTED_REPORTS_AND_UNRELATED_PREEXISTING_WORK_UNCHANGED"})
        RECEIPT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": value["status"], "protected_reports": current_reports,
                          "authorized_updates": [v["path"] for v in authorized], "unchanged_nonreport_files": len(unchanged)}))
    elif action == "changes":
        value = json.loads(RECEIPT.read_text(encoding="utf-8"))
        current = git("ls-files", "--others", "--exclude-standard").splitlines()
        created = sorted(set(current) - set(value["preexisting_untracked_files"]))
        output = ROOT / "docs/research/carla_runtime_calibration_changes.json"
        created = sorted(set(created) | {output.relative_to(ROOT).as_posix()})
        updates = [{"path": name, "sha256_before": before["sha256"], "sha256_after": digest(ROOT / name)}
                   for name, before in value["preexisting_nonreport_files"].items()
                   if digest(ROOT / name) != before["sha256"]]
        result = {"task_modified_files": updates, "task_created_files": created,
                  "preexisting_modified_files": value["preexisting_modified_files"],
                  "preexisting_untracked_count": len(value["preexisting_untracked_files"]),
                  "git_status_short": git("status", "--short"), "git_diff_stat": git("diff", "--stat"),
                  "staged_files": git("diff", "--cached", "--name-only").splitlines()}
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"updates": updates, "created": created, "staged_files": result["staged_files"]}))
    elif action == "cleanup":
        import shutil
        target = (ROOT / ".runtime_calibration_work").resolve()
        baseline = json.loads(RECEIPT.read_text(encoding="utf-8"))
        assert target.parent == ROOT.resolve() and target.name == ".runtime_calibration_work"
        assert not any(n.startswith(".runtime_calibration_work/") for n in baseline["preexisting_untracked_files"])
        if target.exists():
            assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(target)
                       for p in target.rglob("*"))
            shutil.rmtree(target)
        print("Removed only task-created, checked workspace test fixtures")
    elif action == "validation":
        import xml.etree.ElementTree as ET
        xml_path = ROOT / "docs/research/carla_runtime_calibration_tests.xml"
        suites = ET.parse(xml_path).getroot().findall("testsuite")
        counts = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        assert counts == {"tests": 63, "failures": 0, "errors": 0, "skipped": 0}
        per_file = {}
        for suite in suites:
            for case in suite.findall("testcase"):
                name = case.get("classname")
                per_file[name] = per_file.get(name, 0) + 1
        check = subprocess.run(["git", "-c", "core.whitespace=cr-at-eol", "diff", "--check"],
                               cwd=ROOT, capture_output=True, text=True)
        assert check.returncode == 0, check.stdout + check.stderr
        preservation = json.loads(RECEIPT.read_text(encoding="utf-8"))
        assert preservation["status"] == "PASS_PROTECTED_REPORTS_AND_UNRELATED_PREEXISTING_WORK_UNCHANGED"
        prefix = str(sys.executable) + " -X utf8 -B"
        test_args = " -m pytest tests/unit/test_runtime_calibration_audit.py tests/unit/test_replay_integration.py tests/unit/test_dashboard.py tests/integration/test_full_pipeline_integration.py -q -p no:cacheprovider"
        value = {"status": "AUDIT_VALIDATED_CALIBRATION_STOPPED", "interpreter": sys.executable,
                 "audit_script_sha256": digest(Path(__file__)),
                 "new_test_source_sha256": digest(ROOT / "tests/unit/test_runtime_calibration_audit.py"),
                 "final_tests": {**counts, "passed": counts["tests"], "by_module": per_file,
                                 "junit_path": xml_path.relative_to(ROOT).as_posix(), "junit_sha256": digest(xml_path)},
                 "commands": [prefix + " docs/research/scripts/audit_runtime_calibration.py baseline",
                              prefix + " docs/research/scripts/audit_runtime_calibration.py capture_core",
                              prefix + " docs/research/scripts/audit_runtime_calibration.py audit",
                              prefix + test_args + " --basetemp=C:/Users/Aditya/Cognix2_recovered/.runtime_calibration_work/pytest-1",
                              prefix + test_args + " --basetemp=C:/Users/Aditya/Cognix2_recovered/.runtime_calibration_work/pytest-2 --junitxml=docs/research/carla_runtime_calibration_tests.xml",
                              prefix + " docs/research/scripts/audit_runtime_calibration.py cleanup",
                              prefix + " docs/research/scripts/audit_runtime_calibration.py verify",
                              "git -c core.whitespace=cr-at-eol diff --check"],
                 "initial_test_attempt": {"passed": 52, "failed": 11,
                    "causes": ["test comparison did not fix Monte Carlo Shapley permutations", "fixture used sorted_ids instead of sorted_canonical_ids"],
                    "resolution": "Fixed test-only RNG control and exact partition field; no runtime/scientific tuning"},
                 "existing_deprecation_warnings": 14, "git_diff_check_exit_code": check.returncode,
                 "calibration_generation_started": False, "runtime_calibration_artifact": None,
                 "n_cal": None, "quantile": None, "alpha_for_candidate_contract": .05, "artifact_seal": None,
                 "artifact_loader_or_seal_success_claimed": False,
                 "manual_real_replay": {"status": "NOT_RUN_CALIBRATION_NOT_FROZEN", "TEST_labels_decoded": False},
                 "TEST_metrics_computed": False, "TEST_used_for_calibration": False,
                 "research_or_runtime_model_training": False, "new_runtime_calibration_or_threshold_tuning": False,
                 "existing_integration_tests_fit_isolated_synthetic_fixtures": True,
                 "protected_reports": preservation["protected_reports_after"],
                 "previous_dataset_integration_receipts_unchanged": True,
                 "git_staged_files": git("diff", "--cached", "--name-only").splitlines()}
        output = ROOT / "docs/research/carla_runtime_calibration_validation.json"
        output.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": value["status"], "tests": counts, "by_module": per_file, "artifact": None}))
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()

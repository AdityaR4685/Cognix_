"""Design-only byte integrity audit. No scientific modules are imported.

External reads are restricted to explicit sealed upstream/graph files and the
20 TRAIN scenario paths in the frozen manifest. No TEST traversal or reads.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def workspace():
    roots = ["cognix", "reports", "artifacts", "results", "configs", "tests", "docs", "experiments", "dashboard", "robotics", "simulation"]
    paths = [p for name in roots for p in (ROOT / name).rglob("*")
             if p.is_file() and not p.is_relative_to(HERE)
             and "__pycache__" not in p.parts and "node_modules" not in p.parts]
    paths += [p for p in ROOT.iterdir() if p.is_file()]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}


def external():
    old = read(ROOT / "reports/carla_gat_paired_execution_v1/preservation_snapshot.json")
    entries = {}
    for group in ("upstream", "artifact"):
        for name, expected in old[group].items():
            p = ROOT / "reports/carla_gat_preregistration_v1/protocol.json" if name == "protocol.json" else Path(name)
            if not p.is_absolute():
                p = ROOT / p
            # Only exact historical files; refuse any TEST payload identity.
            assert "test" not in [x.lower() for x in p.parts], str(p)
            entries[str(p)] = {"expected": expected, "actual": digest(p), "group": group}
    return entries


def seal(directory, name="SHA256SUMS"):
    p = directory / name
    checks = []
    if p.with_name(name + ".sha256").exists():
        checks.append(digest(p) == p.with_name(name + ".sha256").read_text().split()[0])
    count = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        target = (directory / rel).resolve()
        assert target.is_relative_to(directory.resolve()), str(target)
        checks.append(digest(target) == expected)
        count += 1
    return {"path": str(p), "sha256": digest(p), "files": count, "passed": all(checks)}


def dataset(full_hash):
    manifest = read(ROOT / "reports/carla_train_expand20_v1/frozen_dataset_manifest.json")
    output, failures = {}, []
    for s in manifest["scenarios"]:
        base = Path(s["actual_path"]).resolve()
        assert "train" in base.parts and "test" not in [x.lower() for x in base.parts]
        assert base.name == s["scenario_id"].split("/")[-1]
        for rel, rec in s["source_files"].items():
            p = (base / rel).resolve()
            assert p.is_relative_to(base)
            st = p.stat()
            output[str(p)] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
            if st.st_size != rec["bytes"]:
                failures.append({"file": str(p), "reason": "size"})
            if full_hash and digest(p) != rec["sha256"]:
                failures.append({"file": str(p), "reason": "sha256"})
        print(json.dumps({"TRAIN_integrity_scenario": s["scenario_id"], "files": len(s["source_files"]), "hashes_checked": full_hash}), flush=True)
    assert len(manifest["scenarios"]) == 20
    return {"scenarios": 20, "files": len(output), "full_sha256_verified": full_hash,
            "manifest_sha256": digest(ROOT / "reports/carla_train_expand20_v1/frozen_dataset_manifest.json"),
            "failures": failures, "stats": output}


def git():
    return {"HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
            "status": subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["before", "after"])
    args = parser.parse_args()
    if args.mode == "before":
        assert not (HERE / "preservation_snapshot.json").exists()
        snap = {"time_utc": datetime.now(timezone.utc).isoformat(), "workspace_files": workspace(), "external": external(), "git": git()}
        historical = read(ROOT / "reports/carla_gat_paired_execution_v1/preservation_snapshot.json")
        snap["historical_changes"] = [n for n, h in historical["workspace_files"].items() if snap["workspace_files"].get(n) != h]
        snap["seals"] = [seal(ROOT / "reports/carla_gat_paired_runs_v1"),
                         seal(ROOT / "reports/carla_gat_paired_execution_v1", "local_delivery_SHA256SUMS"),
                         seal(ROOT / "reports/carla_gat_preregistration_v1"),
                         seal(ROOT / "reports/carla_kaggle_environment_lock_v1")]
        snap["seals"] += [seal(d) for d in sorted((ROOT / "reports/carla_gat_paired_raw_runs_v1").iterdir()) if d.is_dir()]
        write("preservation_snapshot.json", snap)
        print(json.dumps({"workspace_files_snapshotted": len(snap["workspace_files"]), "historical_changes": snap["historical_changes"], "seals_passed": all(s["passed"] for s in snap["seals"])}), flush=True)
        d = dataset(True)
        write("train_dataset_integrity.json", d)
        assert not d["failures"], "TRAIN source hashes changed"
        assert not snap["historical_changes"] and all(s["passed"] for s in snap["seals"])
        assert all(v["expected"] == v["actual"] for v in snap["external"].values())
    else:
        before = read(HERE / "preservation_snapshot.json")
        now, ex, g = workspace(), external(), git()
        changed = [n for n, h in before["workspace_files"].items() if now.get(n) != h]
        added = sorted(set(now) - set(before["workspace_files"]))
        prior_data = read(HERE / "train_dataset_integrity.json")
        now_data = dataset(False)
        data_changed = [n for n, v in prior_data["stats"].items() if now_data["stats"].get(n) != v]
        previous_status = set(before["git"]["status"].splitlines())
        current_status = set(g["status"].splitlines())
        outside_status_changes = sorted(x for x in previous_status ^ current_status if "reports/carla_final_evaluation_preregistration_v1/" not in x)
        passed = not changed and not added and not data_changed and not outside_status_changes and g["HEAD"] == before["git"]["HEAD"] and g["tracked_diff_sha256"] == before["git"]["tracked_diff_sha256"] and all(v["expected"] == v["actual"] for v in ex.values())
        result = {"passed": passed, "workspace_files_verified": len(now), "changed_preexisting_files": changed,
                  "added_outside_protocol_directory": added, "external": ex, "TRAIN_scenarios": 20,
                  "TRAIN_files_sha256_verified_against_frozen_manifest": prior_data["files"],
                  "TRAIN_post_audit_stat_changes": data_changed, "prior_seals": before["seals"],
                  "git_HEAD": g["HEAD"], "git_HEAD_and_tracked_diff_preserved": g["HEAD"] == before["git"]["HEAD"] and g["tracked_diff_sha256"] == before["git"]["tracked_diff_sha256"],
                  "git_status_changes_outside_protocol_directory": outside_status_changes,
                  "scope": "byte hashes of source/artifacts/reports/results; 20 TRAIN sources SHA256 against sealed manifest followed by unchanged size/mtime; no scientific imports or payload analysis",
                  "activity_attestation": {"TEST_payload_access": False, "TEST_partition_execution": False,
                      "new_graph_training": False, "conformal_fitting": False, "threshold_tuning": False,
                      "secondary_GNSS": False, "commit_or_push": False},
                  "attestation_limit": "Activity follows this milestone's executed commands, not an OS-wide file-access trace. Hashes alone do not prove absence of reads or historical pushes."}
        write("integrity_results.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("external", "prior_seals")}), flush=True)
        assert passed


if __name__ == "__main__":
    main()

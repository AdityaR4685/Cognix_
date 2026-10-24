"""Read-only integration audit and preservation receipts; no model execution."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "docs/research/carla_dataset_integration_preservation.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reports_inventory():
    records = {}
    directories = []
    for path in sorted((ROOT / "reports").rglob("*")):
        name = path.relative_to(ROOT).as_posix()
        if path.is_file():
            stat = path.stat()
            records[name] = {"sha256": digest(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        elif path.is_dir():
            directories.append(name)
    canonical = json.dumps({"files": records, "directories": directories}, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "file_count": len(records),
            "byte_count": sum(v["bytes"] for v in records.values()), "directory_count": len(directories)}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main():
    action = sys.argv[1]
    if action == "read":
        path = ROOT / sys.argv[2]
        lines = path.read_text(encoding="utf-8").splitlines()
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        end = int(sys.argv[4]) if len(sys.argv) > 4 else len(lines)
        print(f"{path} ({len(lines)} lines)")
        for i in range(start - 1, min(end, len(lines))):
            print(f"{i+1}: {lines[i]}")
    elif action == "baseline":
        if RECEIPT.exists():
            raise RuntimeError("Refusing to overwrite an existing preservation baseline")
        status = git("status", "--short")
        modified = git("diff", "--name-only").splitlines()
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
        untracked.remove(Path(__file__).relative_to(ROOT).as_posix())
        preserved = {name: {"sha256": digest(ROOT / name), "bytes": (ROOT / name).stat().st_size}
                     for name in modified + untracked if not name.startswith("reports/")}
        baseline = {"preexisting_git_status": status, "preexisting_modified_files": modified,
                    "preexisting_untracked_files": untracked, "preexisting_nonreport_files": preserved,
                    "protected_reports_before": reports_inventory(), "status": "BASELINE_RECORDED"}
        RECEIPT.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"baseline": str(RECEIPT), "protected_reports": baseline["protected_reports_before"],
                          "preexisting_modified_count": len(modified), "preexisting_untracked_count": len(untracked)}))
    elif action == "verify":
        baseline = json.loads(RECEIPT.read_text(encoding="utf-8"))
        for name, expected in baseline["preexisting_nonreport_files"].items():
            assert digest(ROOT / name) == expected["sha256"], f"Pre-existing work changed: {name}"
        current = reports_inventory()
        assert current == baseline["protected_reports_before"], "Protected report inventory changed"
        baseline["protected_reports_after"] = current
        baseline["status"] = "PASS_ALL_PROTECTED_REPORTS_AND_PREEXISTING_WORK_UNCHANGED"
        RECEIPT.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": baseline["status"], "protected_reports": current}))
    elif action == "environment":
        import importlib.util
        print(json.dumps({"python": sys.executable, "modules": {name: bool(importlib.util.find_spec(name))
              for name in ["numpy", "torch", "pytest", "fastapi", "uvicorn", "PIL", "httpx", "sklearn", "pandas", "pyarrow"]},
              "archives": {name: Path(name).is_file() for name in ["E:/carlanomaly-base-train.tar.gz", "E:/carlanomaly-base-test.tar.gz"]}}))
    elif action == "json":
        value = json.loads((ROOT / sys.argv[2]).read_text(encoding="utf-8"))
        print(str(value)[:6000])
    elif action == "hash":
        for name in sys.argv[2:]:
            print(name, digest(ROOT / name))
    elif action == "keys":
        value = json.loads((ROOT / sys.argv[2]).read_text())
        print({k: list(v)[:5] if isinstance(v, dict) else (v[:1] if isinstance(v, list) else v) for k, v in value.items()})
    elif action == "cleanup":
        import shutil
        target = (ROOT / ".integration_work").resolve()
        baseline = json.loads(RECEIPT.read_text())
        assert target.parent == ROOT.resolve() and target.name == ".integration_work"
        assert not any(n.startswith(".integration_work/") for n in baseline["preexisting_untracked_files"])
        if target.exists():
            assert all(not p.is_symlink() and p.resolve().is_relative_to(target) for p in target.rglob("*"))
            shutil.rmtree(target)
        print("Removed only task-created, verified workspace test fixtures")
    elif action == "changes":
        output = ROOT / "docs/research/carla_dataset_integration_changes.json"
        baseline = json.loads(RECEIPT.read_text())
        modified = sorted(set(git("diff", "--name-only").splitlines()) - set(baseline["preexisting_modified_files"]))
        created = set(git("ls-files", "--others", "--exclude-standard").splitlines()) - set(baseline["preexisting_untracked_files"])
        created.add(output.relative_to(ROOT).as_posix())
        result = {"task_modified_files": modified, "task_created_files": sorted(created),
                  "preserved_preexisting_modified_files": baseline["preexisting_modified_files"],
                  "preserved_preexisting_untracked_count": len(baseline["preexisting_untracked_files"]),
                  "git_diff_stat": git("diff", "--stat"), "git_status_short": git("status", "--short"),
                  "staged_files": git("diff", "--cached", "--name-only").splitlines()}
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"modified": modified, "created": sorted(created), "staged_files": result["staged_files"]}))


if __name__ == "__main__":
    main()

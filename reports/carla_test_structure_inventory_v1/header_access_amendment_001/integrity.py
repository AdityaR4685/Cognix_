"""Local metadata/source hash gates only. No archive or scientific operations."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent
ROOT = HERE.parents[2]
PROTOCOL = ROOT / "reports/carla_final_evaluation_preregistration_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    assert path.suffix.lower() not in {".gz", ".tar", ".zip", ".pt", ".npz", ".feather", ".jpg", ".png"}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def manifest_verify(directory, name):
    manifest = directory / name
    assert (directory / (name + ".sha256")).read_text().split() == [sha(manifest), name]
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        p = (directory / rel).resolve()
        assert p.is_relative_to(directory.resolve()) and rel not in seen
        assert sha(p) == digest, rel
        seen.add(rel)
    return len(seen)


def seal(directory, name, names):
    with (directory / name).open("x", encoding="utf-8", newline="\n") as f:
        for rel in sorted(names):
            f.write(sha(directory / rel) + "  " + rel + "\n")
    with (directory / (name + ".sha256")).open("x", encoding="utf-8", newline="\n") as f:
        f.write(sha(directory / name) + "  " + name + "\n")
    return manifest_verify(directory, name)


def protected_verify():
    bindings = read(HERE / "protected_bindings.json")
    for rel, digest in bindings["files"].items():
        p = (ROOT / rel).resolve()
        assert p.is_relative_to(ROOT) and sha(p) == digest, rel
    assert subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() == bindings["git_HEAD"]
    assert not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=no"], text=True).strip()
    for name in ("source_scenario_inventory.json", "eligible_scenario_inventory.json", "INVENTORY_SHA256SUMS"):
        assert not (PRIOR / name).exists(), name
    return {"passed": True, "protected_files_verified": len(bindings["files"]),
            "git_HEAD_unchanged": True, "tracked_worktree_clean": True,
            "Amendment_003_and_seed_receipt_unchanged": True, "prior_stopped_attempt_unchanged": True,
            "no_inventory_created": True, "verification_UTC": now()}


def amendment_verify():
    count = manifest_verify(HERE, "AMENDMENT_SHA256SUMS")
    assert read(HERE / "synthetic_test_results.json")["passed"]
    result = protected_verify()
    result.update(amendment_sealed_files_verified=count,
                  amendment_manifest_sha256=sha(HERE / "AMENDMENT_SHA256SUMS"))
    return result

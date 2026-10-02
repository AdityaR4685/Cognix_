"""Read-only packaging of frozen dependencies; never builds graph data or trains."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def closure(seeds):
    """Include unchanged package initializers and the static local import closure."""
    pending = list(seeds)
    found = set()
    def resolve(name):
        path = ROOT / name.replace(".", "/")
        if path.with_suffix(".py").is_file():
            pending.append(path.with_suffix(".py").relative_to(ROOT).as_posix())
        elif (path / "__init__.py").is_file():
            pending.append((path / "__init__.py").relative_to(ROOT).as_posix())
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        found.add(relative)
        path = ROOT / relative
        for parent in path.parents:
            if parent == ROOT:
                break
            init = parent / "__init__.py"
            if init.is_file():
                pending.append(init.relative_to(ROOT).as_posix())
        package = relative[:-3].replace("/", ".").split(".")[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("cognix"):
                        resolve(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = ".".join(package[:len(package) - node.level + 1])
                    name = ".".join(filter(None, (base, node.module)))
                else:
                    name = node.module or ""
                if name.startswith("cognix"):
                    resolve(name)
                    for alias in node.names:
                        resolve(name + "." + alias.name)
    return found


def main():
    from cognix.adapters.carla import graph_fit_export as ge
    from cognix.adapters.carla import graph_training as tr
    from cognix.adapters.carla import graph_training_data as gd
    artifact = Path(os.environ["TEMP"]) / "carla_graph_fit_export_v1"
    seal = ROOT / "reports/carla_gat_preregistration_v1"
    expected = json.loads((ROOT / "reports/carla_graph_trainers_v1/proposed_environment_lock_validated.json").read_text())
    ge.require(tr.trainer_identity() == expected["trainer_identity"], "frozen trainer identity mismatch")
    protocol, splits, upstream = ge.verify_frozen_dependencies(ROOT)
    data = gd.load_graph_dataset(artifact, seal)
    ge.require(data.arrays["node_features"].shape == (89970, 3, 3), "frozen graph shape")
    ge.require(len(data.train_indices) == 71976 and len(data.validation_indices) == 17994, "frozen split sizes")
    baseline = json.loads((ROOT / "reports/carla_graph_trainers_v1/initial_audit.json").read_text())
    ge.require(all(digest(ROOT / rel) == h for rel, h in baseline["repository_sha256"].items()), "previous milestone source changed")
    stage = HERE / "bundle"
    stage.mkdir(exist_ok=False)
    trainer = ["cognix/adapters/carla/" + name for name in expected["trainer_identity"]["source_sha256"]]
    sources = closure(trainer + list(protocol["bindings"]["generic_graph_source_sha256"]))
    files = set(sources)
    files.update("reports/carla_gat_preregistration_v1/" + line.split("  ", 1)[1]
                 for line in (seal / "SHA256SUMS").read_text().splitlines())
    files.update("reports/carla_gat_preregistration_v1/" + name for name in ("SHA256SUMS", "SHA256SUMS.sha256"))
    files.update("reports/carla_graph_trainers_v1/" + name for name in
                 ("train.py", "validate_environment.py", "proposed_environment_lock_validated.json", "future_commands.md"))
    files.update("reports/carla_kaggle_validation_bundle_v1/" + name for name in
                 ("validate_kaggle.py", "launch_full.py", "README.md"))
    files.add("LICENSE")
    records = {}
    for relative in sorted(files):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        records[relative] = digest(target)
    for source in sorted(artifact.iterdir()):
        ge.require(source.name in {"manifest.json", *json.loads((artifact / "manifest.json").read_text())["files"]}, "unexpected artifact file")
        target = stage / "graph_artifact" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[target.relative_to(stage).as_posix()] = digest(target)
    # Only definitions imported by frozen modules are included. No raw images/cache arrays.
    manifest = {"schema_version": 1, "files": records, "frozen_lock": expected,
                "graph_shape": [89970, 3, 3], "train_rows": 71976, "validation_rows": 17994,
                "node_order": list(ge.NODE_ORDER), "feature_order": list(ge.FEATURE_ORDER),
                "train_scenarios": splits["GRAPH_TRAIN"], "validation_scenarios": splits["GRAPH_VALIDATION"],
                "smoke_rows": 128, "smoke_seed": 101, "raw_data_packaged": False}
    write(stage / "bundle_manifest.json", manifest)
    records["bundle_manifest.json"] = digest(stage / "bundle_manifest.json")
    sums = "".join(h + "  " + name + "\n" for name, h in sorted(records.items()))
    (stage / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    sums_hash = digest(stage / "SHA256SUMS")
    (stage / "SHA256SUMS.sha256").write_text(sums_hash + "  SHA256SUMS\n", encoding="utf-8", newline="\n")
    archive = HERE / "cognix_kaggle_validation_v1.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for source in sorted(stage.rglob("*")):
            if source.is_file():
                info = zipfile.ZipInfo(source.relative_to(stage).as_posix(), (2026, 10, 2, 0, 0, 0))
                z.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    write(HERE / "preparation_results.json", {"status": "PREPARED_KAGGLE_UNVERIFIED",
          "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
          "bundle_SHA256SUMS_sha256": sums_hash, "file_count": len(records) + 2,
          "source_file_count": len(sources), "frozen_hashes_verified": expected["expected_hashes"],
          "trainer_identity": tr.trainer_identity(), "upstream_hashes_verified": upstream,
          "regression_baseline": "608 passed (previous validated milestone; not rerun here)",
          "no_raw_data": True, "no_full_training": True, "no_TEST": True,
          "no_conformal_fit": True, "no_scientific_aggregation": True})
    protected = {p.relative_to(ROOT).as_posix(): digest(p) for directory in ("cognix", "tests", "reports")
                 for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and HERE not in p.parents and ROOT / "reports/carla_kaggle_environment_lock_v1" not in p.parents}
    write(HERE / "preservation_snapshot.json", {"files": protected,
          "upstream": upstream, "artifact": {str(p): digest(p) for p in artifact.iterdir() if p.is_file()},
          "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "bundle_seal": sums_hash, "files": len(records) + 2, "CUDA_executed": False}))


if __name__ == "__main__":
    main()

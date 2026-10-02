"""Future-only full launcher. Default action is preflight and command printing."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from validate_kaggle import ROOT, actual_environment, digest, integrity, modules, read, require


def preflight(path):
    path = Path(path).resolve()
    require(path.is_file(), "validated environment lock required")
    checksum = path.with_name(path.name + ".sha256")
    require(checksum.is_file() and digest(path) == checksum.read_text().split()[0], "execution lock seal mismatch")
    lock = read(path)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == lock["validated_execution_environment"].get("CUDA_VISIBLE_DEVICES"),
            "validated GPU exposure mismatch; use the same authorized GPU-0 isolation")
    require(lock.get("status") == "ARTIFACT_RUNTIME_AND_CUDA_FIXTURES_VALIDATED"
            and lock.get("CUDA_smoke_validation_passed") is True, "all actual CUDA gates must pass")
    for name, expected in lock["evidence_sha256"].items():
        evidence = (path.parent / name).resolve()
        require(evidence.is_relative_to(path.parent) and digest(evidence) == expected, "execution evidence mismatch: " + name)
        require(read(evidence).get("passed") is True, "failed execution evidence: " + name)
    hashes = integrity()
    require(hashes["bundle_SHA256SUMS_sha256"] == lock["bundle_SHA256SUMS_sha256"], "validated source bundle changed")
    require(hashes["expected_hashes"] == lock["expected_hashes"] and hashes["trainer_identity"] == lock["trainer_identity"], "frozen identities differ")
    environment = actual_environment("cuda")
    expected = lock["validated_execution_environment"]
    for key in ("Python", "PyTorch", "NumPy", "torch_CUDA_runtime", "cuDNN_version", "devices"):
        require(environment[key] == expected[key], "validated runtime mismatch: " + key)
    # Physical GPU UUID can change between sessions; model/capability/memory must match.
    for key in ("name", "major", "minor", "total_memory", "multi_processor_count"):
        require(environment["device_properties"][0][key] == expected["device_properties"][0][key], "GPU specification mismatch: " + key)
    np, torch, F, ge, tr, gd, gm = modules()
    tr.verify_environment_lock(lock, "cuda")
    return lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("nograph", "standard_gat", "epistemic_gat"))
    parser.add_argument("--seed", type=int, choices=(101, 202, 303, 404, 505))
    parser.add_argument("--execute-full-experiment", action="store_true", help="future separately authorized milestone only")
    args = parser.parse_args()
    require((args.method is None) == (args.seed is None), "specify both method and seed, or neither for all 15 runs")
    preflight(args.environment_lock)
    methods = (args.method,) if args.method else ("nograph", "standard_gat", "epistemic_gat")
    seeds = (args.seed,) if args.seed else (101, 202, 303, 404, 505)
    plan = [{"method": method, "seed": seed, "output": str(args.output / method / ("seed_" + str(seed)))}
            for seed in seeds for method in methods]
    print(json.dumps({"runs": plan, "all_planned_runs_retained": True, "replace_failed_seed": False,
                      "select_best_seed": False, "execution_requested": args.execute_full_experiment}, indent=2))
    if not args.execute_full_experiment:
        return 0
    command = [sys.executable, str(ROOT / "reports/carla_graph_trainers_v1/train.py")]
    command += ["train", "--method", args.method, "--seed", str(args.seed)] if args.method else ["all"]
    command += ["--execute-full", "--device", "cuda", "--artifact", str(ROOT / "graph_artifact"),
                "--preregistration", str(ROOT / "reports/carla_gat_preregistration_v1"),
                "--environment-lock", str(args.environment_lock.resolve()), "--output", str(args.output.resolve())]
    print(shlex.join(command), flush=True)
    # Frozen orchestration retains every planned run, failures and outputs, without seed replacement.
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())

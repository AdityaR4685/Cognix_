"""Future train/all commands; this milestone invokes only --smoke."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla.graph_training_data import METHODS, SEEDS, load_graph_dataset, make_smoke_fixture, SPLIT_SHA256
from cognix.adapters.carla.graph_training import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "all"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute-full", action="store_true", help="future explicit execution gate; never used in this milestone")
    parser.add_argument("--artifact", type=Path, default=Path(os.environ.get("TEMP", "/tmp")) / "carla_graph_fit_export_v1")
    parser.add_argument("--preregistration", type=Path, default=ROOT / "reports/carla_gat_preregistration_v1")
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention-audit", action="store_true")
    args = parser.parse_args()
    ge.require(args.smoke != args.execute_full, "choose smoke fixtures or explicitly authorized future full execution")
    ge.require(args.command != "train" or args.method is not None, "train requires --method")
    lock = None
    if args.smoke:
        ge.require(ge.sha256_file(args.preregistration / "protocol.json") == ge.PROTOCOL_SHA256
                   and ge.sha256_file(args.preregistration / "split_manifest.json") == SPLIT_SHA256, "smoke protocol binding")
        splits = json.loads((args.preregistration / "split_manifest.json").read_text())
        dataset = make_smoke_fixture(splits)
    else:
        ge.require(args.environment_lock is not None, "future full execution needs a validated environment lock")
        # Integrity verification precedes every future run and model construction.
        dataset = load_graph_dataset(args.artifact, args.preregistration)
        lock = json.loads(args.environment_lock.read_text())
    methods = METHODS if args.command == "all" else (args.method,)
    seeds = (101,) if args.smoke and args.command == "all" else SEEDS if args.command == "all" else (args.seed,)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = [{"method": method, "seed": seed} for seed in seeds for method in methods]
    ge.save_json(args.output / "plan.json", {"runs": plan, "smoke": args.smoke,
                 "all_planned_seeds_retained": True, "replace_failed_seeds": False, "select_best_seed": False})
    runs = []
    for item in plan:
        directory = args.output / item["method"] / f"seed_{item['seed']}"
        try:
            summary = train_model(dataset, item["method"], item["seed"], directory,
                                  execute_full=args.execute_full, environment_lock=lock,
                                  device=args.device, audit_attention=args.attention_audit)
            runs.append({**item, "status": "complete", "output": str(directory), "selected_epoch": summary["selected_epoch"]})
        except Exception as exc:
            # Record this planned failure; do not delete, retry/replace the seed or select another.
            runs.append({**item, "status": "failed", "error": repr(exc), "output": str(directory)})
    result = {"complete": all(r["status"] == "complete" for r in runs), "runs": runs,
              "smoke": args.smoke, "interpretation": "engineering fixtures only" if args.smoke else "internal graph development"}
    ge.save_json(args.output / "orchestration.json", result)
    print(json.dumps(result))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

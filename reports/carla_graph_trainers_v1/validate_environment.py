"""Print actual future execution versions and validate data before training.

Run this on Kaggle only in a future authorized execution milestone. It never
uploads data or starts training. Local mode emits the proposed lock instead.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cognix.adapters.carla import graph_fit_export as ge
from cognix.adapters.carla.graph_training_data import load_graph_dataset, make_smoke_fixture, METHODS
from cognix.adapters.carla.graph_training import proposed_environment_lock, configure_determinism, environment_record, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "reports/carla_gat_preregistration_v1")
    parser.add_argument("--validate-kaggle", action="store_true")
    parser.add_argument("--cuda-smoke-output", type=Path,
                        help="future Kaggle-only deterministic fixture repeats; never used in this local milestone")
    args = parser.parse_args()
    lock = proposed_environment_lock()
    if args.validate_kaggle:
        ge.require(args.artifact is not None, "artifact path is required")
        load_graph_dataset(args.artifact, args.preregistration)
        configure_determinism(0)
        actual = environment_record()
        ge.require(actual["CUDA_available"] and len(actual["devices"]) == 1 and "T4" in actual["devices"][0], "single T4 is required")
        # Observed versions, never invented/presumed. Future review locks this file.
        lock["Kaggle_observed_versions"] = actual
        lock["validated_execution_environment"] = actual
        lock["status"] = "ARTIFACT_AND_RUNTIME_VALIDATED_CUDA_KERNEL_SMOKE_REQUIRED"
        lock["CUDA_smoke_validation_passed"] = False
        if args.cuda_smoke_output is not None:
            dataset = make_smoke_fixture(json.loads((args.preregistration / "split_manifest.json").read_text()))
            args.cuda_smoke_output.mkdir(parents=True, exist_ok=False)
            evidence = {}
            for method in METHODS:
                first = train_model(dataset, method, 101, args.cuda_smoke_output / method / "first", device="cuda")
                repeat = train_model(dataset, method, 101, args.cuda_smoke_output / method / "repeat", device="cuda")
                ge.require(first["checkpoint_content_sha256"] == repeat["checkpoint_content_sha256"]
                           and first["prediction_content_sha256"] == repeat["prediction_content_sha256"],
                           "CUDA fixture nondeterminism; stop without fallback")
                evidence[method] = {"checkpoint_content_sha256": first["checkpoint_content_sha256"],
                                    "prediction_content_sha256": first["prediction_content_sha256"], "exact_repeat": True}
            lock["CUDA_smoke_validation_passed"] = True
            lock["CUDA_smoke_evidence"] = evidence
            lock["status"] = "ARTIFACT_RUNTIME_AND_CUDA_FIXTURES_VALIDATED"
    ge.save_json(args.output, lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

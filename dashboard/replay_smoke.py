"""Bounded real replay engineering check; no training, labels or metric suite."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cognix import DecisionEngine
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.adapters.carla.replay_source import CarlAnomalyReplaySource
from cognix.adapters.carla.runtime_artifacts import load_runtime_artifacts
from cognix.runtime.replay import ReplayController
from cognix.runtime.bridge import RuntimeBridge
from dashboard.payload import build_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--split", choices=["TRAIN", "TEST"], default="TEST")
    parser.add_argument("--scenario")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    source = CarlAnomalyReplaySource.configured(ROOT, args.split, args.archive, args.max_samples)
    controller = ReplayController(source)
    controller.select(args.scenario or source.sequences()[0])
    observation = controller.step()
    checks = {}
    for method in ("NoGraph", "StandardGAT", "EpistemicGAT"):
        agents, graph, provenance = load_runtime_artifacts(ROOT, method, 101)
        engine = DecisionEngine(graph=graph, belief=EpistemicWeightedFusion(),
                                attribution=EpistemicShapley(), mode="research")
        bridge = RuntimeBridge(engine, agents, provenance)
        payload = build_payload(bridge.run(observation))
        assert payload["mode"] == "DATASET" and payload["active_agents"] == ["Camera", "IMU", "Seg"]
        assert not payload["calibration_available"] and payload["conformal_set"] is None
        assert len(payload["shapley"]) == 3 and payload["graph_executed"]
        assert payload["module_status"]["uq"]["method"] == "bootstrap_oneclass_ensemble"
        assert not payload["provenance"]["label_visible_to_inference"]
        json.dumps(payload, allow_nan=False)
        checks[method] = payload
        print(json.dumps({"method": method, "sample": observation.sample_id,
                          "agents": payload["active_agents"], "decision": payload["decision"],
                          "probability": payload["confidence"], "graph": payload["graph_executed"],
                          "communication": payload["graph_communication_executed"],
                          "calibration": payload["calibration_available"], "latency": payload["latency"]}))
    second = controller.step()
    previous = controller.step(-1)
    assert previous.sample_id == observation.sample_id
    receipt = {"status": "PASS_SUPPORTED_PARTIAL_CHAIN", "purpose": "engineering replay, not scientific evaluation",
               "selection": "first catalog scenario and seed 101; all three configured strategies, no result selection",
               "scenario": observation.sequence_id, "first_sample": observation.sample_id,
               "next_sample": second.sample_id, "previous_sample": previous.sample_id,
               "training": False, "threshold_tuning": False, "recalibration": False,
               "official_label_payloads_opened": False, "test_metrics_computed": False, "checks": checks}
    if args.output:
        Path(args.output).write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

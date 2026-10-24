"""Application composition boundary for the currently installed domain adapters."""
import json
from pathlib import Path
import numpy as np
from cognix import DecisionEngine, CognixConfig
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.agents import CameraAgent, DepthAgent, LiDARAgent, GNSSAgent, IMUAgent, SegAgent
from cognix.adapters.carla.replay_source import SyntheticReplaySource, CarlAnomalyReplaySource
from cognix.adapters.carla.runtime_artifacts import load_runtime_artifacts
from cognix.runtime.replay import ReplayController
from cognix.runtime.bridge import RuntimeBridge


def benchmark_reference(root):
    path = Path(root) / "results/universal_evaluation/summaries/all_summaries.json"
    if not path.is_file():
        return {}, "Benchmark summaries unavailable (Illustrative profiles)"
    data = json.loads(path.read_text())
    table = {}
    for name, models in data.items():
        entry = models.get("EpistemicGAT")
        if entry:
            def metric(key, scale=1):
                mean = entry.get(key, {}).get("mean")
                return float(mean) * scale if mean is not None else None
            table[name] = {"ece": metric("ece"), "cov": metric("cp_coverage", 100),
                           "epi": metric("mean_epistemic"), "esc": metric("escalation_rate", 100)}
    return table, "Persisted universal benchmark; illustrative synthetic reference profiles"


def synthetic_runtime(root):
    from cognix.artifacts import load_synthetic_artifacts
    graph = calibrator = None
    cal_reason = None
    graph_reason = None
    load_error = None
    try:
        graph, calibrator, _ = load_synthetic_artifacts(Path(root) / "artifacts/synthetic", graph_type="EpistemicGAT")
    except Exception as exc:
        load_error = str(exc)
        print(f"Synthetic artifacts unavailable: {load_error}")
        exc_lower = load_error.lower()
        if "conformal" in exc_lower or "calibrat" in exc_lower:
            cal_reason = f"Synthetic collective conformal calibrator unavailable: {load_error}"
            graph_reason = "Synthetic graph artifact unavailable due to bundle loading failure"
        elif "gat" in exc_lower or "graph" in exc_lower:
            graph_reason = f"Synthetic graph artifact unavailable: {load_error}"
            cal_reason = (
                f"Synthetic artifact bundle could not be loaded (graph artifact failed: {load_error}); "
                "collective conformal calibration availability could not be established"
            )
        else:
            graph_reason = f"Synthetic artifact bundle could not be loaded: {load_error}"
            cal_reason = (
                f"Synthetic artifact bundle could not be loaded ({load_error}); "
                "collective conformal calibration availability could not be established"
            )
    if calibrator is None and cal_reason is None:
        cal_reason = "No collective conformal calibrator bound to synthetic runtime"
    if graph is None and graph_reason is None:
        graph_reason = "No graph model bound to synthetic runtime"

    engine = DecisionEngine(config=CognixConfig(), graph=graph, calibrator=calibrator,
                            attribution=EpistemicShapley(), mode="production")
    dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=1, seed=42)
    agents = [CameraAgent(), DepthAgent(), LiDARAgent(), GNSSAgent(), IMUAgent(), SegAgent()]
    for index, agent in enumerate(agents):
        agent.rng = np.random.default_rng(42 + index)
    comparisons, note = benchmark_reference(root)
    provenance = {"artifact_id": "synthetic demonstration artifacts",
                  "graph_strategy": "EpistemicGAT" if graph else "Unavailable", "seed": 42,
                  "uq_method": "synthetic probability-noise simulation",
                  "probability_label": "Synthetic collective ACT probability",
                  "comparison_table": comparisons, "comparison_table_note": note,
                  "decision_semantics": "Synthetic framework policy demonstration"}
    if cal_reason:
        provenance["calibration_unavailable_reason"] = cal_reason
    if graph_reason:
        provenance["graph_unavailable_reason"] = graph_reason
    if load_error:
        provenance["artifact_load_error"] = load_error

    bridge = RuntimeBridge(engine, agents, provenance)
    controller = ReplayController(SyntheticReplaySource(dataset))
    controller.select("NORMAL")
    controller.playing = True
    return engine, dataset, agents, bridge, controller


def dataset_runtime(root, request):
    source = CarlAnomalyReplaySource.configured(root, request.split.upper(), request.archive, request.max_samples)
    agents, graph, provenance = load_runtime_artifacts(Path(root), request.graph_strategy, request.seed, request.require_calibration)
    engine = DecisionEngine(config=CognixConfig(), graph=graph, belief=EpistemicWeightedFusion(),
                            attribution=EpistemicShapley(), mode="research")
    bridge = RuntimeBridge(engine, agents, provenance)
    return engine, None, agents, bridge, ReplayController(source)

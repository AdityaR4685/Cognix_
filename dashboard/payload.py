"""Domain-neutral DecisionResult presentation. No dataset reading or inference."""
import copy
import time
import numpy as np
from cognix.runtime.replay import plain

CONFORMAL_CLASS_MAP = {0: "ESCALATE", 1: "ACT"}


def format_conformal_set(prediction_set):
    if prediction_set is None:
        return None
    mapped = []
    for item in prediction_set:
        if isinstance(item, bool):
            raise ValueError(f"Unexpected conformal prediction set class index: {item!r}")
        try:
            idx = int(item)
            if idx in CONFORMAL_CLASS_MAP:
                mapped.append(CONFORMAL_CLASS_MAP[idx])
                continue
        except (ValueError, TypeError):
            pass
        if isinstance(item, str) and item in CONFORMAL_CLASS_MAP.values():
            mapped.append(item)
        else:
            raise ValueError(f"Unexpected conformal prediction set class index: {item!r}")
    return mapped


def build_payload(cycle, latency_history=(), annotation=None):
    observation, result = cycle.observation, cycle.result
    source = plain(observation.provenance)
    runtime = plain(cycle.runtime_provenance)
    provenance = {**source, **runtime, "source_id": observation.source_id,
                  "domain": observation.domain_id, "sequence_id": observation.sequence_id,
                  "sample_id": observation.sample_id, "label_visible_to_inference": False}
    status = copy.deepcopy(result.metadata.get("module_status", {}))
    calibrated = bool(status.get("calibration", {}).get("executed"))
    graph = bool(status.get("gnn", {}).get("executed"))
    strategy = runtime.get("graph_strategy", "Unavailable")
    communication = bool(graph and (result.communication_statistics or {}).get("edges"))
    unavailable = runtime.get("calibration_unavailable_reason") if not calibrated else None
    methods = {"uq": runtime.get("uq_method", status.get("uq", {}).get("method")),
               "gnn": strategy, "belief": runtime.get("fusion", status.get("belief", {}).get("method")),
               "attribution": "collective epistemic uncertainty Shapley"}
    agents = []
    for agent in cycle.agents:
        uq = agent._uncertainty
        prediction = result.agent_predictions.get(agent.agent_id)
        prediction = float(prediction.value if hasattr(prediction, "value") else prediction)
        agents.append({"name": agent.agent_id, "prediction": prediction,
                       "upstream_prediction": float(agent._prediction.value),
                       "epistemic": float(uq.epistemic), "aleatoric": float(uq.aleatoric),
                       "weight": result.agent_trust_weights.get(agent.agent_id),
                       "healthy": observation.metadata.get("agent_health", {}).get(agent.agent_id)})
    samples = list(latency_history) + [cycle.runtime_ms]
    latencies = dict(result.latency_ms)
    latencies.update({"source_read_ms": observation.metadata.get("source_read_ms"),
                     "agent_inference_ms": cycle.agent_inference_ms, "runtime_ms": cycle.runtime_ms,
                     "total": cycle.runtime_ms, "uncertainty": cycle.agent_inference_ms,
                     "fusion": result.latency_ms.get("belief_fusion"),
                     "calibration": result.latency_ms.get("calibration") if calibrated else None,
                     "decision": result.latency_ms.get("decision"),
                     "p50": float(np.percentile(samples, 50)), "p95": float(np.percentile(samples, 95)),
                     "p99": float(np.percentile(samples, 99)), "measured_timer": "time.perf_counter"})
    if not graph:
        latencies["graph_refinement"] = None
    if not status.get("attribution", {}).get("executed"):
        latencies["attribution"] = None
    raw_prediction_set = (result.calibration_metrics or {}).get("prediction_set") if calibrated else None
    prediction_set = format_conformal_set(raw_prediction_set)
    trace = [f"{key}: {'executed' if value.get('executed') else 'unavailable'}"
             + (f" ({value['failure_reason']})" if value.get("failure_reason") else "")
             for key, value in status.items()]
    if unavailable:
        trace.append("Calibration unavailable: " + unavailable)
    payload = {"timestamp": time.time(), "observation_timestamp": observation.timestamp,
               "tick": observation.metadata.get("tick", observation.sample_id),
               "sample_id": observation.sample_id, "scenario": observation.sequence_id,
               "mode": source.get("mode", "UNKNOWN"), "domain": observation.domain_id,
               "source_id": observation.source_id, "active_agents": [a["name"] for a in agents],
               "scenario_details": plain(observation.metadata.get("scenario_details", {"severity": None, "affected": "Unavailable", "expected": None})),
               "research_meta": {"data_source": observation.source_id, "dataset": source.get("dataset"),
                                 "experiment": "Engineering replay demonstration", "run_id": runtime["run_id"],
                                 "model": runtime.get("artifact_id", "Configured runtime"), "seed": runtime.get("seed")},
               "provenance": provenance, "probability_label": runtime.get("probability_label", "Collective probability"),
               "prediction_semantics": runtime.get("prediction_semantics"),
               "decision_semantics": runtime.get("decision_semantics"),
               "comparison_table": runtime.get("comparison_table", {}),
               "comparison_table_note": runtime.get("comparison_table_note", "No scientific metrics computed during replay"),
               "decision": result.decision.value, "risk_level": result.risk_level.value,
               "confidence": result.confidence, "collective_probability": result.confidence,
               "calibrated_confidence": result.calibrated_confidence if calibrated else None,
               "calibration_available": calibrated, "calibration_unavailable_reason": unavailable,
               "conformal_set": prediction_set, "conformal_target_coverage": (result.calibration_metrics or {}).get("target_coverage") if calibrated else None,
               "graph_executed": graph, "graph_strategy": strategy,
               "graph_communication_executed": communication,
               "communication_statistics": result.communication_statistics if communication else None,
               "epistemic": result.epistemic_uncertainty, "aleatoric": result.aleatoric_uncertainty,
               "total_unc": result.total_uncertainty, "dominant_unc": result.dominant_uncertainty_source.upper(),
               "escalation": result.escalation_required, "abstained": result.abstained,
               "requested_information": result.requested_information,
               "agents": agents, "shapley": result.agent_contributions,
               "top_agent": max(result.agent_trust_weights, key=result.agent_trust_weights.get, default=None),
               "module_status": status, "module_methods": methods,
               "module_failure_reasons": {k: v.get("failure_reason") for k, v in status.items()},
               "latency": latencies, "decision_trace": trace,
               "reasoning": result.explanation, "explanation": result.explanation,
               "ground_truth": None}
    if unavailable:
        payload["module_failure_reasons"]["calibration"] = unavailable
    # Inference has completed before optional annotation is even requested.
    if annotation is not None:
        payload["ground_truth"] = {"kind": "post-inference ground-truth display annotation",
                                   "value": plain(annotation() if callable(annotation) else annotation),
                                   "used_for_inference": False}
    return plain(payload)

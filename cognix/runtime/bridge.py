"""A strict observation boundary around the existing DecisionEngine."""
from dataclasses import dataclass
from typing import Any
import math
import time
from uuid import uuid4
from cognix.runtime.replay import ObservationEnvelope, SourceError, plain


class CachedAgent:
    def __init__(self, agent, inputs):
        self.agent_id = agent.agent_id
        if getattr(agent, "healthy", True) is not True:
            raise SourceError(f"Agent unavailable: {self.agent_id}")
        self._prediction = agent.predict(inputs)
        self._uncertainty = agent.estimate_uncertainty(inputs)
        values = [self._prediction.value, self._uncertainty.epistemic,
                  self._uncertainty.aleatoric, self._uncertainty.total]
        if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in values):
            raise SourceError(f"Invalid prediction/uncertainty: {self.agent_id}")
        self._metadata = agent.metadata()

    def predict(self, inputs):
        return self._prediction

    def estimate_uncertainty(self, inputs):
        return self._uncertainty

    def metadata(self):
        return self._metadata


@dataclass
class RuntimeCycle:
    observation: ObservationEnvelope
    result: Any
    agents: list[CachedAgent]
    runtime_ms: float
    agent_inference_ms: float
    runtime_provenance: dict


class RuntimeBridge:
    def __init__(self, engine, agents, provenance=None, agent_order=None):
        self.engine, self.agents = engine, list(agents)
        self.agent_order = tuple(agent_order or [a.agent_id for a in self.agents])
        if tuple(a.agent_id for a in self.agents) != self.agent_order or len(set(self.agent_order)) != len(self.agents):
            raise SourceError("Runtime agent order/identity mismatch")
        self.provenance = dict(provenance or {})
        self.run_id = f"REPLAY_{uuid4().hex}"

    def run(self, observation: ObservationEnvelope) -> RuntimeCycle:
        if tuple(observation.agent_inputs) != self.agent_order:
            raise SourceError("Observation agent schema/order does not match runtime artifacts")
        start = time.perf_counter()
        # Deliberately forward ONLY inputs. Agents using the existing dict convention
        # require a dict rather than MappingProxyType; array contents stay immutable.
        inputs = dict(observation.agent_inputs)
        cached = [CachedAgent(a, inputs) for a in self.agents]
        inference_ms = (time.perf_counter() - start) * 1000
        result = self.engine.decide(cached, inputs, agent_reliabilities={k: 1.0 for k in self.agent_order})
        if tuple(result.agent_predictions) != self.agent_order:
            raise SourceError("DecisionEngine omitted required agent outputs")
        return RuntimeCycle(observation, result, cached, (time.perf_counter() - start) * 1000,
                            inference_ms, {**plain(self.provenance), "run_id": self.run_id,
                            "pipeline": "COGNIX DecisionEngine", "label_visible_to_inference": False,
                            "reliability_source": "explicit uniform demonstration assumption"})

"""
Unit tests for cognix.agents.registry.AgentRegistry.

All tests are fully deterministic and use only in-process, local objects.
No network calls are made.
"""
import threading
import time
import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from cognix.agents.registry import AgentRegistry
from cognix.agents.base import StandardAgent
from cognix.agents.adapters import (
    CallableAgentAdapter,
    SklearnAgentAdapter,
    PyTorchAgentAdapter,
    LLMAgentAdapter,
)
from cognix.core.interfaces import AgentInterface, UncertaintyEstimator
from cognix.core.types import PredictionResult, UncertaintyResult


# ── Minimal helpers ────────────────────────────────────────────────────────────

class _ConstantUncertaintyEstimator(UncertaintyEstimator):
    """Always returns a fixed uncertainty result."""
    def __init__(self, epistemic: float = 0.1, aleatoric: float = 0.1):
        self._ep = epistemic
        self._al = aleatoric

    def estimate(self, model, observation) -> UncertaintyResult:
        return UncertaintyResult(
            prediction=0.5,
            epistemic=self._ep,
            aleatoric=self._al,
            total=self._ep + self._al,
        )


def _make_standard_agent(agent_id: str, healthy: bool = True) -> StandardAgent:
    """Create a minimal StandardAgent with a tiny PyTorch model."""
    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    agent = StandardAgent(
        agent_id=agent_id,
        model=model,
        uncertainty_estimator=_ConstantUncertaintyEstimator(),
        domain_metadata={"agent_id": agent_id, "name": f"Agent-{agent_id}"},
    )
    agent.healthy = healthy
    return agent


def _make_callable_adapter(agent_id: str, healthy: bool = True) -> CallableAgentAdapter:
    adapter = CallableAgentAdapter(agent_id, lambda x: 0.75, name=f"Adapter-{agent_id}")
    adapter.healthy = healthy
    return adapter


# ── 1. Register an agent ───────────────────────────────────────────────────────

def test_register_agent():
    """Registering an agent makes it retrievable."""
    registry = AgentRegistry()
    agent = _make_callable_adapter("agent_01")
    registry.register(agent)
    assert registry.get("agent_01") is agent


# ── 2. Retrieve registered agent ──────────────────────────────────────────────

def test_get_registered_agent():
    """get() returns the exact agent object that was registered."""
    registry = AgentRegistry()
    a1 = _make_callable_adapter("a1")
    a2 = _make_callable_adapter("a2")
    registry.register(a1)
    registry.register(a2)
    assert registry.get("a1") is a1
    assert registry.get("a2") is a2


# ── 3. Duplicate registration behavior ────────────────────────────────────────

def test_duplicate_registration_replaces_old():
    """Registering a second agent with the same ID replaces the first silently."""
    registry = AgentRegistry()
    first = _make_callable_adapter("dup_id")
    second = _make_callable_adapter("dup_id")
    registry.register(first)
    registry.register(second)
    assert registry.get("dup_id") is second
    assert len(registry.list_agents()) == 1


# ── 4. Unregister ─────────────────────────────────────────────────────────────

def test_unregister_removes_agent():
    """Unregistering an agent removes it from the registry."""
    registry = AgentRegistry()
    agent = _make_callable_adapter("remove_me")
    registry.register(agent)
    registry.unregister("remove_me")
    assert registry.get("remove_me") is None
    assert len(registry.list_agents()) == 0


def test_unregister_nonexistent_is_silent():
    """Unregistering an ID that is not registered does not raise."""
    registry = AgentRegistry()
    registry.unregister("does_not_exist")  # Should not raise


# ── 5. Missing-agent lookup behavior ──────────────────────────────────────────

def test_get_missing_agent_returns_none():
    """get() returns None when the requested ID is not in the registry."""
    registry = AgentRegistry()
    assert registry.get("ghost") is None


# ── 6. list_agents behavior ───────────────────────────────────────────────────

def test_list_agents_empty():
    """Empty registry returns an empty list."""
    registry = AgentRegistry()
    assert registry.list_agents() == []


def test_list_agents_all_registered():
    """list_agents() returns all registered agents regardless of health."""
    registry = AgentRegistry()
    a1 = _make_callable_adapter("la1", healthy=True)
    a2 = _make_callable_adapter("la2", healthy=False)
    registry.register(a1)
    registry.register(a2)
    agents = registry.list_agents()
    assert len(agents) == 2
    assert a1 in agents
    assert a2 in agents


def test_list_agents_returns_snapshot():
    """list_agents() returns a copy; mutating it does not affect the registry."""
    registry = AgentRegistry()
    registry.register(_make_callable_adapter("snap1"))
    snapshot = registry.list_agents()
    snapshot.clear()  # should NOT affect the registry
    assert len(registry.list_agents()) == 1


# ── 7. healthy_agents includes healthy agents ─────────────────────────────────

def test_healthy_agents_includes_healthy():
    """healthy_agents() includes agents where agent.healthy == True."""
    registry = AgentRegistry()
    h = _make_callable_adapter("healthy_one", healthy=True)
    registry.register(h)
    result = registry.healthy_agents()
    assert h in result


# ── 8. healthy_agents excludes unhealthy agents ───────────────────────────────

def test_healthy_agents_excludes_unhealthy():
    """healthy_agents() excludes agents where agent.healthy == False."""
    registry = AgentRegistry()
    healthy = _make_callable_adapter("h_agent", healthy=True)
    sick = _make_callable_adapter("sick_agent", healthy=False)
    registry.register(healthy)
    registry.register(sick)
    result = registry.healthy_agents()
    assert healthy in result
    assert sick not in result


def test_healthy_agents_all_unhealthy_returns_empty():
    """healthy_agents() returns [] when all agents are unhealthy."""
    registry = AgentRegistry()
    registry.register(_make_callable_adapter("u1", healthy=False))
    registry.register(_make_callable_adapter("u2", healthy=False))
    assert registry.healthy_agents() == []


def test_healthy_agents_returns_snapshot():
    """Mutating the returned list does not affect registry.healthy_agents."""
    registry = AgentRegistry()
    registry.register(_make_callable_adapter("hsnap", healthy=True))
    snap = registry.healthy_agents()
    snap.clear()
    assert len(registry.healthy_agents()) == 1


# ── 9. Registry works with StandardAgent ─────────────────────────────────────

def test_registry_with_standard_agent():
    """AgentRegistry correctly registers and queries a StandardAgent."""
    registry = AgentRegistry()
    agent = _make_standard_agent("std_01", healthy=True)
    registry.register(agent)
    assert registry.get("std_01") is agent
    assert agent in registry.healthy_agents()

    agent.healthy = False
    assert agent not in registry.healthy_agents()


# ── 10. Registry works with one repaired adapter ──────────────────────────────

def test_registry_with_sklearn_adapter():
    """AgentRegistry works correctly with a fitted SklearnAgentAdapter."""
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([0, 1, 0, 1])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    registry = AgentRegistry()
    adapter = SklearnAgentAdapter("sklearn_reg_01", clf)
    registry.register(adapter)

    retrieved = registry.get("sklearn_reg_01")
    assert retrieved is adapter
    assert adapter in registry.healthy_agents()

    adapter.healthy = False
    assert adapter not in registry.healthy_agents()


# ── 11. Concurrent reads ──────────────────────────────────────────────────────

def test_concurrent_reads():
    """Multiple threads reading the registry simultaneously do not raise."""
    registry = AgentRegistry()
    for i in range(10):
        registry.register(_make_callable_adapter(f"cr_{i}", healthy=(i % 2 == 0)))

    errors = []

    def reader():
        try:
            for _ in range(100):
                _ = registry.list_agents()
                _ = registry.healthy_agents()
                _ = registry.get("cr_5")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == [], f"Concurrent reads raised errors: {errors}"


# ── 12. Concurrent registration/unregistration ───────────────────────────────

def test_concurrent_register_unregister():
    """Concurrent registration and unregistration are thread-safe."""
    registry = AgentRegistry()
    errors = []

    def register_agents(start: int, count: int):
        try:
            for i in range(start, start + count):
                registry.register(_make_callable_adapter(f"conc_{i}"))
        except Exception as exc:
            errors.append(exc)

    def unregister_agents(start: int, count: int):
        try:
            for i in range(start, start + count):
                registry.unregister(f"conc_{i}")
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for _ in range(200):
                _ = registry.list_agents()
                _ = registry.healthy_agents()
        except Exception as exc:
            errors.append(exc)

    writers = [threading.Thread(target=register_agents, args=(i * 20, 20)) for i in range(5)]
    removers = [threading.Thread(target=unregister_agents, args=(i * 20, 20)) for i in range(5)]
    readers = [threading.Thread(target=reader) for _ in range(4)]

    all_threads = writers + removers + readers
    for t in all_threads:
        t.start()
    for t in all_threads:
        t.join(timeout=15.0)

    assert errors == [], f"Concurrent register/unregister raised errors: {errors}"

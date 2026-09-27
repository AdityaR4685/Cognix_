import threading
from typing import Dict, List, Optional
from cognix.core.interfaces import AgentInterface


class AgentRegistry:
    """
    Thread-safe registry for managing active agents in the COGNIX framework.

    Provides mechanisms to register, unregister, and query agents based on
    their health status.

    Thread-safety: All public methods acquire an internal ``threading.RLock``
    so they are safe for concurrent reads and writes from multiple threads.
    """

    def __init__(self):
        self._agents: Dict[str, AgentInterface] = {}
        self._lock = threading.RLock()

    def register(self, agent: AgentInterface) -> None:
        """Register a new agent in the registry.

        The agent ID is resolved from ``agent.metadata()`` using the key
        ``"agent_id"`` (falling back to ``"id"``). If an agent with the same
        ID is already registered it will be silently replaced.
        """
        with self._lock:
            meta = agent.metadata()
            agent_id = meta.get("id") or meta.get("agent_id") or "unknown_agent"
            self._agents[agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        """Unregister an agent by its ID.

        Silently ignored if the agent is not registered.
        """
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]

    def get(self, agent_id: str) -> Optional[AgentInterface]:
        """Retrieve an agent by its ID.

        Returns ``None`` if the agent is not registered.
        """
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentInterface]:
        """Return a snapshot list of all registered agents."""
        with self._lock:
            return list(self._agents.values())

    def healthy_agents(self) -> List[AgentInterface]:
        """Return a snapshot list of all registered agents that are currently healthy.

        Uses the canonical ``agent.healthy`` boolean attribute, which is the
        standard health contract exposed by ``StandardAgent`` and all COGNIX
        adapter classes (``CallableAgentAdapter``, ``SklearnAgentAdapter``,
        ``PyTorchAgentAdapter``, ``LLMAgentAdapter``).

        Agents that do not expose a ``healthy`` attribute are conservatively
        assumed healthy so that legacy or third-party agents are not silently
        excluded.
        """
        with self._lock:
            return [
                agent for agent in self._agents.values()
                if getattr(agent, "healthy", True)
            ]

    # ── Context-manager support (acquire/release the RLock directly) ──────────

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

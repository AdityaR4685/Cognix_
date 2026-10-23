"""
COGNIX Decision Engine — public API entry point.

Usage::

    from cognix import DecisionEngine

    engine = DecisionEngine(
        uncertainty='mc_dropout',
        belief='epistemic_weighted',
        calibrator='conformal',
        config=CognixConfig(),
    )
    result = engine.decide(agents=[...], input_data=data, context={})
    print(result.summary())
"""
from __future__ import annotations

import logging
from typing import Any

from cognix.engine.pipeline import CognixPipeline
from cognix.engine.result import DecisionResult

logger = logging.getLogger(__name__)


def _resolve_module(category: str, value: Any) -> Any:
    """Resolve module shorthand strings or booleans to canonical COGNIX instances."""
    if value is None:
        return None

    if category == "uncertainty" and isinstance(value, str):
        val = value.lower().strip()
        if val == "mc_dropout":
            from cognix.uncertainty.mc_dropout import MCDropout
            return MCDropout()
        elif val == "deep_ensemble":
            from cognix.uncertainty.deep_ensemble import DeepEnsemble
            return DeepEnsemble()
        elif val == "none":
            return None

    elif category == "belief" and isinstance(value, str):
        val = value.lower().strip()
        if val == "epistemic_weighted":
            from cognix.belief.fusion import EpistemicWeightedFusion
            return EpistemicWeightedFusion()
        elif val in ("average", "uniform"):
            from cognix.belief.fusion import AverageFusion
            return AverageFusion()
        elif val == "none":
            return None

    elif category == "calibrator" and isinstance(value, str):
        val = value.lower().strip()
        if val == "conformal":
            from cognix.calibration.conformal import ConformalPredictor
            return ConformalPredictor()
        elif val == "none":
            return None

    elif category == "attribution" and isinstance(value, str):
        val = value.lower().strip()
        if val in ("epistemic_shapley", "shapley"):
            from cognix.attribution.epistemic_shapley import EpistemicShapley
            return EpistemicShapley()
        elif val == "none":
            return None

    elif category == "escalation":
        if value is True or (isinstance(value, str) and value.lower().strip() in ("true", "escalation", "default")):
            from cognix.decision.escalation import EscalationEngine
            return EscalationEngine()
        elif value is False or (isinstance(value, str) and value.lower().strip() in ("false", "none")):
            return None

    elif category == "graph" and isinstance(value, str):
        val = value.lower().strip()
        if val in ("epistemic_gat", "standard_gat"):
            try:
                if val == "epistemic_gat":
                    from cognix.graph.epistemic_gat import EpistemicGAT
                    return EpistemicGAT()
                elif val == "standard_gat":
                    from cognix.graph.standard_gat import StandardGAT
                    return StandardGAT()
            except ImportError as e:
                raise ImportError(
                    f"Graph module '{val}' requires PyTorch. "
                    "Install with: pip install 'cognix-uncertainty-framework[graph]'"
                ) from e
        elif val == "no_graph":
            from cognix.graph.no_graph import NoGraph
            return NoGraph()
        elif val == "none":
            return None

    return value


class DecisionEngine:
    """
    COGNIX Decision Engine — the primary public API.

    Accepts string shorthand or object instances for each module.
    Builds and wires the CognixPipeline internally.

    Example::

        engine = DecisionEngine(
            uncertainty='mc_dropout',
            belief='epistemic_weighted',
            calibrator='conformal',
            escalation=True,
        )
        result = engine.decide(agents=my_agents, input_data=data)
        print(result.summary())
    """

    def __init__(
        self,
        config: Any = None,
        uncertainty: Any = None,
        belief: Any = None,
        calibrator: Any = None,
        communication: Any = None,
        attribution: Any = None,
        escalation: Any = None,
        graph: Any = None,
        mode: str = "production",
    ) -> None:
        from cognix.config.schema import CognixConfig
        self.config = config or CognixConfig()
        self.mode = mode

        self.uncertainty = _resolve_module("uncertainty", uncertainty)
        self.belief = _resolve_module("belief", belief)
        self.calibrator = _resolve_module("calibrator", calibrator)
        self.communication = communication
        self.attribution = _resolve_module("attribution", attribution)
        self.escalation_module = _resolve_module("escalation", escalation)
        self.graph = _resolve_module("graph", graph)

        self.pipeline = CognixPipeline(
            config=self.config,
            uncertainty_estimator=self.uncertainty,
            belief_fuser=self.belief,
            calibrator=self.calibrator,
            graph=self.graph,
            communication=self.communication,
            attribution=self.attribution,
            escalation=self.escalation_module,
            mode=mode,
        )

        logger.info(
            "COGNIX DecisionEngine initialized | "
            "uncertainty=%s | belief=%s | calibrator=%s | graph=%s | escalation=%s",
            type(self.uncertainty).__name__ if self.uncertainty else "None",
            type(self.belief).__name__ if self.belief else "None",
            type(self.calibrator).__name__ if self.calibrator else "None",
            type(self.graph).__name__ if self.graph else "None",
            type(self.escalation_module).__name__ if self.escalation_module else "None",
        )

    def decide(
        self,
        agents: list[Any],
        input_data: Any,
        context: dict[str, Any] | None = None,
        agent_reliabilities: dict[str, float] | None = None,
    ) -> DecisionResult:
        """Run the full COGNIX pipeline and return a structured DecisionResult."""
        return self.pipeline.run(
            agents, input_data, context or {},
            agent_reliabilities=agent_reliabilities,
        )

    def configure(self, config: Any) -> None:
        """Update the engine configuration."""
        self.config = config
        self.pipeline.config = config

    def __enter__(self) -> "DecisionEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"DecisionEngine("
            f"uncertainty={type(self.uncertainty).__name__ if self.uncertainty else None}, "
            f"belief={type(self.belief).__name__ if self.belief else None}, "
            f"calibrator={type(self.calibrator).__name__ if self.calibrator else None})"
        )

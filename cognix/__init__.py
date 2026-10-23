"""
COGNIX: A domain-agnostic, uncertainty-aware multi-agent AI decision framework.

Research hypothesis:
    Can epistemic uncertainty be used as a unified trust signal to dynamically
    influence multi-agent belief fusion, inter-agent communication, attribution,
    and risk-aware decision making under agent disagreement, sensor degradation,
    and distribution shift?

Version: 0.1.0
License: MIT
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cognix-uncertainty-framework")
except PackageNotFoundError:
    try:
        __version__ = version("cognix")
    except PackageNotFoundError:
        __version__ = "0.1.0"

__author__ = "COGNIX Contributors"
__description__ = "A domain-agnostic, uncertainty-aware multi-agent AI decision framework"

# Core public API
from cognix.engine.decision_engine import DecisionEngine
from cognix.engine.result import DecisionResult, RiskLevel, DecisionOutcome
from cognix.agents.base import StandardAgent
from cognix.agents.registry import AgentRegistry
from cognix.uncertainty.mc_dropout import MCDropout as MonteCarloDropout
from cognix.uncertainty.deep_ensemble import DeepEnsemble
from cognix.uncertainty.decomposition import UncertaintyDecomposition
from cognix.belief.bayesian import BayesianBelief
from cognix.belief.fusion import EpistemicWeightedFusion, AverageFusion
from cognix.calibration.conformal import ConformalPredictor
from cognix.calibration.temperature import TemperatureScaling
from cognix.decision.escalation import EscalationEngine
from cognix.config.schema import CognixConfig

__all__ = [
    "__version__",
    "DecisionEngine",
    "DecisionResult",
    "RiskLevel",
    "DecisionOutcome",
    "StandardAgent",
    "AgentRegistry",
    "MonteCarloDropout",
    "DeepEnsemble",
    "UncertaintyDecomposition",
    "BayesianBelief",
    "EpistemicWeightedFusion",
    "AverageFusion",
    "ConformalPredictor",
    "TemperatureScaling",
    "EscalationEngine",
    "CognixConfig",
]

from .base import UncertaintyEstimator, UncertaintyEstimate
from .mc_dropout import MCDropout as MonteCarloDropout
from .deep_ensemble import DeepEnsemble
from .decomposition import UncertaintyDecomposition
from .ood import OODDetector

try:
    from .heteroscedastic import HeteroscedasticLoss
except ImportError:
    HeteroscedasticLoss = None

__all__ = [
    "UncertaintyEstimator",
    "UncertaintyEstimate",
    "MonteCarloDropout",
    "DeepEnsemble",
    "UncertaintyDecomposition",
    "OODDetector",
]
if HeteroscedasticLoss is not None:
    __all__.append("HeteroscedasticLoss")

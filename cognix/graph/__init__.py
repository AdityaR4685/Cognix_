from .gcn import GCNNetwork
from .gat import GATNetwork
from .routing import build_fully_connected_graph, top_k_routing
from .no_graph import NoGraph

try:
    from .epistemic_gat import EpistemicGAT
except ImportError:
    EpistemicGAT = None  # type: ignore[assignment,misc]

__all__ = [
    "GCNNetwork",
    "GATNetwork",
    "NoGraph",
    "build_fully_connected_graph",
    "top_k_routing",
]
if EpistemicGAT is not None:
    __all__.append("EpistemicGAT")

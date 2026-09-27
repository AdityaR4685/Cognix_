from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.core.interfaces import GraphRefinement
from cognix.core.types import GraphResult
import numpy as np

class StandardGAT(GraphRefinement):
    """
    Standard Graph Attention Network Plugin.
    
    This acts as a wrapper around the PyTorch EpistemicGAT implementation 
    but strictly disables the epistemic prior, causing it to fall back
    to a standard learned attention mechanism.
    """
    
    def __init__(self, num_layers: int = 2, input_dim: int = 3, hidden_dim: int = 8, output_dim: int = 4):
        self._gat = EpistemicGAT(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_epistemic_prior=False  # Key difference
        )
        
    def fit(self, *args, **kwargs):
        # Expose the internal training method for the pipeline to use
        return self._gat.fit(*args, **kwargs)
        
    def is_trained(self):
        return self._gat.is_trained()
        
    def forward(
        self,
        node_features: np.ndarray,
        adjacency_matrix: np.ndarray,
        epistemic_uncertainties: dict[str, float],
        agent_order: list[str]
    ) -> GraphResult:
        
        return self._gat(
            node_features,
            adjacency_matrix,
            epistemic_uncertainties,
            agent_order
        )

    def save(self, filepath: str) -> None:
        """Save trained model weights to file."""
        from cognix.artifacts.persistence import save_standard_gat
        save_standard_gat(self, filepath)

    def load(self, filepath: str) -> "StandardGAT":
        """Load trained model weights from file."""
        from cognix.artifacts.persistence import load_standard_gat
        loaded = load_standard_gat(filepath, expected_input_dim=self._gat.input_dim, expected_output_dim=self._gat.output_dim)
        self._gat.load_state_dict(loaded._gat.state_dict())
        self._gat._trained = True
        self._gat.eval()
        return self

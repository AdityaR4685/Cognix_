import numpy as np
from cognix.core.interfaces import GraphRefinement
from cognix.core.types import GraphResult

class NoGraph(GraphRefinement):
    """
    An identity GraphRefinement plugin.
    
    Returns the initial node predictions (feature 0) unchanged.
    Used for ablations and architectures that do not require spatial/graph reasoning.
    """
    
    def forward(
        self,
        node_features,
        adjacency_matrix,
        epistemic_uncertainties: dict[str, float],
        agent_order: list[str]
    ) -> GraphResult:

        # node_features[:, 0] contains agent probabilities p_i in [0, 1].
        # The pipeline applies sigmoid(H_prime[:, 0]) downstream, so we must
        # return LOGITS (not probabilities) to avoid a double-sigmoid.
        # logit(p) = log(p / (1 - p))  maps p -> logit space.
        # sigmoid(logit(p)) == p  (identity round-trip).
        try:
            import torch
            if isinstance(node_features, torch.Tensor):
                node_features = node_features.detach().cpu().numpy()
        except ImportError:
            pass

        N = node_features.shape[0]
        probs = np.clip(node_features[:, 0], 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1.0 - probs))          # logit transform
        H_prime = logits.reshape(N, 1).astype(np.float32)

        # Attention is identity (no graph reasoning)
        attn = np.eye(N, dtype=np.float32)

        return GraphResult(
            node_outputs=H_prime,
            attention=[attn],
            metadata={"type": "NoGraph"}
        )

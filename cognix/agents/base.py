from typing import Any, Optional

from cognix.core.interfaces import AgentInterface, UncertaintyEstimator
from cognix.core.types import PredictionResult, UncertaintyResult

class StandardAgent(AgentInterface):
    """
    Standard implementation of a COGNIX Agent.
    
    This agent wraps an underlying model (e.g., PyTorch, sklearn) and an 
    UncertaintyEstimator plugin. It provides a standardized interface for the
    DecisionEngine to query predictions and uncertainty estimates without needing
    to know the internal workings of the model.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        model: Any, 
        uncertainty_estimator: UncertaintyEstimator,
        domain_metadata: Optional[dict] = None
    ):
        self.agent_id = agent_id
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self._metadata = domain_metadata or {}
        
        # Ensure metadata includes the agent_id
        self._metadata["agent_id"] = agent_id
        
        # Health status for experimental simulation (fault injection)
        self.healthy = True

    def predict(self, observation: Any) -> PredictionResult:
        """
        Generate a prediction given the input observation.
        """
        if not self.healthy:
            return PredictionResult(value=0.5, confidence=0.0, metadata={"status": "unhealthy"})
            
        import torch
        # Simplified for PyTorch models in synthetic experiment
        self.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(observation)
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            out = self.model(x_t).item()
            
        return PredictionResult(
            value=out,
            confidence=float(out)
        )

    def estimate_uncertainty(self, observation: Any) -> UncertaintyResult:
        """
        Estimate uncertainty using the injected UncertaintyEstimator plugin.
        """
        if not self.healthy:
            return UncertaintyResult(
                prediction=0.5,
                epistemic=1.0,  # Max uncertainty
                aleatoric=1.0,
                total=2.0
            )
            
        return self.uncertainty_estimator.estimate(self.model, observation)

    def metadata(self) -> dict:
        """Return agent metadata."""
        return self._metadata

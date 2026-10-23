"""
Bandwidth-Constrained Top-K Routing.
"""
from typing import Dict, List

class TopKCommunication:
    """
    Selects the Top-K agents to broadcast their beliefs based on confidence or epistemic uncertainty.
    """
    def __init__(self, k: int = 3, criterion: str = 'epistemic'):
        """
        Args:
            k: Number of agents allowed to broadcast.
            criterion: 'epistemic' (highest uncertainty broadcasts) or 'confidence' (highest confidence broadcasts).
        """
        self.k = k
        self.criterion = criterion
        self.last_message_count = 0

    def select_broadcasters(self, predictions: Dict[str, float], uncertainties: Dict[str, float]) -> List[str]:
        """
        Returns a list of agent IDs selected to broadcast.
        """
        agent_ids = list(predictions.keys())
        n = len(agent_ids)
        if n <= self.k:
            self.last_message_count = n * (n - 1)
            return agent_ids

        scores = {}
        for aid in agent_ids:
            if self.criterion == 'epistemic':
                scores[aid] = uncertainties.get(aid, 0.0)
            else:
                scores[aid] = predictions.get(aid, 0.0)

        # Sort descending
        sorted_agents = sorted(agent_ids, key=lambda a: scores[a], reverse=True)
        selected = sorted_agents[:self.k]
        
        self.last_message_count = self.k * (n - 1)
        return selected

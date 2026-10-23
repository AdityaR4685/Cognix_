"""
Information Gain Routing.
"""
from typing import Dict, List

class InformationGainRouter:
    """
    Selects agents to broadcast based on their information gain heuristic.
    IG_approx = Confidence / (Epistemic + epsilon)
    """
    def __init__(self, threshold: float = 1.5):
        """
        Args:
            threshold: Minimum IG ratio required to broadcast.
        """
        self.threshold = threshold
        self.last_message_count = 0

    def select_broadcasters(self, predictions: Dict[str, float], uncertainties: Dict[str, float]) -> List[str]:
        """
        Returns a list of agent IDs whose information gain exceeds the threshold.
        """
        eps = 1e-8
        agent_ids = list(predictions.keys())
        n = len(agent_ids)
        selected = []

        for aid in agent_ids:
            conf = predictions.get(aid, 0.5)
            epi = uncertainties.get(aid, 1.0)
            
            ig_score = conf / (epi + eps)
            if ig_score >= self.threshold:
                selected.append(aid)

        self.last_message_count = len(selected) * (n - 1)
        return selected

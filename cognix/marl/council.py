"""
Decision Council profiles for multi-agent coordination.
"""
from typing import Any

class DecisionCouncil:
    """
    Assigns behavioral profiles to agents and coordinates their macro-level decisions.
    """
    def __init__(self, agents: list[Any]):
        self.agents = agents
        self.profiles = {}
        self._assign_profiles()
        
    def _assign_profiles(self):
        """
        Assigns standard profiles (Optimist, Risk-Averse, Judge) to the agents.
        """
        n = len(self.agents)
        if n >= 3:
            self.profiles[self.agents[0]] = "OPTIMIST"     # Seeks high reward, ignores high epistemic uncertainty
            self.profiles[self.agents[1]] = "RISK_AVERSE"  # Avoids actions with high epistemic uncertainty
            self.profiles[self.agents[2]] = "JUDGE"        # Fuses outputs based on reliability
        else:
            for agent in self.agents:
                self.profiles[agent] = "NEUTRAL"
                
    def get_profile(self, agent: Any) -> str:
        return self.profiles.get(agent, "NEUTRAL")
        
    def bias_uncertainty(self, agent: Any, base_uncertainty: float) -> float:
        """
        Adjusts the reported uncertainty based on the agent's profile.
        """
        profile = self.get_profile(agent)
        if profile == "OPTIMIST":
            return base_uncertainty * 0.5  # Under-reports uncertainty
        elif profile == "RISK_AVERSE":
            return min(1.0, base_uncertainty * 1.5)  # Over-reports uncertainty
        return base_uncertainty

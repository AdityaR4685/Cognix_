"""
Epistemic Shapley Attribution.

PROPOSED COGNIX MECHANISM (requires experimental validation):
  phi_i = sigma_e(coalition WITH agent i) - sigma_e(coalition WITHOUT agent i)

This is analogous to standard Shapley values but applied to epistemic uncertainty reduction
instead of model output. It approximates the marginal contribution of each agent to
collective epistemic uncertainty.

Note: Standard Shapley values (Shapley, 1953) are an established game-theoretic concept.
The specific application to epistemic uncertainty in multi-agent systems is a proposed mechanism.
"""

import random
from typing import Callable, Any

from cognix.core.interfaces import AttributionMethod

class EpistemicShapley(AttributionMethod):
    def compute(self, agents: list[str], prediction_function: Callable[[list[str]], float], target_output: Any = None) -> dict[str, float]:
        """
        Use Monte Carlo approximation of Shapley values.
        """
        shapley_values = {agent: 0.0 for agent in agents}
        n = len(agents)
        
        if n == 0:
            return shapley_values
            
        num_samples = 100
        for _ in range(num_samples):
            permutation = list(agents)
            random.shuffle(permutation)
            
            coalition = []
            prev_uncertainty = prediction_function(coalition)
            
            for agent in permutation:
                coalition.append(agent)
                curr_uncertainty = prediction_function(coalition)
                # Contribution is reduction in uncertainty (or change)
                # phi_i = sigma_e(coalition WITH agent i) - sigma_e(coalition WITHOUT agent i)
                marginal_contribution = curr_uncertainty - prev_uncertainty
                shapley_values[agent] += marginal_contribution
                prev_uncertainty = curr_uncertainty
                
        # Average over samples
        for agent in agents:
            shapley_values[agent] /= num_samples
            
        return shapley_values

    def interpret(self, shapley_values: dict[str, float]) -> dict[str, Any]:
        if not shapley_values:
            return {}
            
        ranking = sorted(shapley_values.items(), key=lambda x: x[1])
        top_contributor = ranking[0][0] if ranking else None
        
        summary = f"Top contributor to uncertainty reduction: {top_contributor}"
        
        return {
            'top_contributor': top_contributor,
            'ranking': ranking,
            'summary': summary
        }

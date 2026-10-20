"""
Decision Escalation Engine with Conformal and Epistemic Support.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Any, Tuple

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

class DecisionOutcome(Enum):
    ACT = "ACT"
    WAIT = "WAIT"
    REQUEST_INFORMATION = "REQUEST_INFORMATION"
    ESCALATE = "ESCALATE"
    ABSTAIN = "ABSTAIN"

@dataclass
class EscalationResult:
    outcome: DecisionOutcome
    escalation: bool
    reason: str
    risk_level: RiskLevel

class EscalationEngine:
    """
    Determines if a decision should be escalated to a human operator or higher-tier system.
    Supports Point-estimate Confidence, Epistemic Uncertainty, and Conformal Set Sizes.
    Shapley Attribution is informational (explanation-only).
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.7,
        low_conf_thresh: float = 0.4,
        high_unc_thresh: float = 0.5,
        max_conformal_set_size: int = 3,
    ):
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.high_unc_thresh = high_unc_thresh
        self.max_conformal_set_size = max_conformal_set_size
        
    def evaluate(self, 
                 confidence: float, 
                 epistemic_uncertainty: float,
                 conformal_set_size: int = 1,
                 max_shapley_value: float = 0.0) -> EscalationResult:
        """
        Evaluate decision escalation conditions.

        Note: max_shapley_value is retained for API backward compatibility,
        but is informational and unused for decision gating (Shapley attribution
        is explanation-only).
        """
        # Conformal Prediction Thresholding
        if conformal_set_size >= self.max_conformal_set_size:
            return EscalationResult(DecisionOutcome.ESCALATE, True, f"Conformal set size ({conformal_set_size}) exceeds safe limit.", RiskLevel.HIGH)
            
        # Three-Tier Point Estimate Logic
        if confidence < self.low_conf_thresh or epistemic_uncertainty > self.high_unc_thresh:
            return EscalationResult(DecisionOutcome.ESCALATE, True, "Low confidence or high uncertainty.", RiskLevel.HIGH)
            
        if confidence < self.high_conf_thresh:
            return EscalationResult(DecisionOutcome.WAIT, False, "Confidence in ambiguous zone; precautionary wait.", RiskLevel.MODERATE)
            
        return EscalationResult(DecisionOutcome.ACT, False, "Confident and certain.", RiskLevel.LOW)

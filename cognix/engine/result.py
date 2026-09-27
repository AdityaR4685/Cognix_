import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

class DecisionOutcome(Enum):
    ACT = "ACT"
    WAIT = "WAIT"
    REQUEST_INFORMATION = "REQUEST_INFORMATION"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"

@dataclass
class DecisionResult:
    # Core decision
    decision: DecisionOutcome
    confidence: float
    calibrated_confidence: Optional[float]
    
    # Uncertainty
    aleatoric_uncertainty: Optional[float]
    epistemic_uncertainty: Optional[float]
    total_uncertainty: float
    
    # Risk and escalation
    risk_level: RiskLevel
    escalation_required: bool
    abstained: bool
    requested_information: bool
    
    # Agent-level information
    agent_contributions: dict[str, float]
    agent_trust_weights: dict[str, float]
    agent_predictions: dict[str, float]
    
    # Communication
    communication_statistics: Optional[dict]
    
    # Calibration
    calibration_metrics: Optional[dict]
    
    # Explanation
    explanation: str
    reasoning_steps: list[str]
    
    # Metadata
    latency_ms: dict[str, float]
    total_latency_ms: float
    timestamp: float
    session_id: Optional[str]
    metadata: dict[str, Any]
    
    @property
    def is_safe_decision(self) -> bool:
        return not self.escalation_required and not self.abstained
    
    @property
    def dominant_uncertainty_source(self) -> str:
        if self.aleatoric_uncertainty is None or self.epistemic_uncertainty is None:
            return "unknown"
        if self.epistemic_uncertainty > self.aleatoric_uncertainty:
            return "epistemic"
        return "aleatoric"
    
    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        data = dataclasses.asdict(self)
        data['decision'] = self.decision.value
        data['risk_level'] = self.risk_level.value
        return data
    
    def summary(self) -> str:
        """Short human-readable summary."""
        return f"Decision: {self.decision.value} (Conf: {self.confidence:.2f}, Risk: {self.risk_level.value}, Uncertainty: {self.total_uncertainty:.2f})"

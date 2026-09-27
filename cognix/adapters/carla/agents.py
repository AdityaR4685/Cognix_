"""
CarlAnomaly Sensor Agents
Maps each sensor stream to a COGNIX AgentInterface.
"""
from typing import Optional, Dict
import numpy as np

from typing import Optional, Dict, Any
import numpy as np

from cognix.core.interfaces import AgentInterface
from cognix.core.types import PredictionResult, UncertaintyResult
from cognix.adapters.carla.features import (
    rgb_to_hazard_prob,
    depth_to_obstacle_score,
    lidar_to_density_score,
    gnss_to_drift_score,
    imu_to_motion_score,
    seg_to_complexity_score
)

class CarlAnomalyAgent(AgentInterface):
    def __init__(self, agent_id: str, feature_fn, pass_count: int = 50):
        self.agent_id = agent_id
        self.feature_fn = feature_fn
        self.pass_count = pass_count
        self.rng = np.random.default_rng(abs(hash(agent_id)) % (2**32))
        self._metadata = {"agent_id": agent_id}

    def _extract_data(self, observation: Any) -> Any:
        if isinstance(observation, dict):
            return observation.get(self.agent_id, observation)
        return observation

    def predict(self, observation: Any) -> PredictionResult:
        obs = self._extract_data(observation)
        probs = self.feature_fn(obs)
        prob_hazard = float(probs[0])
        # Pipeline fuses PredictionResult.value as P(ACT/safe).
        # Feature extractors return P(hazard), so invert here at the adapter
        # boundary. The original hazard probability is retained in metadata.
        prob_act = 1.0 - prob_hazard
        return PredictionResult(
            value=prob_act,
            confidence=prob_act,
            metadata={
                "hazard_probability": prob_hazard,
                "prediction_semantics": "P(ACT/safe)",
            }
        )

    def estimate_uncertainty(self, observation: Any) -> UncertaintyResult:
        obs = self._extract_data(observation)
        mc_preds = []
        # Epistemic simulation: add noise proportional to anomaly magnitude directly to probability
        base_prob = float(self.feature_fn(obs)[0])
        dist = max(0.0, base_prob - 0.1) # 0.1 is normal
        for _ in range(self.pass_count):
            noise_prob = base_prob + self.rng.normal(0, 0.02 + dist * 0.4)
            p = np.clip(noise_prob, 0.001, 0.999)
            mc_preds.append([1.0 - p, p])
        
        mc_preds = np.array(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)
        
        entropies = -np.sum(mc_preds * np.log(mc_preds + 1e-8), axis=1)
        aleatoric = float(np.mean(entropies))
        
        total_entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))
        epistemic = max(0.0, total_entropy - aleatoric)
        
        return UncertaintyResult(
            prediction=float(mean_probs[1]),
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total_entropy
        )

    def metadata(self) -> dict:
        return self._metadata

class CameraAgent(CarlAnomalyAgent):
    def __init__(self, agent_id="Camera"):
        super().__init__(agent_id, rgb_to_hazard_prob)

class DepthAgent(CarlAnomalyAgent):
    def __init__(self, agent_id="Depth"):
        super().__init__(agent_id, depth_to_obstacle_score)

class LiDARAgent(CarlAnomalyAgent):
    def __init__(self, agent_id="LiDAR"):
        super().__init__(agent_id, lidar_to_density_score)

class GNSSAgent(CarlAnomalyAgent):
    def __init__(self, agent_id="GNSS"):
        super().__init__(agent_id, gnss_to_drift_score)

class IMUAgent(CarlAnomalyAgent):
    def __init__(self, agent_id="IMU"):
        super().__init__(agent_id, imu_to_motion_score)

class SegAgent(CarlAnomalyAgent):
    def __init__(self, agent_id="Seg"):
        super().__init__(agent_id, seg_to_complexity_score)

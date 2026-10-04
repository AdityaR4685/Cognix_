from abc import ABC, abstractmethod
from cognix.core.types import (
    PredictionResult, 
    UncertaintyResult, 
    GraphResult, 
    FusionResult
)

class UncertaintyEstimator(ABC):
    @abstractmethod
    def estimate(self, model, observation) -> UncertaintyResult:
        pass

class AgentInterface(ABC):
    @abstractmethod
    def predict(self, observation) -> PredictionResult:
        pass
        
    @abstractmethod
    def estimate_uncertainty(self, observation) -> UncertaintyResult:
        pass
        
    @abstractmethod
    def metadata(self) -> dict:
        pass

class GraphRefinement(ABC):
    @abstractmethod
    def forward(self, node_features, adjacency, epistemic_uncertainties, agent_order) -> GraphResult:
        pass

class BeliefFuser(ABC):
    @abstractmethod
    def fuse(self, predictions, uncertainties, reliabilities) -> FusionResult:
        pass

class Calibrator(ABC):
    @abstractmethod
    def fit(self, cal_outputs, cal_labels):
        pass
        
    @abstractmethod
    def predict(self, new_output, alpha: float = 0.05):
        pass

class AttributionMethod(ABC):
    @abstractmethod
    def compute(self, agents, prediction_function, target_output):
        pass

class RiskStrategy(ABC):
    @abstractmethod
    def assess(self, prediction_set):
        pass

class TransportInterface(ABC):
    @abstractmethod
    def send(self, message) -> None:
        pass
        
    @abstractmethod
    def receive(self):
        pass

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConformalPredictionSet:
    prediction_set: list[int]
    confidence: float
    quantile: float
    coverage_target: float

from cognix.core.interfaces import Calibrator

class ConformalPredictor(Calibrator):
    """
    Inductive Conformal Prediction.
    Reference: Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
    Reference for inductive CP: Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). 
    Inductive Confidence Machines for Regression.
    """
    def __init__(self):
        self.cal_scores: Optional[np.ndarray] = None
        self.n_cal = 0

    def fit(self, cal_outputs: np.ndarray, cal_labels: np.ndarray):
        """
        Fit on calibration set, compute nonconformity scores.
        Nonconformity score for classification: 1 - p_hat[true_class] (for softmax outputs)
        cal_outputs: (N, C) predicted probabilities
        cal_labels: (N,) true integer labels
        """
        N = cal_outputs.shape[0]
        scores = 1.0 - cal_outputs[np.arange(N), cal_labels]
        self.cal_scores = np.sort(scores)
        self.n_cal = N

    def save(self, filepath: str) -> None:
        """Save calibration scores to a .npy file."""
        from cognix.artifacts.persistence import save_conformal
        save_conformal(self, filepath)

    def load(self, filepath: str) -> "ConformalPredictor":
        """Load calibration scores from a .npy file."""
        from cognix.artifacts.persistence import load_conformal
        loaded = load_conformal(filepath)
        self.cal_scores = loaded.cal_scores
        self.n_cal = loaded.n_cal
        return self

    def predict(self, test_outputs: np.ndarray, alpha: float = 0.05) -> list[ConformalPredictionSet]:
        """
        Produce prediction sets for new examples using split conformal prediction.

        test_outputs : (M, C) predicted class probabilities

        Coverage guarantee (Vovk et al.):
            P(Y_test in C(X_test)) >= 1 - alpha
        provided that cal and test scores are exchangeable.

        Implementation: the quantile is taken over the AUGMENTED calibration
        set {s_1, ..., s_n, +inf}, which has n+1 elements.
        q_val = the ceil((n+1)(1-alpha))-th smallest value in the augmented set.
        If that index exceeds n, q_val = +inf (include all classes).
        """
        if self.cal_scores is None:
            raise RuntimeError("Conformal predictor must be calibrated before prediction.")

        M, C = test_outputs.shape
        # Augmented set: append +inf as the (n+1)-th score
        augmented = np.append(self.cal_scores, np.inf)
        n_aug = len(augmented)  # = n_cal + 1

        q_idx = int(np.ceil(n_aug * (1 - alpha)))   # index into augmented set
        q_idx = min(q_idx, n_aug)                    # clamp to valid range
        q_val = float(np.sort(augmented)[q_idx - 1]) # 1-indexed → 0-indexed
            
        prediction_sets = []
        for i in range(M):
            # nonconformity score for all classes y: 1 - p_hat[y]
            scores_y = 1.0 - test_outputs[i]
            pred_set = np.where(scores_y <= q_val)[0].tolist()
            
            prediction_sets.append(ConformalPredictionSet(
                prediction_set=pred_set,
                confidence=1.0 - alpha,
                quantile=q_val,
                coverage_target=1.0 - alpha
            ))
            
        return prediction_sets

def evaluate_coverage(prediction_sets: list[ConformalPredictionSet], true_labels: np.ndarray) -> float:
    """Empirical coverage of the prediction sets."""
    covered = 0
    for i, p_set in enumerate(prediction_sets):
        if true_labels[i] in p_set.prediction_set:
            covered += 1
    return covered / len(true_labels) if len(true_labels) > 0 else 0.0

class TwoStageConformalPredictor:
    """
    TwoStageConformalPredictor: individual threshold tau_i per agent, 
    then collective threshold on aggregated output.
    """
    def __init__(self):
        self.individual_predictors = {}
        self.ensemble_predictor = ConformalPredictor()

    def calibrate(self, agent_outputs: dict[str, np.ndarray], ensemble_outputs: np.ndarray, labels: np.ndarray):
        for agent_id, outputs in agent_outputs.items():
            cp = ConformalPredictor()
            cp.fit(outputs, labels)
            self.individual_predictors[agent_id] = cp
            
        self.ensemble_predictor.fit(ensemble_outputs, labels)

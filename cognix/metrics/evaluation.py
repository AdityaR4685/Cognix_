"""
COGNIX Metrics & Evaluation Module.

Contains mathematically rigorous, centralized metric calculations 
for evaluating calibration, accuracy, and out-of-distribution detection.
"""
import numpy as np
from typing import List, Dict

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def calculate_ece(predictions: np.ndarray, ground_truths: np.ndarray, bins: int = 10, n_bins: int = None) -> float:
    """
    Expected Calibration Error for binary classification (reliability-diagram definition).

    ECE = sum_{m=1}^{M} (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Where:
        conf(B_m) = mean predicted probability in bin m  (= mean p_hat)
        acc(B_m)  = fraction of positives in bin m       (= mean y_true)

    This is the standard definition from Guo et al. (2017) adapted for binary
    classification.  It measures how well predicted probabilities correspond to
    empirical positive rates.

    NOTE: A previous version used max(p, 1-p) as confidence and round(p)==y as
    accuracy.  Both were incorrect for the reliability-diagram formulation and
    produced ECE = 0.5 for perfectly calibrated predictions.

    Accepts both 'bins' and 'n_bins' for backward compatibility.
    """
    if n_bins is not None:
        bins = n_bins
    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truths = np.asarray(ground_truths, dtype=np.float64)
    if len(predictions) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        # Include right endpoint on last bin to catch p_hat == 1.0
        if i < bins - 1:
            mask = (predictions >= lower) & (predictions < upper)
        else:
            mask = (predictions >= lower) & (predictions <= upper)

        if np.any(mask):
            bin_preds  = predictions[mask]
            bin_truths = ground_truths[mask]

            avg_conf = float(np.mean(bin_preds))          # mean predicted probability
            avg_acc  = float(np.mean(bin_truths))          # fraction of positives

            weight = len(bin_preds) / len(predictions)
            ece += weight * abs(avg_acc - avg_conf)

    return float(ece)

# Alias for backwards compatibility and test imports
expected_calibration_error = calculate_ece

def accuracy(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    """Calculate binary accuracy."""
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(np.round(predictions) == ground_truths))

def brier_score(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    """Calculate Brier Score (Mean Squared Error for probabilities)."""
    if len(predictions) == 0:
        return 0.0
    if _SKLEARN_AVAILABLE:
        return float(brier_score_loss(ground_truths, predictions))
    return float(np.mean((predictions - ground_truths) ** 2))

def coverage(predictions: np.ndarray, thresholds: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate empirical coverage for conformal prediction sets.
    Assuming thresholds define a prediction set [lower, upper].
    For binary classification with scores p, if threshold tau is given,
    we cover if p >= tau (class 1) or 1-p >= tau (class 0).
    """
    pass # Defined properly in conformal calibrator, here for completeness

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Receiver Operating Characteristic curve."""
    if len(np.unique(labels)) < 2 or not _SKLEARN_AVAILABLE:
        return float('nan')
    return float(roc_auc_score(labels, scores))

def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve."""
    if len(np.unique(labels)) < 2 or not _SKLEARN_AVAILABLE:
        return float('nan')
    return float(average_precision_score(labels, scores))

def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """False Positive Rate at 95% True Positive Rate."""
    if len(np.unique(labels)) < 2 or not _SKLEARN_AVAILABLE:
        return float('nan')
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    if idx < len(fpr):
        return float(fpr[idx])
    return float('nan')

def communication_reduction(messages_sent: int, n_agents: int, directed: bool = True) -> float:
    """
    Calculate communication reduction ratio.
    Max possible messages = N*(N-1) if directed, N*(N-1)/2 if undirected.
    """
    if n_agents <= 1:
        return 0.0
    max_msgs = n_agents * (n_agents - 1)
    if not directed:
        max_msgs /= 2.0
    
    if max_msgs == 0:
        return 0.0
    return float(1.0 - (messages_sent / max_msgs))


class LatencyTracker:
    """Tracks latency percentiles across repeated executions."""
    def __init__(self):
        self.records: Dict[str, List[float]] = {}
        
    def record(self, stage: str, elapsed_ms: float):
        if stage not in self.records:
            self.records[stage] = []
        self.records[stage].append(elapsed_ms)
        
    def get_percentiles(self, stage: str) -> Dict[str, float]:
        if stage not in self.records or not self.records[stage]:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        data = np.array(self.records[stage])
        return {
            "mean": float(np.mean(data)),
            "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99))
        }
    
    def get_all(self) -> Dict[str, Dict[str, float]]:
        return {stage: self.get_percentiles(stage) for stage in self.records}

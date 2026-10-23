"""
Temperature Scaling for Probability Calibration.
"""
import numpy as np

class TemperatureScaling:
    """
    Temperature Scaling optimizer using L-BFGS-B (via SciPy).
    Reference: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    """
    def __init__(self, initial_temp: float = 1.0):
        self.temperature = initial_temp
        self.is_fitted = False

    def _nll(self, temp: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> float:
        """Negative Log Likelihood with Temperature Scaling."""
        t = temp[0]
        scaled_logits = logits / t
        
        # stable softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # log likelihood
        N = logits.shape[0]
        p_correct = probs[np.arange(N), labels]
        p_correct = np.clip(p_correct, 1e-12, 1.0)
        
        return float(-np.mean(np.log(p_correct)))

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Fit temperature on a validation set.
        Args:
            logits: Array of shape (N, C)
            labels: Array of shape (N,)
        """
        try:
            from scipy.optimize import minimize
            
            # Bound temperature between 0.01 and 100
            res = minimize(
                self._nll, 
                np.array([self.temperature]), 
                args=(logits, labels), 
                method='L-BFGS-B', 
                bounds=[(0.01, 100.0)]
            )
            
            self.temperature = float(res.x[0])
            self.is_fitted = True
            
        except ImportError:
            raise ImportError("scipy is required for Temperature Scaling optimization.")

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits and return calibrated probabilities.
        """
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

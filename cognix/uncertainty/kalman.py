"""
Kalman Filtering for Uncertainty Tracking over time.
"""

class LinearKalmanFilter:
    """
    A basic Linear Kalman Filter for 1D state tracking (e.g., confidence or uncertainty tracking).
    """
    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 1e-2):
        self.Q = process_variance
        self.R = measurement_variance
        
        self.post_estimate = 0.5
        self.post_variance = 1.0
        
    def update(self, measurement: float) -> tuple[float, float]:
        """
        Process a new measurement and update state.
        Returns: (state_estimate, uncertainty_variance)
        """
        # Predict step
        prior_estimate = self.post_estimate
        prior_variance = self.post_variance + self.Q
        
        # Update step
        innovation = measurement - prior_estimate
        innovation_variance = prior_variance + self.R
        
        kalman_gain = prior_variance / innovation_variance
        
        self.post_estimate = prior_estimate + kalman_gain * innovation
        self.post_variance = (1 - kalman_gain) * prior_variance
        
        return self.post_estimate, self.post_variance

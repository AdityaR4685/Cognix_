import numpy as np
from .base import UncertaintyEstimate

class UncertaintyDecomposition:
    """
    Decomposes total predictive uncertainty into Aleatoric and Epistemic components.
    
    Reference:
    Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in 
    Bayesian Deep Learning for Computer Vision? NeurIPS 2017.
    """
    
    @staticmethod
    def decompose_classification(samples: np.ndarray) -> UncertaintyEstimate:
        """
        Decompose uncertainty for classification tasks using predictive entropy.
        
        Args:
            samples: Predictions from Bayesian approximation, shape (T, N, C) or (T, C).
                     Expected to be probabilities (softmax outputs).
        """
        samples = np.clip(samples, 1e-9, 1 - 1e-9)
        p_mean = np.mean(samples, axis=0)
        
        # Total Uncertainty: H[p(y|x)]
        total_unc = -np.sum(p_mean * np.log(p_mean), axis=-1)
        
        # Expected Entropy (Aleatoric): E[H[p(y|x,w)]]
        entropies = -np.sum(samples * np.log(samples), axis=-1)
        aleatoric_unc = np.mean(entropies, axis=0)
        
        # Epistemic Uncertainty = Total - Aleatoric
        epistemic_unc = total_unc - aleatoric_unc
        
        return UncertaintyEstimate(
            aleatoric=float(np.mean(aleatoric_unc)),
            epistemic=float(np.mean(epistemic_unc)),
            total=float(np.mean(total_unc)),
            raw_samples=samples,
            method="entropy_decomposition"
        )

    @staticmethod
    def decompose_regression(mu_samples: np.ndarray, var_samples: np.ndarray) -> UncertaintyEstimate:
        """
        Decompose uncertainty for regression models that predict both mean (mu) and variance (sigma^2).
        
        Args:
            mu_samples: Predicted means, shape (T, N) or (T,).
            var_samples: Predicted variances (sigma^2), shape (T, N) or (T,).
        """
        # Aleatoric (Data Uncertainty): Mean of predicted variances
        aleatoric_unc = np.mean(var_samples, axis=0)
        
        # Epistemic (Model Uncertainty): Variance of predicted means
        epistemic_unc = np.var(mu_samples, axis=0)
        
        # Total Uncertainty = Aleatoric + Epistemic
        total_unc = aleatoric_unc + epistemic_unc
        
        return UncertaintyEstimate(
            aleatoric=float(np.mean(aleatoric_unc)),
            epistemic=float(np.mean(epistemic_unc)),
            total=float(np.mean(total_unc)),
            raw_samples=np.stack([mu_samples, var_samples], axis=-1),
            method="variance_decomposition"
        )

import numpy as np
from typing import Optional

class OODDetector:
    """
    Out-Of-Distribution (OOD) Detection Module.
    
    Supported Methods:
    1. 'max_softmax' (Hendrycks & Gimpel, 2017)
    2. 'energy' (Liu et al., 2020)
    3. 'mahalanobis' (Lee et al., 2018)
    """
    def __init__(self, method: str = 'max_softmax'):
        valid_methods = ['max_softmax', 'energy', 'mahalanobis']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.method = method
        
        # For Mahalanobis
        self.class_means: Optional[np.ndarray] = None
        self.precision_matrix: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the detector on in-distribution training data. Required for Mahalanobis."""
        if self.method != 'mahalanobis':
            return # Other methods don't require fitting
            
        # Fit Mahalanobis parameters
        classes = np.unique(labels)
        self.class_means = []
        
        # Compute class centroids
        for c in classes:
            class_features = features[labels == c]
            self.class_means.append(np.mean(class_features, axis=0))
        self.class_means = np.array(self.class_means)
        
        # Compute tied precision matrix (inverse covariance)
        cov = np.cov(features, rowvar=False)
        # Add small jitter for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self.precision_matrix = np.linalg.inv(cov)

    def score(self, logits_or_features: np.ndarray) -> np.ndarray:
        """
        Compute OOD scores. Higher score means MORE likely to be In-Distribution (ID).
        Lower score means MORE likely to be OOD.
        """
        if self.method == 'max_softmax':
            # expects logits
            exp_x = np.exp(logits_or_features - np.max(logits_or_features, axis=-1, keepdims=True))
            softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            return np.max(softmax, axis=-1)
            
        elif self.method == 'energy':
            # expects logits
            # Energy score: T * log(sum(exp(logits / T)))
            # We use T=1. Score is negated energy so higher is ID.
            return np.log(np.sum(np.exp(logits_or_features), axis=-1))
            
        elif self.method == 'mahalanobis':
            # expects features
            if self.class_means is None or self.precision_matrix is None:
                raise ValueError("Mahalanobis detector must be fitted before scoring.")
            
            scores = []
            for x in logits_or_features:
                dists = []
                for mu in self.class_means:
                    diff = x - mu
                    dist = -np.dot(np.dot(diff.T, self.precision_matrix), diff)
                    dists.append(dist)
                scores.append(np.max(dists)) # max of negative distance
            return np.array(scores)
            
        return np.zeros(len(logits_or_features))

    def is_ood(self, logits_or_features: np.ndarray, threshold: float) -> np.ndarray:
        """Return boolean array where True indicates OOD sample (score < threshold)."""
        scores = self.score(logits_or_features)
        return scores < threshold

    def evaluate(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
        """Evaluate OOD detection performance (AUROC)."""
        try:
            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            y_scores = np.concatenate([id_scores, ood_scores])
            auroc = roc_auc_score(y_true, y_scores)
            return {"auroc": auroc}
        except ImportError:
            return {"error": "scikit-learn required for AUROC evaluation"}

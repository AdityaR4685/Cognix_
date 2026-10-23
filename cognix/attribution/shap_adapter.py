"""
SHAP and Tree SHAP Adapter for COGNIX.
Reference: Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
"""
from typing import Any, Callable
import numpy as np

try:
    import shap
except ImportError:
    shap = None

class SHAPAdapter:
    """
    Adapter for integrating SHAP values into the COGNIX explanation pipeline.
    """
    def __init__(self, model_type: str = "generic"):
        self.model_type = model_type
        self.explainer = None
        
    def fit(self, model: Any, background_data: np.ndarray):
        """
        Initialize the appropriate SHAP explainer based on model type.
        """
        if shap is None:
            raise ImportError(
                "SHAPAdapter requires 'shap'. "
                "Install with: pip install 'cognix-uncertainty-framework[explainability]'"
            )
        if self.model_type == "tree":
            # For XGBoost, LightGBM, RandomForest, etc.
            self.explainer = shap.TreeExplainer(model)
        elif self.model_type == "deep":
            # For PyTorch / TensorFlow deep models
            self.explainer = shap.DeepExplainer(model, background_data)
        else:
            # Generic fallback (Kernel SHAP)
            self.explainer = shap.KernelExplainer(model.predict if hasattr(model, 'predict') else model, background_data)
            
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for the given input data.
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer must be fitted with background data first.")
            
        shap_values = self.explainer.shap_values(input_data)
        return shap_values

    def interpret(self, shap_values: np.ndarray, feature_names: list[str] = None) -> dict[str, Any]:
        """
        Provide a structured interpretation of the top SHAP features.
        """
        if shap_values.ndim > 1:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_abs_shap = np.abs(shap_values)
            
        top_indices = np.argsort(mean_abs_shap)[::-1][:5]
        
        ranking = []
        for idx in top_indices:
            name = feature_names[idx] if feature_names else f"Feature_{idx}"
            ranking.append((name, float(mean_abs_shap[idx])))
            
        top_contributor = ranking[0][0] if ranking else None
        summary = f"Top feature by SHAP importance: {top_contributor}"
        
        return {
            'top_contributor': top_contributor,
            'ranking': ranking,
            'summary': summary
        }

"""
LIME (Local Interpretable Model-agnostic Explanations) Adapter for COGNIX.
Reference: Ribeiro et al. (2016). "Why Should I Trust You?" Explaining the Predictions of Any Classifier.
"""
from typing import Any, Callable
import numpy as np

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

class LIMEAdapter:
    """
    Adapter for integrating LIME explanations into the COGNIX pipeline.
    """
    def __init__(self, training_data: np.ndarray, feature_names: list[str] = None, class_names: list[str] = None, mode: str = "classification"):
        if LimeTabularExplainer is None:
            raise ImportError(
                "LIMEAdapter requires 'lime'. "
                "Install with: pip install 'cognix-uncertainty-framework[explainability]'"
            )
        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True
        )
        
    def compute(self, model_predict_fn: Callable, instance: np.ndarray, num_features: int = 5) -> dict[str, float]:
        """
        Generate LIME explanation for a specific instance.
        """
        # Ensure instance is 1D
        if instance.ndim > 1:
            instance = instance.flatten()
            
        exp = self.explainer.explain_instance(
            instance, 
            model_predict_fn, 
            num_features=num_features
        )
        
        # Convert LIME explanation list to dictionary
        return dict(exp.as_list())

    def interpret(self, explanation: dict[str, float]) -> dict[str, Any]:
        """
        Provide a structured interpretation of the LIME explanation.
        """
        if not explanation:
            return {}
            
        # Sort by absolute weight (importance)
        ranking = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        top_condition = ranking[0][0] if ranking else None
        
        summary = f"Primary condition driving the decision: {top_condition}"
        
        return {
            'top_contributor': top_condition,
            'ranking': ranking,
            'summary': summary
        }

"""
TreeExplainer implementation for global and local attribution.
"""

import shap
import pandas as pd
import numpy as np
from typing import Tuple

class AQIShapExplainer:
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)
        
    def get_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, shap.Explanation]:
        """Generate SHAP values and Explanation object."""
        explanation = self.explainer(X)
        return explanation.values, explanation
        
    def top_k_features(self, X: pd.DataFrame, k: int = 5) -> list:
        """Return top K global driving factors for AQI."""
        shap_vals = self.explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:k]
        return [X.columns[i] for i in top_indices]

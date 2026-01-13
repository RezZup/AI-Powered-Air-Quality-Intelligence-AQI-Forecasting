"""
Model evaluation metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 coefficient of determination."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None))) * 100)
    
    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 2)
    }

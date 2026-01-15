"""
Gradient boosted regression pipeline using XGBoost.
"""

import xgboost as xgb
from typing import Dict, Any
from src.config import DEFAULT_XGB_PARAMS

def train_xgboost_regressor(
    X_train, y_train, X_val, y_val, params: Dict[str, Any] = None
) -> xgb.XGBRegressor:
    """Train XGBoost with early stopping to prevent over-fitting."""
    model_params = params or DEFAULT_XGB_PARAMS.copy()
    
    model = xgb.XGBRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

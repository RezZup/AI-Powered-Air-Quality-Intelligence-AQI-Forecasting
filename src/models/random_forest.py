"""
Random Forest Regressor with Out-Of-Bag evaluation.
"""

from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any

def build_random_forest_model(
    n_estimators: int = 200,
    max_depth: int = 14,
    random_state: int = 42
) -> RandomForestRegressor:
    """Instantiate tuned Random Forest model."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )

"""
Isolation Forest detector for extreme pollution anomaly flagging.
"""

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

class AQIAnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns:
            1 for inliers, -1 for anomalous episodic spikes.
        """
        return self.model.fit_predict(X)

"""
Baseline linear regression and Ridge regularized benchmarks.
"""

from sklearn.linear_model import LinearRegression, Ridge
from typing import Tuple, Dict
import numpy as np

class BaselineAQIModel:
    def __init__(self, alpha: float = 1.0):
        self.ridge = Ridge(alpha=alpha)
        self.linear = LinearRegression()
        
    def fit(self, X_train, y_train):
        self.ridge.fit(X_train, y_train)
        self.linear.fit(X_train, y_train)
        
    def predict(self, X_val) -> Tuple[np.ndarray, np.ndarray]:
        return self.linear.predict(X_val), self.ridge.predict(X_val)

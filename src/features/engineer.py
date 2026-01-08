"""
Temporal lag and rolling statistical feature engineering.
"""

import pandas as pd
import numpy as np

def generate_lag_features(df: pd.DataFrame, lags: list = [1, 3, 7]) -> pd.DataFrame:
    """Construct leakage-safe past lag indicators per city."""
    df_feat = df.copy()
    for lag in lags:
        df_feat[f"AQI_lag_{lag}"] = df_feat.groupby("City")["AQI"].shift(lag)
        df_feat[f"PM25_lag_{lag}"] = df_feat.groupby("City")["PM2.5"].shift(lag)
    return df_feat

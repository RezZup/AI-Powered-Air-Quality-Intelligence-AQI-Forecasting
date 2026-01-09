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

"""
Temporal lag, rolling window, and cyclical feature engineering.
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

def generate_rolling_features(df: pd.DataFrame, windows: list = [3, 7]) -> pd.DataFrame:
    """Construct rolling moving averages and volatility."""
    df_feat = df.copy()
    for w in windows:
        df_feat[f"AQI_roll_mean_{w}"] = df_feat.groupby("City")["AQI"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df_feat[f"AQI_roll_std_{w}"] = df_feat.groupby("City")["AQI"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
    return df_feat

def add_cyclical_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonal sine and cosine features."""
    df_feat = df.copy()
    day_of_year = df_feat["Date"].dt.dayofyear
    df_feat["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_feat["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df_feat["month"] = df_feat["Date"].dt.month
    return df_feat

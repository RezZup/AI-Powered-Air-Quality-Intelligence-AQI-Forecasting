"""Tests for feature generation pipelines."""
import pandas as pd
from src.features.engineer import generate_lag_features, generate_rolling_features

def test_lag_computation_correctness():
    df = pd.DataFrame({
        "City": ["Delhi", "Delhi", "Delhi", "Delhi"],
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        "AQI": [100.0, 150.0, 200.0, 250.0],
        "PM2.5": [50.0, 75.0, 100.0, 125.0]
    })
    lagged = generate_lag_features(df, lags=[1])
    assert pd.isna(lagged["AQI_lag_1"].iloc[0])
    assert lagged["AQI_lag_1"].iloc[1] == 100.0
    assert lagged["AQI_lag_1"].iloc[2] == 150.0

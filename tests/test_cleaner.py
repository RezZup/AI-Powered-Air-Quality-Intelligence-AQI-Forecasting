"""Tests for data cleaning and interpolation."""
import pandas as pd
import numpy as np
from src.data.cleaner import clean_station_time_series

def test_interpolation_respects_city_boundaries():
    data = {
        "City": ["CityA", "CityA", "CityA", "CityB", "CityB"],
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"]),
        "PM2.5": [100.0, np.nan, 200.0, 50.0, np.nan],
        "AQI": [120.0, np.nan, 220.0, 60.0, np.nan]
    }
    df = pd.DataFrame(data)
    cleaned = clean_station_time_series(df)
    
    city_a_mid = cleaned[(cleaned["City"] == "CityA") & (cleaned["Date"] == "2026-01-02")]["PM2.5"].values[0]
    assert np.isclose(city_a_mid, 150.0)

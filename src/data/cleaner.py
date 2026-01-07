"""
Data imputation ensuring temporal continuity per city.
"""

import pandas as pd
from typing import List
from src.config import CORE_POLLUTANTS

def clean_station_time_series(df: pd.DataFrame, max_interpolation_gap: int = 7) -> pd.DataFrame:
    """
    Perform per-city time-based linear interpolation.
    Prevents cross-city data contamination.
    """
    df_clean = df.copy()
    numeric_cols = [c for c in CORE_POLLUTANTS if c in df_clean.columns]
    if "AQI" in df_clean.columns:
        numeric_cols.append("AQI")
        
    cleaned_frames = []
    for city, group in df_clean.groupby("City"):
        group = group.sort_values("Date").set_index("Date")
        # Time-based interpolation within city boundary
        group[numeric_cols] = group[numeric_cols].interpolate(
            method="time", limit=max_interpolation_gap, limit_direction="both"
        )
        cleaned_frames.append(group.reset_index())
        
    return pd.concat(cleaned_frames, ignore_index=True)

"""
Data imputation ensuring temporal continuity per city with fallback logic.
"""

import pandas as pd
from typing import List
from src.config import CORE_POLLUTANTS

def clean_station_time_series(df: pd.DataFrame, max_interpolation_gap: int = 7) -> pd.DataFrame:
    """
    Perform per-city time-based linear interpolation with station median fallback.
    """
    df_clean = df.copy()
    numeric_cols = [c for c in CORE_POLLUTANTS if c in df_clean.columns]
    if "AQI" in df_clean.columns:
        numeric_cols.append("AQI")
        
    cleaned_frames = []
    for city, group in df_clean.groupby("City"):
        group = group.sort_values("Date").set_index("Date")
        # Step 1: Linear temporal interpolation
        group[numeric_cols] = group[numeric_cols].interpolate(
            method="time", limit=max_interpolation_gap, limit_direction="both"
        )
        # Step 2: City median fallback for boundary edges
        for col in numeric_cols:
            if group[col].isna().any():
                group[col] = group[col].fillna(group[col].median())
                
        cleaned_frames.append(group.reset_index())
        
    result = pd.concat(cleaned_frames, ignore_index=True)
    # Global median as safety net for sparse stations
    for col in numeric_cols:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())
            
    return result

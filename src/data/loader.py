"""
Raw dataset loader with validation and station filtering.
"""

import os
import pandas as pd
from typing import Optional, List

REQUIRED_COLUMNS = ["City", "Date", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"]

def load_cpcb_data(file_path: str, target_cities: Optional[List[str]] = None) -> pd.DataFrame:
    """Load and validate CPCB city-day dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path not found: {file_path}")
    
    df = pd.read_csv(file_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing essential columns: {missing_cols}")
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
    
    if target_cities:
        df = df[df["City"].isin(target_cities)].reset_index(drop=True)
        
    return df

"""Tests for data loader functionality."""
import pytest
import pandas as pd
import tempfile
import os
from src.data.loader import load_cpcb_data

def test_loader_validates_columns():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("City,Date,PM2.5\nDelhi,2026-01-01,150.0\n")
        tmp_name = f.name
        
    try:
        with pytest.raises(ValueError):
            load_cpcb_data(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

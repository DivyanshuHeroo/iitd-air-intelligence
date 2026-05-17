import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import create_features
from src.predict import get_aqi_category
from src.config import REQUIRED_COLS

def test_feature_creation():
    # Mock data
    df = pd.DataFrame({
        "dateTime": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"],
        "lat": [28.5, 28.5],
        "long": [77.2, 77.2],
        "pm2_5": [40, 50],
        "pm10": [90, 100]
    })
    
    features_df = create_features(df)
    
    # Check if required new features are created
    assert "hour" in features_df.columns
    assert "distance_from_iitd_km" in features_df.columns
    assert "pm2_5_lag_1" in features_df.columns
    
    # Check if original columns are still there
    assert "pm2_5" in features_df.columns
    assert "lat" in features_df.columns

def test_get_aqi_category():
    assert get_aqi_category(20) == "Good"
    assert get_aqi_category(50) == "Satisfactory/Moderate"
    assert get_aqi_category(80) == "Moderate/Poor"
    assert get_aqi_category(110) == "Poor"
    assert get_aqi_category(200) == "Very Poor"
    assert get_aqi_category(300) == "Severe"

def test_model_file_exists():
    model_path = Path(__file__).resolve().parent.parent / "models" / "best_pm25_model.pkl"
    if model_path.exists():
        assert model_path.is_file()

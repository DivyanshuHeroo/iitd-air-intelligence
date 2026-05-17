import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src import config
import os

_MODEL = None
_FEATURE_COLS = None

def load_model_and_features():
    global _MODEL, _FEATURE_COLS
    
    model_path = config.MODELS_DIR / "best_pm25_model.pkl"
    features_path = config.MODELS_DIR / "feature_columns.json"
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise Exception("Model or feature columns not found. Train the model first.")
        
    if _MODEL is None:
        _MODEL = joblib.load(model_path)
    if _FEATURE_COLS is None:
        with open(features_path, "r") as f:
            _FEATURE_COLS = json.load(f)
            
    return _MODEL, _FEATURE_COLS

def get_aqi_category(pm2_5_val):
    if pm2_5_val <= 30:
        return "Good"
    elif pm2_5_val <= 60:
        return "Satisfactory/Moderate"
    elif pm2_5_val <= 90:
        return "Moderate/Poor"
    elif pm2_5_val <= 120:
        return "Poor"
    elif pm2_5_val <= 250:
        return "Very Poor"
    else:
        return "Severe"

def predict_pm25(input_data: dict) -> dict:
    model, feature_cols = load_model_and_features()
    
    # Ensure all required features are present
    df = pd.DataFrame([input_data])
    
    # Missing features filled with 0 (or some default) just to prevent error
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    # Order columns
    df = df[feature_cols]
    
    pred = model.predict(df)[0]
    
    return {
        "predicted_pm2_5": float(pred),
        "category": get_aqi_category(pred),
        "model_name": type(model).__name__,
        "input_location": {"lat": input_data.get("lat"), "long": input_data.get("long")}
    }

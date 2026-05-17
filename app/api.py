from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_pm25, load_model_and_features
from src.features import haversine_distance
from src import config

app = FastAPI(title="IITD Air Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def classify_pm25_category(pm25: float) -> str:
    if pm25 <= 30: return "Good"
    elif pm25 <= 60: return "Satisfactory"
    elif pm25 <= 90: return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else: return "Severe"

def get_category_interpretation(category: str) -> str:
    interpretations = {
        "Good": "Minimal impact.",
        "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
        "Moderate": "May cause breathing discomfort to people with lung disease.",
        "Poor": "May cause breathing discomfort to most people on prolonged exposure.",
        "Very Poor": "May cause respiratory illness on prolonged exposure.",
        "Severe": "May cause respiratory impact even on healthy people."
    }
    return interpretations.get(category, "")

class PredictionInput(BaseModel):
    lat: float
    long: float
    temperature: float
    humidity: float
    pressure: float
    hour: int
    day: int
    dayofweek: int
    month: int
    is_weekend: int = 0
    pm2_5_lag_1: float
    pm2_5_lag_3: float
    pm2_5_lag_6: float
    pm10_lag_1: float
    pm10_lag_3: float
    pm10_lag_6: float
    pm2_5_rolling_3: float
    pm2_5_rolling_6: float
    pm10_rolling_3: float
    pm10_rolling_6: float

@app.get("/")
def read_root():
    return {
        "project": "IITD Air Intelligence",
        "description": "Hyperlocal PM2.5 Prediction for Delhi-NCR",
        "endpoints": ["/health", "/model_info", "/predict_pm25"]
    }

@app.get("/health")
def health_check():
    model_loaded = False
    feature_columns_loaded = False
    
    try:
        load_model_and_features()
        model_loaded = True
        feature_columns_loaded = True
    except Exception:
        pass
        
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "feature_columns_loaded": feature_columns_loaded
    }

@app.get("/model_info")
def model_info():
    try:
        model, features = load_model_and_features()
        return {
            "model_path": str(config.BEST_MODEL_PATH),
            "num_features": len(features),
            "feature_names": features,
            "target_variable": config.TARGET_COL,
            "warning": "This model is for educational/portfolio demonstration only. It is not an official health system."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_pm25")
def predict(input_data: PredictionInput):
    try:
        # Compute derived features
        dist = haversine_distance(input_data.lat, input_data.long, config.IITD_LAT, config.IITD_LONG)
        hour_sin = np.sin(2 * np.pi * input_data.hour / 24)
        hour_cos = np.cos(2 * np.pi * input_data.hour / 24)
        dayofweek_sin = np.sin(2 * np.pi * input_data.dayofweek / 7)
        dayofweek_cos = np.cos(2 * np.pi * input_data.dayofweek / 7)
        
        full_input = input_data.dict()
        full_input["distance_from_iitd_km"] = dist
        full_input["hour_sin"] = hour_sin
        full_input["hour_cos"] = hour_cos
        full_input["dayofweek_sin"] = dayofweek_sin
        full_input["dayofweek_cos"] = dayofweek_cos
        
        result = predict_pm25(full_input)
        pm25 = float(result["predicted_pm2_5"])
        category = classify_pm25_category(pm25)
        interpretation = get_category_interpretation(category)
        
        return {
            "predicted_pm2_5": pm25,
            "category": category,
            "interpretation": interpretation,
            "model_name": result["model_name"],
            "input_location": result["input_location"],
            "note": "For informational purposes only."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

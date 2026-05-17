import pandas as pd
import numpy as np
from src.config import IITD_LAT, IITD_LONG

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Time features
    if "dateTime" in df.columns:
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df["hour"] = df["dateTime"].dt.hour
        df["day"] = df["dateTime"].dt.day
        df["dayofweek"] = df["dateTime"].dt.dayofweek
        df["month"] = df["dateTime"].dt.month
        df["is_weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
        
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # 2. Spatial and weather features
    if "lat" in df.columns and "long" in df.columns:
        df["location_id"] = df["lat"].round(3).astype(str) + "_" + df["long"].round(3).astype(str)
        df["distance_from_iitd_km"] = haversine_distance(df["lat"], df["long"], IITD_LAT, IITD_LONG)
        df["lat_rounded"] = df["lat"].round(2)
        df["long_rounded"] = df["long"].round(2)
        
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    
    # 3. Lags, rolling, and target definition
    if "location_id" in df.columns and "dateTime" in df.columns:
        df = df.sort_values(by=["location_id", "dateTime"])
        grouped = df.groupby("location_id")
        
        # TARGET: Predict next-hour PM2.5
        if "pm2_5" in df.columns:
            df["pm2_5_target_next_1h"] = grouped["pm2_5"].shift(-1)
            
            # Categories for classification
            bins = [0, 30, 60, 90, 120, 250, 10000]
            labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
            df["pm2_5_category_next_1h"] = pd.cut(df["pm2_5_target_next_1h"], bins=bins, labels=labels, right=True)
            
            # Lag features
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f"pm2_5_lag_{lag}"] = grouped["pm2_5"].shift(lag)
                
            # Rolling features
            for window in [3, 6, 12, 24]:
                df[f"pm2_5_rolling_mean_{window}"] = grouped["pm2_5"].shift(1).rolling(window=window, min_periods=1).mean()
                df[f"pm2_5_rolling_std_{window}"] = grouped["pm2_5"].shift(1).rolling(window=window, min_periods=1).std()
                
            # Expanding means (leakage safe by shifting)
            df["location_expanding_pm25_mean"] = grouped["pm2_5"].shift(1).expanding().mean()
            
            # Grouped expanding means
            df["location_hour"] = df["location_id"] + "_" + df["hour"].astype(str)
            df["location_hour_expanding_pm25_mean"] = df.groupby("location_hour")["pm2_5"].shift(1).expanding().mean()
            df["hour_expanding_pm25_mean"] = df.groupby("hour")["pm2_5"].shift(1).expanding().mean()
            
        if "pm10" in df.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f"pm10_lag_{lag}"] = grouped["pm10"].shift(lag)
            for window in [3, 6, 12, 24]:
                df[f"pm10_rolling_mean_{window}"] = grouped["pm10"].shift(1).rolling(window=window, min_periods=1).mean()
                df[f"pm10_rolling_std_{window}"] = grouped["pm10"].shift(1).rolling(window=window, min_periods=1).std()
                
    if "dateTime" in df.columns:
        df = df.sort_values(by="dateTime").reset_index(drop=True)
        
    return df

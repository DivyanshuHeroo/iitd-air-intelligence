import pandas as pd
import numpy as np
from src.config import IITD_LAT, IITD_LONG

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates temporal, spatial, weather, lag, and rolling features.
    Assumes df is already sorted by dateTime and cleaned.
    """
    df = df.copy()
    
    # 1. Time features
    if "dateTime" in df.columns:
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df["hour"] = df["dateTime"].dt.hour
        df["day"] = df["dateTime"].dt.day
        df["dayofweek"] = df["dateTime"].dt.dayofweek
        df["month"] = df["dateTime"].dt.month
        df["is_weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
        
        # 2. Cyclical time features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    # 3. Spatial features
    if "lat" in df.columns and "long" in df.columns:
        df["location_id"] = df["lat"].round(3).astype(str) + "_" + df["long"].round(3).astype(str)
        df["distance_from_iitd_km"] = haversine_distance(df["lat"], df["long"], IITD_LAT, IITD_LONG)
    
    # Sort data correctly to prevent leakage (first by location, then by time)
    if "location_id" in df.columns and "dateTime" in df.columns:
        df = df.sort_values(by=["location_id", "dateTime"])
        
        # Generate grouped lags and rolling features
        grouped = df.groupby("location_id")
        
        # 5. Pollution lag features
        if "pm2_5" in df.columns:
            df["pm2_5_lag_1"] = grouped["pm2_5"].shift(1)
            df["pm2_5_lag_3"] = grouped["pm2_5"].shift(3)
            df["pm2_5_lag_6"] = grouped["pm2_5"].shift(6)
            
            # 6. Rolling features
            df["pm2_5_rolling_3"] = grouped["pm2_5"].shift(1).rolling(window=3, min_periods=1).mean()
            df["pm2_5_rolling_6"] = grouped["pm2_5"].shift(1).rolling(window=6, min_periods=1).mean()
            
        if "pm10" in df.columns:
            df["pm10_lag_1"] = grouped["pm10"].shift(1)
            df["pm10_lag_3"] = grouped["pm10"].shift(3)
            df["pm10_lag_6"] = grouped["pm10"].shift(6)
            
            df["pm10_rolling_3"] = grouped["pm10"].shift(1).rolling(window=3, min_periods=1).mean()
            df["pm10_rolling_6"] = grouped["pm10"].shift(1).rolling(window=6, min_periods=1).mean()
            
    # We will keep them and let the training code drop NA
    # Re-sort purely by dateTime just in case
    if "dateTime" in df.columns:
        df = df.sort_values(by="dateTime").reset_index(drop=True)
        
    return df

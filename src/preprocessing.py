import pandas as pd
import numpy as np
from src.config import REQUIRED_COLS

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw pollution data by standardizing types, dropping missing targets,
    and removing invalid values.
    """
    print(f"Original shape: {df.shape}")
    
    # Check required columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
    
    # Keep only required columns that are present
    cols_to_keep = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[cols_to_keep].copy()
    
    # Parse dateTime
    if "dateTime" in df.columns:
        df["dateTime"] = pd.to_datetime(df["dateTime"], errors='coerce')
        # Drop rows where dateTime could not be parsed
        df.dropna(subset=["dateTime"], inplace=True)
        # Sort by dateTime
        df.sort_values(by="dateTime", inplace=True)
        
    # Drop rows with missing target pm2_5
    if "pm2_5" in df.columns:
        initial_count = len(df)
        df.dropna(subset=["pm2_5"], inplace=True)
        print(f"Dropped {initial_count - len(df)} rows due to missing pm2_5")
    
    # Remove impossible values
    if "pm2_5" in df.columns:
        df = df[df["pm2_5"] >= 0]
    if "pm10" in df.columns:
        df = df[df["pm10"] >= 0]
    if "pm1_0" in df.columns:
        df = df[df["pm1_0"] >= 0]
        
    if "lat" in df.columns and "long" in df.columns:
        # Valid bounds for India/Delhi approximate
        df = df[(df["lat"] >= 28.0) & (df["lat"] <= 29.0)]
        df = df[(df["long"] >= 76.0) & (df["long"] <= 78.0)]
        
    if "humidity" in df.columns:
        df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]
        
    # Optional constraints
    # temperature -10 to 55 C
    # pressure 900 to 1100 mb
    
    print(f"Cleaned shape: {df.shape}")
    print("Missing value counts after cleaning:")
    print(df.isnull().sum())
    
    if "dateTime" in df.columns:
        print(f"Date range: {df['dateTime'].min()} to {df['dateTime'].max()}")
        
    if "lat" in df.columns and "long" in df.columns:
        # Number of unique approximate locations
        locs = df.groupby([df['lat'].round(3), df['long'].round(3)]).size()
        print(f"Number of unique approximate locations: {len(locs)}")
        
    return df

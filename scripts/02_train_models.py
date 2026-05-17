import sys
import os
import json
import joblib
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.train import train_classifiers

def main():
    print("Loading featured data...")
    features_path = config.PROCESSED_DATA_DIR / "features_pollution_data.csv"
    if not features_path.exists():
        print(f"Error: Features data not found at {features_path}")
        sys.exit(1)
        
    df = pd.read_csv(features_path)
    
    # Define features
    feature_cols = [
        "hour", "day", "dayofweek", "month", "is_weekend",
        "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
        "month_sin", "month_cos",
        "lat", "long", "lat_rounded", "long_rounded", "distance_from_iitd_km",
        "pressure", "temperature", "humidity", "temp_humidity_interaction",
        "location_expanding_pm25_mean", "location_hour_expanding_pm25_mean", "hour_expanding_pm25_mean",
        "pm2_5_lag_1_minus_lag_3", "pm2_5_lag_1_minus_lag_6",
        "pm10_lag_1_minus_lag_3", "pm10_lag_1_minus_lag_6",
        "previous_pm25_category", "previous_pm25_category_3h", "previous_pm25_category_6h"
    ]
    
    # Add lag and rolling features if they exist
    for col in [
        "pm2_5_lag_1", "pm2_5_lag_2", "pm2_5_lag_3", "pm2_5_lag_6", "pm2_5_lag_12", "pm2_5_lag_24",
        "pm10_lag_1", "pm10_lag_2", "pm10_lag_3", "pm10_lag_6", "pm10_lag_12", "pm10_lag_24",
        "pm2_5_rolling_mean_3", "pm2_5_rolling_mean_6", "pm2_5_rolling_mean_12", "pm2_5_rolling_mean_24",
        "pm2_5_rolling_std_3", "pm2_5_rolling_std_6", "pm2_5_rolling_std_12", "pm2_5_rolling_std_24",
        "pm10_rolling_mean_3", "pm10_rolling_mean_6", "pm10_rolling_mean_12", "pm10_rolling_mean_24",
        "pm10_rolling_std_3", "pm10_rolling_std_6", "pm10_rolling_std_12", "pm10_rolling_std_24"
    ]:
        if col in df.columns:
            feature_cols.append(col)
            
    # Filter out missing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
            
    print(f"Using {len(feature_cols)} features for training.")
    
    # Save feature columns list
    with open(config.MODELS_DIR / "classifier_feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
        
    print("Training Classification Models...")
    
    class_metrics, best_models = train_classifiers(df, feature_cols)
    
    if not class_metrics.empty:
        print("\nClassification Results:")
        print(class_metrics.to_string(index=False))
        class_metrics.to_csv(config.METRICS_DIR / "category_classification_results.csv", index=False)
        
        # Save best models
        if "next_1h_6class" in best_models:
            joblib.dump(best_models["next_1h_6class"], config.MODELS_DIR / "best_pm25_category_classifier.pkl")
        
        if "next_1h_binary" in best_models:
            joblib.dump(best_models["next_1h_binary"], config.MODELS_DIR / "best_binary_classifier.pkl")
            
        if "next_1h_3class" in best_models:
            joblib.dump(best_models["next_1h_3class"], config.MODELS_DIR / "best_3class_classifier.pkl")
        
if __name__ == "__main__":
    main()

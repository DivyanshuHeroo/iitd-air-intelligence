import sys
import os
import json
import joblib
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.train import train_and_evaluate

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
        "location_expanding_pm25_mean", "location_hour_expanding_pm25_mean", "hour_expanding_pm25_mean"
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
    with open(config.MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
        
    print("Training models...")
    # New target: Predict next-hour PM2.5
    target_col = "pm2_5_target_next_1h" if "pm2_5_target_next_1h" in df.columns else config.TARGET_COL
    
    metrics_df, class_metrics, trained_models, test_df = train_and_evaluate(df, target_col, feature_cols)
    
    print("\nModel Leaderboard:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_df.to_csv(config.METRICS_DIR / "model_results.csv", index=False)
    
    if not class_metrics.empty:
        print("\nClassification Results:")
        print(class_metrics.to_string(index=False))
        class_metrics.to_csv(config.METRICS_DIR / "category_classification_results.csv", index=False)
    
    # Save best ML model (exclude persistence)
    ml_models_df = metrics_df[metrics_df["Model"] != "Persistence_Baseline"]
    if not ml_models_df.empty:
        best_model_name = ml_models_df.iloc[0]["Model"]
        print(f"\nBest ML model: {best_model_name}")
        
        # Save model
        best_model = trained_models[best_model_name]
        joblib.dump(best_model, config.MODELS_DIR / "best_pm25_model.pkl")
        
        # Save test predictions for evaluation
        test_df.to_csv(config.PROCESSED_DATA_DIR / "test_predictions.csv", index=False)
        
if __name__ == "__main__":
    main()

import sys
import os
import json
import joblib
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.evaluate import (
    plot_pm25_distribution,
    plot_daily_trend,
    plot_hourly_pattern,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_model_comparison,
    plot_feature_importance,
    generate_error_analysis
)

def main():
    print("Loading data for evaluation...")
    
    clean_path = config.PROCESSED_DATA_DIR / "clean_pollution_data.csv"
    if clean_path.exists():
        clean_df = pd.read_csv(clean_path)
        if "dateTime" in clean_df.columns:
            clean_df["dateTime"] = pd.to_datetime(clean_df["dateTime"])
        
        print("Plotting EDA figures...")
        plot_pm25_distribution(clean_df)
        plot_daily_trend(clean_df)
        plot_hourly_pattern(clean_df)
    
    metrics_path = config.METRICS_DIR / "model_results.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        print("Plotting model comparison...")
        plot_model_comparison(metrics_df)
        
        ml_models_df = metrics_df[metrics_df["Model"] != "Persistence_Baseline"]
        if not ml_models_df.empty:
            best_model_name = ml_models_df.iloc[0]["Model"]
            print(f"Evaluating best model: {best_model_name}")
            
            test_preds_path = config.PROCESSED_DATA_DIR / "test_predictions.csv"
            if test_preds_path.exists():
                test_df = pd.read_csv(test_preds_path)
                print("Plotting actual vs predicted and residuals...")
                plot_actual_vs_predicted(test_df, best_model_name)
                plot_residual_distribution(test_df, best_model_name)
                
                print("Generating error analysis...")
                generate_error_analysis(test_df, best_model_name)
                
            model_path = config.MODELS_DIR / "best_pm25_model.pkl"
            features_path = config.MODELS_DIR / "feature_columns.json"
            
            if model_path.exists() and features_path.exists():
                model = joblib.load(model_path)
                with open(features_path, "r") as f:
                    feature_names = json.load(f)
                    
                print("Plotting feature importance...")
                plot_feature_importance(model, feature_names)
                
if __name__ == "__main__":
    main()

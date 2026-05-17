import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return np.nan
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def train_and_evaluate(df: pd.DataFrame, target_col: str, feature_cols: list):
    """
    Trains models and returns metrics, models, and test data with predictions.
    """
    df = df.dropna(subset=[target_col] + feature_cols).copy()
    df = df.sort_values(by="dateTime").reset_index(drop=True)
    
    # Time-based split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    models = {
        "Persistence_Baseline": "persistence", # Special case
        "Ridge_Regression": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    metrics = []
    trained_models = {}
    
    for name, model in models.items():
        if name == "Persistence_Baseline":
            if "pm2_5_lag_1" in feature_cols:
                y_pred = test_df["pm2_5_lag_1"]
                # For training baseline, pred is train_df lag
                train_pred = train_df["pm2_5_lag_1"]
            else:
                continue
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_models[name] = model
            
        test_df[f"{name}_pred"] = y_pred
            
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        metrics.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values(by="MAE")
    
    return metrics_df, trained_models, test_df

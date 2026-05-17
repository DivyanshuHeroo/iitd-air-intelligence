import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import joblib

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return np.nan
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def train_and_evaluate(df: pd.DataFrame, target_col: str, feature_cols: list):
    """
    Trains models with time-series CV and returns metrics, models, and test data predictions.
    """
    df = df.dropna(subset=[target_col] + feature_cols).copy()
    df = df.sort_values(by="dateTime").reset_index(drop=True)
    
    # Cap target at 99th percentile for robust training
    p99 = df[target_col].quantile(0.99)
    df["target_clipped"] = np.clip(df[target_col], 0, p99)
    
    # Time-based split for final evaluation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df["target_clipped"] # Use clipped target for training
    y_train_true = train_df[target_col] # True for evaluation
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col] # Use unclipped target for real evaluation
    
    models = {
        "Persistence_Baseline": "persistence",
        "Ridge_Regression": Ridge(alpha=10.0),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Extra_Trees": ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42),
        "XGBoost_Default": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    }
    
    metrics = []
    trained_models = {}
    
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Calculate baseline metrics
        if name == "Persistence_Baseline":
            if "pm2_5_lag_1" in feature_cols:
                y_pred = test_df["pm2_5_lag_1"]
            else:
                continue
                
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            baseline_mae = mae
            baseline_rmse = rmse
            
            metrics.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape,
                "MAE_improvement_percent": 0.0,
                "RMSE_improvement_percent": 0.0
            })
            test_df[f"{name}_pred"] = y_pred
            continue
            
        # Fit on full training data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_models[name] = model
        
        test_df[f"{name}_pred"] = y_pred
            
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Compute improvement vs baseline
        mae_imp = ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae else 0
        rmse_imp = ((baseline_rmse - rmse) / baseline_rmse) * 100 if baseline_rmse else 0
        
        metrics.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
            "MAE_improvement_percent": mae_imp,
            "RMSE_improvement_percent": rmse_imp
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values(by="MAE")
    
    # -----------------------------------
    # Classification Task
    # -----------------------------------
    print("Training Classification Task...")
    target_class = "pm2_5_category_next_1h"
    if target_class in df.columns:
        df_class = df.dropna(subset=[target_class] + feature_cols).copy()
        
        train_df_c = df_class.iloc[:split_idx]
        test_df_c = df_class.iloc[split_idx:]
        
        Xc_train = train_df_c[feature_cols]
        yc_train = train_df_c[target_class]
        Xc_test = test_df_c[feature_cols]
        yc_test = test_df_c[target_class]
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(Xc_train, yc_train)
        yc_pred = clf.predict(Xc_test)
        
        acc = accuracy_score(yc_test, yc_pred)
        macro_f1 = f1_score(yc_test, yc_pred, average="macro")
        weighted_f1 = f1_score(yc_test, yc_pred, average="weighted")
        
        class_metrics = pd.DataFrame([{
            "Model": "RandomForestClassifier",
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1
        }])
        
        trained_models["Classifier"] = clf
    else:
        class_metrics = pd.DataFrame()
    
    return metrics_df, class_metrics, trained_models, test_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR

def plot_pm25_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["pm2_5"], bins=50, kde=True)
    plt.title("PM2.5 Distribution")
    plt.xlabel("PM2.5")
    plt.savefig(FIGURES_DIR / "pm25_distribution.png")
    plt.close()

def plot_daily_trend(df: pd.DataFrame):
    if "dateTime" in df.columns:
        df_daily = df.groupby(df["dateTime"].dt.date)["pm2_5"].mean().reset_index()
        plt.figure(figsize=(12, 6))
        plt.plot(df_daily["dateTime"], df_daily["pm2_5"])
        plt.title("Daily Average PM2.5 Trend")
        plt.xlabel("Date")
        plt.ylabel("PM2.5")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "daily_pm25_trend.png")
        plt.close()

def plot_hourly_pattern(df: pd.DataFrame):
    if "hour" in df.columns:
        df_hourly = df.groupby("hour")["pm2_5"].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x="hour", y="pm2_5", data=df_hourly, color="skyblue")
        plt.title("Average Hourly PM2.5 Pattern")
        plt.xlabel("Hour of Day")
        plt.ylabel("PM2.5")
        plt.savefig(FIGURES_DIR / "hourly_pm25_pattern.png")
        plt.close()

def plot_actual_vs_predicted(df: pd.DataFrame, best_model_name: str):
    pred_col = f"{best_model_name}_pred"
    if pred_col in df.columns:
        plt.figure(figsize=(10, 10))
        plt.scatter(df["pm2_5"], df[pred_col], alpha=0.5)
        
        max_val = max(df["pm2_5"].max(), df[pred_col].max())
        plt.plot([0, max_val], [0, max_val], 'r--') # diagonal
        
        plt.title(f"Actual vs Predicted PM2.5 ({best_model_name})")
        plt.xlabel("Actual PM2.5")
        plt.ylabel("Predicted PM2.5")
        plt.savefig(FIGURES_DIR / "actual_vs_predicted.png")
        plt.close()

def plot_residual_distribution(df: pd.DataFrame, best_model_name: str):
    pred_col = f"{best_model_name}_pred"
    if pred_col in df.columns:
        residuals = df["pm2_5"] - df[pred_col]
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=50, kde=True)
        plt.title(f"Residual Distribution ({best_model_name})")
        plt.xlabel("Residual (Actual - Predicted)")
        plt.savefig(FIGURES_DIR / "residual_distribution.png")
        plt.close()

def plot_model_comparison(metrics_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="MAE", y="Model", data=metrics_df)
    plt.title("Model Comparison (MAE)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png")
    plt.close()
    
def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png")
        plt.close()

def generate_error_analysis(test_df: pd.DataFrame, best_model_name: str):
    pred_col = f"{best_model_name}_pred"
    if pred_col in test_df.columns:
        error_df = pd.DataFrame()
        error_df["dateTime"] = test_df.get("dateTime")
        error_df["lat"] = test_df.get("lat")
        error_df["long"] = test_df.get("long")
        error_df["actual_pm2_5"] = test_df["pm2_5"]
        error_df["predicted_pm2_5"] = test_df[pred_col]
        error_df["absolute_error"] = np.abs(test_df["pm2_5"] - test_df[pred_col])
        error_df["hour"] = test_df.get("hour")
        error_df["dayofweek"] = test_df.get("dayofweek")
        
        error_df.to_csv(METRICS_DIR / "error_analysis.csv", index=False)

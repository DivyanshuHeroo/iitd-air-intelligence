import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_ROOT  # Alias

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODEL_DIR = PROJECT_ROOT / "models"
MODELS_DIR = MODEL_DIR  # Alias

REPORT_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR = REPORT_DIR  # Alias

FIGURE_DIR = REPORT_DIR / "figures"
FIGURES_DIR = FIGURE_DIR  # Alias

METRIC_DIR = REPORT_DIR / "metrics"
METRICS_DIR = METRIC_DIR  # Alias

# Specific File Paths
BEST_MODEL_PATH = MODEL_DIR / "best_pm25_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_RESULTS_PATH = METRIC_DIR / "model_results.csv"

# Coordinates
IITD_LAT = 28.5450
IITD_LONG = 77.1926

# Model Columns
TARGET_COL = "pm2_5"
REQUIRED_COLS = ["dateTime", "lat", "long", "pressure", "temperature", "humidity", "pm1_0", "pm2_5", "pm10"]

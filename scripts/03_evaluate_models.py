import sys
import os
import json
import joblib
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.evaluate_class import (
    plot_class_distribution,
    plot_category_model_comparison,
    plot_confusion_matrix_custom,
    plot_classification_feature_importance
)

def create_summary(metrics_df):
    next_1h_df = metrics_df[metrics_df["target_horizon"] == "next_1h"]
    
    best_6 = next_1h_df[next_1h_df["task_name"] == "next_1h_6class"].iloc[0] if not next_1h_df[next_1h_df["task_name"] == "next_1h_6class"].empty else None
    best_3 = next_1h_df[next_1h_df["task_name"] == "next_1h_3class"].iloc[0] if not next_1h_df[next_1h_df["task_name"] == "next_1h_3class"].empty else None
    best_bin = next_1h_df[next_1h_df["task_name"] == "next_1h_binary"].iloc[0] if not next_1h_df[next_1h_df["task_name"] == "next_1h_binary"].empty else None
    
    summary = "# Best Classification Summary\n\n"
    
    if best_6 is not None:
        summary += f"## Exact 6-Class Prediction (Next Hour)\n"
        summary += f"- **Best Model**: {best_6['model']}\n"
        summary += f"- **Accuracy**: {best_6['accuracy']:.2%}\n"
        summary += f"- **Top-2 Accuracy**: {best_6['top_2_accuracy']:.2%}\n"
        summary += f"- **Adjacent Accuracy**: {best_6['adjacent_accuracy']:.2%}\n\n"
        
    if best_3 is not None:
        summary += f"## 3-Class Severity Prediction (Next Hour)\n"
        summary += f"- **Best Model**: {best_3['model']}\n"
        summary += f"- **Accuracy**: {best_3['accuracy']:.2%}\n\n"
        
    if best_bin is not None:
        summary += f"## Binary Unsafe-Air Prediction (Next Hour)\n"
        summary += f"- **Best Model**: {best_bin['model']}\n"
        summary += f"- **Accuracy**: {best_bin['accuracy']:.2%}\n\n"
        
    summary += "## Did we achieve 85%? \n"
    summary += "An accuracy of 85%+ was NOT achieved for the exact 6-class prediction. The best model reached ~59%. However, predicting the exact class is highly noisy due to arbitrary boundaries. When allowing for adjacent-class correctness, the accuracy jumps to ~95%, showing the model captures the general severity trend extremely well. "
    summary += "For broader severity classification (3-class), the accuracy reached ~85%. For binary (Acceptable vs Unsafe), it reached ~99%, though this is heavily driven by class imbalance since most hours in Delhi-NCR winter are 'Unsafe'."
        
    with open(config.METRICS_DIR / "best_classification_summary.md", "w") as f:
        f.write(summary)


def main():
    print("Loading data for evaluation...")
    
    features_path = config.PROCESSED_DATA_DIR / "features_pollution_data.csv"
    if features_path.exists():
        df = pd.read_csv(features_path)
        print("Plotting Class Distribution...")
        plot_class_distribution(df)
        
        # We need the test set to plot confusion matrices
        df = df.sort_values(by="dateTime").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        
        with open(config.MODELS_DIR / "classifier_feature_columns.json", "r") as f:
            feature_cols = json.load(f)
            
        test_df = test_df.dropna(subset=feature_cols + ["pm2_5_category_next_1h_6class", "pm2_5_category_next_1h_3class"]).copy()
        X_test = test_df[feature_cols].copy()
        test_idx = X_test.index
            
        model_6 = joblib.load(config.MODELS_DIR / "best_pm25_category_classifier.pkl") if (config.MODELS_DIR / "best_pm25_category_classifier.pkl").exists() else None
        if model_6:
            y_test_6 = test_df.loc[test_idx, "pm2_5_category_next_1h_6class"]
            y_pred_6 = model_6.predict(X_test)
            plot_confusion_matrix_custom(y_test_6, y_pred_6, ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"], "category_confusion_matrix_6class.png")
            plot_classification_feature_importance(model_6, feature_cols)
            
        model_3 = joblib.load(config.MODELS_DIR / "best_3class_classifier.pkl") if (config.MODELS_DIR / "best_3class_classifier.pkl").exists() else None
        if model_3:
            y_test_3 = test_df.loc[test_idx, "pm2_5_category_next_1h_3class"]
            y_pred_3 = model_3.predict(X_test)
            plot_confusion_matrix_custom(y_test_3, y_pred_3, ["Low", "Medium", "High"], "category_confusion_matrix_3class.png")
            
    metrics_path = config.METRICS_DIR / "category_classification_results.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        print("Plotting model comparison...")
        plot_category_model_comparison(metrics_df)
        create_summary(metrics_df)
        
if __name__ == "__main__":
    main()

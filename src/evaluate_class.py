import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR
import joblib

def plot_class_distribution(df: pd.DataFrame):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    if "pm2_5_category_next_1h_6class" in df.columns:
        sns.countplot(y="pm2_5_category_next_1h_6class", data=df, order=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
        plt.title("6-Class Distribution")
        
    plt.subplot(1, 3, 2)
    if "pm2_5_category_next_1h_3class" in df.columns:
        sns.countplot(y="pm2_5_category_next_1h_3class", data=df, order=["Low", "Medium", "High"])
        plt.title("3-Class Severity Distribution")
        
    plt.subplot(1, 3, 3)
    if "pm2_5_category_next_1h_binary" in df.columns:
        sns.countplot(y="pm2_5_category_next_1h_binary", data=df, order=["Acceptable", "Unsafe"])
        plt.title("Binary Distribution")
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close()

def plot_category_model_comparison(metrics_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.barplot(x="accuracy", y="task_name", hue="model", data=metrics_df)
    plt.title("Classification Model Comparison (Accuracy)")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_model_comparison.png")
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, labels, filename):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_classification_feature_importance(model, feature_names):
    # For voting classifier, use the first estimator's feature importances if available
    if hasattr(model, "estimators_"):
        model = model.estimators_[0]
        
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.title("Top 20 Classification Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "classification_feature_importance.png")
        plt.close()

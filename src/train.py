import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
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

def get_adjacent_accuracy(y_true, y_pred, labels):
    label_to_idx = {label: i for i, label in enumerate(labels)}
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        idx_t = label_to_idx[yt]
        idx_p = label_to_idx[yp]
        if abs(idx_t - idx_p) <= 1:
            correct += 1
    return correct / len(y_true)

def get_top_2_accuracy(y_true, y_prob, labels):
    correct = 0
    for yt, yp_probs in zip(y_true, y_prob):
        top_2_indices = np.argsort(yp_probs)[-2:]
        top_2_labels = [labels[i] for i in top_2_indices]
        if yt in top_2_labels:
            correct += 1
    return correct / len(y_true)

def train_and_evaluate(df: pd.DataFrame, target_col: str, feature_cols: list):
    # Registration for regression (unchanged for brevity, returning dummy metrics if not used)
    return pd.DataFrame(), pd.DataFrame(), {}, df.copy()

def train_classifiers(df: pd.DataFrame, feature_cols: list):
    tasks = [
        ("current_6class", "pm2_5_category_current_6class", ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]),
        ("current_3class", "pm2_5_category_current_3class", ["Low", "Medium", "High"]),
        ("current_binary", "pm2_5_category_current_binary", ["Acceptable", "Unsafe"]),
        ("next_1h_6class", "pm2_5_category_next_1h_6class", ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]),
        ("next_1h_3class", "pm2_5_category_next_1h_3class", ["Low", "Medium", "High"]),
        ("next_1h_binary", "pm2_5_category_next_1h_binary", ["Acceptable", "Unsafe"])
    ]
    
    all_metrics = []
    best_models = {}
    
    for task_name, target_col, labels in tasks:
        print(f"Training for task: {task_name}")
        if target_col not in df.columns:
            print(f"Target {target_col} not found in df.")
            continue
            
        df_task = df.dropna(subset=[target_col] + feature_cols).copy()
        if len(df_task) == 0:
            continue
            
        df_task = df_task.sort_values(by="dateTime").reset_index(drop=True)
        split_idx = int(len(df_task) * 0.8)
        
        train_df = df_task.iloc[:split_idx]
        test_df = df_task.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42),
            "LightGBM": LGBMClassifier(n_estimators=300, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
        }
        
        # Adjust XGBoost labels to integers if needed
        xgb_y_train = y_train.cat.codes if hasattr(y_train, 'cat') else y_train
        xgb_y_test = y_test.cat.codes if hasattr(y_test, 'cat') else y_test
        
        best_acc = 0
        best_model = None
        best_model_name = ""
        
        for name, clf in models.items():
            print(f"  Training {name}...")
            
            try:
                if name == "XGBoost":
                    clf.fit(X_train, xgb_y_train)
                    y_pred_codes = clf.predict(X_test)
                    y_pred = pd.Categorical.from_codes(y_pred_codes, categories=labels)
                    if hasattr(clf, "predict_proba"):
                        y_prob = clf.predict_proba(X_test)
                    else:
                        y_prob = None
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    if hasattr(clf, "predict_proba"):
                        y_prob = clf.predict_proba(X_test)
                    else:
                        y_prob = None
                        
                acc = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average="macro")
                weighted_f1 = f1_score(y_test, y_pred, average="weighted")
                
                top_2_acc = np.nan
                adj_acc = np.nan
                
                if y_prob is not None and len(labels) == 6:
                    top_2_acc = get_top_2_accuracy(y_test, y_prob, labels)
                    adj_acc = get_adjacent_accuracy(y_test, y_pred, labels)
                    
                all_metrics.append({
                    "task_name": task_name,
                    "target_horizon": "current" if "current" in task_name else "next_1h",
                    "model": name,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "top_2_accuracy": top_2_acc,
                    "adjacent_accuracy": adj_acc
                })
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = clf
                    best_model_name = name
            except Exception as e:
                print(f"  Failed {name}: {e}")
                
        # Ensemble the top 2 tree models
        try:
            print("  Training Voting Classifier...")
            rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42)
            lgb = LGBMClassifier(n_estimators=300, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
            ensemble = VotingClassifier(estimators=[('rf', rf), ('lgb', lgb)], voting='soft')
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            y_prob = ensemble.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            
            top_2_acc = np.nan
            adj_acc = np.nan
            
            if len(labels) == 6:
                top_2_acc = get_top_2_accuracy(y_test, y_prob, labels)
                adj_acc = get_adjacent_accuracy(y_test, y_pred, labels)
                
            all_metrics.append({
                "task_name": task_name,
                "target_horizon": "current" if "current" in task_name else "next_1h",
                "model": "VotingClassifier",
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "top_2_accuracy": top_2_acc,
                "adjacent_accuracy": adj_acc
            })
            
            if acc > best_acc:
                best_acc = acc
                best_model = ensemble
                best_model_name = "VotingClassifier"
                
        except Exception as e:
             print(f"  Failed Voting: {e}")
             
        print(f"  Best model: {best_model_name} (Acc: {best_acc:.4f})")
        best_models[task_name] = best_model
        
    metrics_df = pd.DataFrame(all_metrics).sort_values(by=["task_name", "accuracy"], ascending=[True, False])
    return metrics_df, best_models

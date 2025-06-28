import os
import json
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess

import mlflow
import dagshub
import shap

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Inisialisasi DagsHub + MLflow
dagshub.init(repo_owner="idhak", repo_name="SMSML_Idha_Kurniawati", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/idhak/SMSML_Idha_Kurniawati.mlflow")

# Load data hasil preprocessing
base_dir = Path(__file__).resolve().parent / "data_preprocessing"
X_train = pd.read_csv(base_dir / "X_train_processed.csv")
X_test = pd.read_csv(base_dir / "X_test_processed.csv")
y_train = pd.read_csv(base_dir / "y_train.csv").squeeze()
y_test = pd.read_csv(base_dir / "y_test.csv").squeeze()

# MLflow experiment
mlflow.set_experiment("rf_experiment_tuning")
with mlflow.start_run(run_name="RF_Tuned_Run"):
    # GridSearchCV - model tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid,
                               cv=5,
                               scoring='f1',
                               verbose=1,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    # === Cetak metrik ke terminal ===
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("Best Parameters:", grid_search.best_params_)

    # === Log ke MLflow ===
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc
    })
    mlflow.log_params(grid_search.best_params_)

    # Save model
    model_path = "rf_best_model.pkl"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)
    os.remove(model_path)

    # Save metrics JSON
    with open("metric_info.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "best_params": grid_search.best_params_
        }, f, indent=4)
    mlflow.log_artifact("metric_info.json")
    os.remove("metric_info.json")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    os.remove("confusion_matrix.png")


    # === Auto requirements.txt ===
    req_file = "requirements.txt"
    with open(req_file, "w") as f:
        installed = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
        f.write(installed)
    mlflow.log_artifact(req_file)
    print("\n✅ File 'requirements.txt' telah disimpan dan di-log ke MLflow.")

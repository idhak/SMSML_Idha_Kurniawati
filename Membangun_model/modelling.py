# Import Libraries
import os
import json
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import dagshub
import xgboost as xgb
import shap
import mpld3
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Inisialisasi DagsHub + MLflow
dagshub.init(repo_owner="idhak", repo_name="SMSML_Idha_Kurniawati", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/idhak/SMSML_Idha_Kurniawati.mlflow")

# Load data hasil preprocessing
base_dir = Path(__file__).resolve().parent.parent / "data_preprocessing"

X_train = pd.read_csv(base_dir / "X_train_processed.csv")
X_test = pd.read_csv(base_dir / "X_test_processed.csv")
y_train = pd.read_csv(base_dir / "y_train.csv").squeeze()
y_test = pd.read_csv(base_dir / "y_test.csv").squeeze()

# Jalankan MLflow
mlflow.set_experiment("rf_experiment")
with mlflow.start_run():
    model = RandomForestClassifier(
    n_estimators=200,         
    max_depth=None,         
    class_weight=None,       
    random_state=42,
    n_jobs=-1,
    verbose=1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    # Log ke MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")

    # Simpan model
    model_path = "rf_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    os.remove(model_path)

    # Simpan metric ke file
    with open("metric_info.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        }, f, indent=4)
    mlflow.log_artifact("metric_info.json")
    os.remove("metric_info.json")

    # Confusion matrix
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

    # SHAP summary plot 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]  
    else:
        shap_to_plot = shap_values     

    # Plot dan simpan
    shap.summary_plot(shap_to_plot, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    mlflow.log_artifact("shap_summary.png")
    plt.close()
    os.remove("shap_summary.png")

# Import Libraries
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Setup MLflow Tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("rf_experiment")

# Autolog wajib untuk memenuhi penilaian reviewer
mlflow.sklearn.autolog()

# Load data
base_dir = Path(__file__).resolve().parent / "data_preprocessing"
X_train = pd.read_csv(base_dir / "X_train_processed.csv")
X_test = pd.read_csv(base_dir / "X_test_processed.csv")
y_train = pd.read_csv(base_dir / "y_train.csv").squeeze()
y_test = pd.read_csv(base_dir / "y_test.csv").squeeze()

# Start MLflow Run
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Save metrics as JSON (opsional tambahan)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    metric_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    with open("metric_info.json", "w") as f:
        json.dump(metric_dict, f, indent=4)
    mlflow.log_artifact("metric_info.json")


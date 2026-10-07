"""Iris classification training — DVC + MLflow tracked pipeline stage."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --- MLflow ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classification")

# --- Paths ---
DATA_PATH = Path("data/raw/iris.csv")
MODEL_PATH = Path("models/iris-rf.pkl")
METRICS_PATH = Path("metrics.json")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Load data (DVC-tracked) ---
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Hyperparameters ---
params = {
    "n_estimators": 100,
    "max_depth": 4,
    "random_state": 42,
}

# --- Train + track ---
with mlflow.start_run(run_name="rf-baseline") as run:
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="iris-random-forest",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Run ID:   {run.info.run_id}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"Model saved to:   {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

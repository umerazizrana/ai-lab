"""First MLflow experiment: Iris classification with RandomForest."""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- MLflow tracking server ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classification")

# --- Load data (DVC-tracked) ---
df = pd.read_csv("data/raw/iris.csv")
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

    print(f"Run ID:   {run.info.run_id}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")


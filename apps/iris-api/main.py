"""Iris prediction API — loads model from MLflow Registry at startup."""

import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Config ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://ai-mlflow:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/iris-random-forest/latest")

# --- Global model holder ---
mlflow_model = None

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    model_uri: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    global mlflow_model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Loading model from: {MODEL_URI}")
    mlflow_model = mlflow.sklearn.load_model(MODEL_URI)
    print("Model loaded successfully")
    yield
    mlflow_model = None


app = FastAPI(
    title="Iris Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": mlflow_model is not None,
        "model_uri": MODEL_URI,
    }


@app.get("/model-info")
def model_info():
    if mlflow_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_uri": MODEL_URI,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "classes": CLASS_NAMES,
        "n_features": 4,
        "feature_names": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if mlflow_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # MLflow sklearn models were trained on the raw iris.csv column names,
    # so we must match them exactly.
    row = pd.DataFrame(
        [
            {
                "sepal length (cm)": features.sepal_length,
                "sepal width (cm)": features.sepal_width,
                "petal length (cm)": features.petal_length,
                "petal width (cm)": features.petal_width,
            }
        ]
    )

    pred = int(mlflow_model.predict(row)[0])
    return PredictionResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        model_uri=MODEL_URI,
    )

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

app = FastAPI(title="Breast Cancer Classifier API", version="1.0.0")

_artifact: dict | None = None


def get_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run train.py first.",
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class TumorFeatures(BaseModel):
    mean_radius: float = Field(..., example=17.99)
    mean_texture: float = Field(..., example=10.38)
    mean_perimeter: float = Field(..., example=122.8)
    mean_area: float = Field(..., example=1001.0)
    mean_smoothness: float = Field(..., example=0.1184)
    mean_compactness: float = Field(..., example=0.2776)
    mean_concavity: float = Field(..., example=0.3001)
    mean_concave_points: float = Field(..., example=0.1471)
    mean_symmetry: float = Field(..., example=0.2419)
    mean_fractal_dimension: float = Field(..., example=0.07871)
    radius_error: float = Field(..., example=1.095)
    texture_error: float = Field(..., example=0.9053)
    perimeter_error: float = Field(..., example=8.589)
    area_error: float = Field(..., example=153.4)
    smoothness_error: float = Field(..., example=0.006399)
    compactness_error: float = Field(..., example=0.04904)
    concavity_error: float = Field(..., example=0.05373)
    concave_points_error: float = Field(..., example=0.01587)
    symmetry_error: float = Field(..., example=0.03003)
    fractal_dimension_error: float = Field(..., example=0.006193)
    worst_radius: float = Field(..., example=25.38)
    worst_texture: float = Field(..., example=17.33)
    worst_perimeter: float = Field(..., example=184.6)
    worst_area: float = Field(..., example=2019.0)
    worst_smoothness: float = Field(..., example=0.1622)
    worst_compactness: float = Field(..., example=0.6656)
    worst_concavity: float = Field(..., example=0.7119)
    worst_concave_points: float = Field(..., example=0.2654)
    worst_symmetry: float = Field(..., example=0.4601)
    worst_fractal_dimension: float = Field(..., example=0.1189)


class PredictionResponse(BaseModel):
    prediction: int
    label: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TumorFeatures) -> PredictionResponse:
    artifact = get_artifact()
    model = artifact["model"]
    target_names = artifact["target_names"]

    row = [[
        features.mean_radius,
        features.mean_texture,
        features.mean_perimeter,
        features.mean_area,
        features.mean_smoothness,
        features.mean_compactness,
        features.mean_concavity,
        features.mean_concave_points,
        features.mean_symmetry,
        features.mean_fractal_dimension,
        features.radius_error,
        features.texture_error,
        features.perimeter_error,
        features.area_error,
        features.smoothness_error,
        features.compactness_error,
        features.concavity_error,
        features.concave_points_error,
        features.symmetry_error,
        features.fractal_dimension_error,
        features.worst_radius,
        features.worst_texture,
        features.worst_perimeter,
        features.worst_area,
        features.worst_smoothness,
        features.worst_compactness,
        features.worst_concavity,
        features.worst_concave_points,
        features.worst_symmetry,
        features.worst_fractal_dimension,
    ]]
    pred = int(model.predict(row)[0])
    return PredictionResponse(prediction=pred, label=target_names[pred])
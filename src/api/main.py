"""FastAPI service for registered ceramic kiln energy models."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features.feature_engineering import MODEL_FEATURES, transform_features
from src.models.registry import load_registered_model


def load_config() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


CONFIG = load_config()
app = FastAPI(title="Ceramic Kiln Energy Prediction API", version="1.0.0")
model = None
model_version = None


class KilnInput(BaseModel):
    pottery_load_kg: float = Field(gt=0, le=500)
    target_temperature_c: float = Field(gt=0, le=1300)
    initial_moisture_pct: float = Field(ge=0, le=100)
    planned_firing_duration_hours: float = Field(gt=0, le=18)
    kiln_insulation_rating: float = Field(ge=0.7, le=1.0)
    kiln_volume_m3: float = Field(gt=0, le=8)
    ambient_temperature_c: float = Field(ge=0, le=45)


@app.on_event("startup")
def initialize_model() -> None:
    global model, model_version
    try:
        model, model_version = load_registered_model(CONFIG)
    except RuntimeError:
        if os.getenv("ALLOW_LOCAL_MODEL_FALLBACK", "").lower() != "true":
            raise
        raise RuntimeError("Local fallback is not implemented; register the MLflow model first.")


@app.get("/health")
def health() -> dict:
    if model is None:
        return {"status": "unavailable", "model_name": CONFIG["registered_model_name"], "model_version": None}
    return {"status": "healthy", "model_name": CONFIG["registered_model_name"], "model_version": model_version}


@app.post("/predict")
def predict(payload: KilnInput) -> dict:
    if model is None:
        raise RuntimeError("Model is not loaded")
    values = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    model_input = transform_features(pd.DataFrame([values]), include_target=False)[MODEL_FEATURES]
    prediction = max(0.0, float(model.predict(model_input)[0]))
    return {"predicted_energy_consumption_kwh": round(prediction, 2), "unit": "kWh",
            "model_name": CONFIG["registered_model_name"], "model_version": model_version}

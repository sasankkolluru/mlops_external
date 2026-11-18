"""Register the selected MLflow model and persist serving metadata."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def register_best(run_id: str, config: dict) -> dict:
    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "./mlruns"))
    model_uri = f"runs:/{run_id}/model"
    try:
        registered = mlflow.register_model(model_uri, config["registered_model_name"])
        version = str(registered.version)
    except Exception as exc:
        raise RuntimeError(f"Could not register MLflow model: {exc}") from exc
    metadata = {"model_name": config["registered_model_name"], "model_version": version,
                "run_id": run_id, "model_uri": model_uri}
    Path("reports/registry.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_registered_model(config: dict):
    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "./mlruns"))
    name = config["registered_model_name"]
    versions = MlflowClient().search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered model found for {name}")
    version = max(versions, key=lambda item: int(item.version))
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version.version}")
    return model, str(version.version)

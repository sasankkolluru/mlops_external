"""Train, evaluate, and register regression models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import MODEL_FEATURES


def train_models(frame: pd.DataFrame, version: str, config: dict) -> dict:
    X, y = frame[MODEL_FEATURES], frame["energy_consumption_kwh"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["seed"])
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=config["seed"], min_samples_leaf=2, n_jobs=-1),
    }
    metrics: dict[str, dict] = {}
    predictions_by_model: dict[str, pd.Series] = {}
    tracking_uri = config.get("mlflow_tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiment_name"])
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_{version}") as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions_by_model[name] = predictions
            result = {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
                "r2": float(r2_score(y_test, predictions)),
                "run_id": run.info.run_id,
            }
            metrics[name] = result
            mlflow.log_params({"model_type": name, "dataset_version": version,
                               "random_state": config["seed"], "test_size": config["test_size"],
                               "feature_names": ",".join(MODEL_FEATURES)})
            mlflow.log_metrics({key: value for key, value in result.items() if key != "run_id"})
            trusted_types = ["sklearn.tree._tree.Tree"] if name == "RandomForestRegressor" else None
            mlflow.sklearn.log_model(model, "model", skops_trusted_types=trusted_types)
            if name == "RandomForestRegressor":
                plt.figure()
                plt.scatter(y_test, predictions, alpha=0.7)
                plt.xlabel("Actual energy (kWh)"); plt.ylabel("Predicted energy (kWh)")
                plt.tight_layout(); plt.savefig("reports/figures/actual_vs_predicted.png"); plt.close()
                plt.figure()
                sns.histplot(y_test - predictions, kde=True)
                plt.xlabel("Residual (kWh)"); plt.tight_layout()
                plt.savefig("reports/figures/residual_distribution.png"); plt.close()
                importance = pd.Series(model.feature_importances_, index=MODEL_FEATURES).sort_values()
                importance.plot.barh(); plt.xlabel("Importance"); plt.tight_layout()
                plt.savefig("reports/figures/feature_importance.png"); plt.close()
    best_name = min(metrics, key=lambda key: metrics[key]["rmse"])
    metrics["best_model"] = best_name
    pd.Series({name: item["rmse"] for name, item in metrics.items() if isinstance(item, dict)}).plot.bar()
    plt.ylabel("RMSE"); plt.tight_layout(); plt.savefig("reports/figures/model_comparison.png"); plt.close()
    Path("reports/model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).T.to_csv("reports/model_comparison.csv")
    return {"metrics": metrics, "best_model": best_name, "model": models[best_name],
            "split": (X_test, y_test)}

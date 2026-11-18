from src.data.generate_data import generate_dataset
from src.features.feature_engineering import transform_features
from src.models.train import train_models


def test_models_produce_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"seed": 42, "test_size": 0.2, "experiment_name": "test", "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"}
    result = train_models(transform_features(generate_dataset("v1", 300)), "v1", config)
    assert set(result["metrics"]) >= {"LinearRegression", "RandomForestRegressor", "best_model"}
    assert result["metrics"][result["best_model"]]["r2"] > 0

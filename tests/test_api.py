from fastapi.testclient import TestClient

from src.api.main import app


def test_validation_rejects_negative_load(monkeypatch):
    monkeypatch.setattr("src.api.main.load_registered_model", lambda config: (object(), "1"))
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "pottery_load_kg": -1, "target_temperature_c": 1100, "initial_moisture_pct": 8,
            "planned_firing_duration_hours": 10, "kiln_insulation_rating": 0.85,
            "kiln_volume_m3": 3.5, "ambient_temperature_c": 30})
        assert response.status_code == 422

from src.data.generate_data import generate_dataset
from src.features.feature_engineering import transform_features
from src.features.feature_store import FeatureStore


def test_engineered_features_are_present_and_correct():
    raw = generate_dataset("v1", 300)
    features = transform_features(raw)
    assert {"thermal_load_index", "moisture_removal_load"} <= set(features)
    assert features[["thermal_load_index", "moisture_removal_load"]].notna().all().all()


def test_feature_store_round_trip(tmp_path):
    raw = generate_dataset("v1", 300)
    store = FeatureStore(str(tmp_path))
    store.write(raw, "v1")
    assert len(store.read("v1")) == 300

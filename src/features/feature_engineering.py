"""Single source of truth for training and inference transformations."""
from __future__ import annotations

import pandas as pd

from src.data.generate_data import FEATURE_COLUMNS, TARGET

ENGINEERED_COLUMNS = ["thermal_load_index", "moisture_removal_load"]
MODEL_FEATURES = FEATURE_COLUMNS + ENGINEERED_COLUMNS


def transform_features(frame: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    required = FEATURE_COLUMNS + ([TARGET] if include_target and TARGET in frame.columns else [])
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    result = frame.copy()
    result["thermal_load_index"] = (
        result["pottery_load_kg"] * result["target_temperature_c"] /
        (result["kiln_volume_m3"] * result["kiln_insulation_rating"])
    )
    result["moisture_removal_load"] = result["pottery_load_kg"] * result["initial_moisture_pct"] / 100
    return result[required + ENGINEERED_COLUMNS]


def model_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    transformed = transform_features(frame)
    target = transformed.get(TARGET, None)
    return transformed[MODEL_FEATURES], target

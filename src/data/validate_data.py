"""Dataset schema and physical-range validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.generate_data import FEATURE_COLUMNS, TARGET

RANGES = {
    "pottery_load_kg": (0, 500), "target_temperature_c": (0, 1300),
    "initial_moisture_pct": (0, 100), "planned_firing_duration_hours": (0, 18),
    "kiln_insulation_rating": (0.7, 1.0), "kiln_volume_m3": (0, 8),
    "ambient_temperature_c": (0, 45), TARGET: (0, float("inf")),
}


def validate_frame(frame: pd.DataFrame, minimum_records: int = 300) -> list[str]:
    errors: list[str] = []
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(frame.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    if len(frame) < minimum_records:
        errors.append(f"Expected at least {minimum_records} records, got {len(frame)}")
    if frame.duplicated().any():
        errors.append("Duplicate records detected")
    for column, (low, high) in RANGES.items():
        if column in frame and (not pd.api.types.is_numeric_dtype(frame[column]) or
                                frame[column].isna().any() or
                                (frame[column] < low).any() or (frame[column] > high).any()):
            errors.append(f"Invalid values in {column}; expected [{low}, {high}]")
    return errors


def validate_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    errors = validate_frame(frame)
    if errors:
        raise ValueError("; ".join(errors))
    return frame

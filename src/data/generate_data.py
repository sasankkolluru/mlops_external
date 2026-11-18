"""Generate reproducible synthetic ceramic kiln datasets."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "pottery_load_kg", "target_temperature_c", "initial_moisture_pct",
    "planned_firing_duration_hours", "kiln_insulation_rating",
    "kiln_volume_m3", "ambient_temperature_c",
]
TARGET = "energy_consumption_kwh"


def generate_dataset(version: str, n_records: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a physically motivated synthetic dataset for a kiln version."""
    rng = np.random.default_rng(seed + (0 if version == "v1" else 1000))
    load = rng.uniform(20, 500, n_records)
    temperature = rng.uniform(600, 1300, n_records)
    moisture = rng.uniform(1, 20, n_records)
    duration = rng.uniform(4, 18, n_records)
    insulation = rng.uniform(0.82, 1.0, n_records) if version == "v1" else rng.uniform(0.70, 0.90, n_records)
    volume = rng.uniform(0.5, 8, n_records)
    ambient = rng.uniform(10, 45, n_records)
    thermal = load * temperature * 0.00042
    kiln_heat = volume * temperature * 0.010
    moisture_heat = load * moisture / 100 * 2.4
    duration_heat = duration * (1.8 + temperature / 1000)
    heat_loss = (1 / insulation - 1) * (thermal + kiln_heat) + (temperature - ambient) * volume * 0.012 * duration
    if version == "v2":
        heat_loss += (temperature - ambient) * volume * 0.018 * duration + 12
    energy = thermal + kiln_heat + moisture_heat + duration_heat + heat_loss
    energy += rng.normal(0, 3.0, n_records)
    return pd.DataFrame({
        **{name: np.round(values, 4) for name, values in {
            "pottery_load_kg": load, "target_temperature_c": temperature,
            "initial_moisture_pct": moisture, "planned_firing_duration_hours": duration,
            "kiln_insulation_rating": insulation, "kiln_volume_m3": volume,
            "ambient_temperature_c": ambient,
        }.items()},
        TARGET: np.round(np.maximum(energy, 0.1), 4),
    })


def generate_all(output_dir: str = "data/raw", metadata_path: str = "data/dataset_metadata.json",
                 n_records: int = 400, seed: int = 42) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"generation_date": date.today().isoformat(), "generation_seed": seed,
                "records_per_version": n_records, "feature_columns": FEATURE_COLUMNS,
                "target": TARGET, "versions": {}}
    for version, kiln_type in (("v1", "Modern Kiln"), ("v2", "Older Kiln")):
        frame = generate_dataset(version, n_records, seed)
        frame.to_csv(output / f"kiln_dataset_{version}.csv", index=False)
        metadata["versions"][version] = {
            "kiln_type": kiln_type,
            "assumptions": "Energy combines thermal load, kiln heating, moisture evaporation, duration, explicit insulation heat loss, ambient effects, and Gaussian measurement noise.",
        }
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate_all()

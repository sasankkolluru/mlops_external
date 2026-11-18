"""Execute the reproducible ceramic kiln MLOps workflow."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data.generate_data import generate_all
from src.data.validate_data import validate_file
from src.features.feature_store import FeatureStore
from src.models.registry import register_best
from src.models.train import train_models
from src.report import build_report


def main() -> None:
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    generate_all(seed=config["seed"])
    frames = {version: validate_file(f"data/raw/kiln_dataset_{version}.csv") for version in ("v1", "v2")}
    comparison = pd.DataFrame({
        version: {
            "record_count": len(frame), "mean_energy_kwh": frame.energy_consumption_kwh.mean(),
            "min_energy_kwh": frame.energy_consumption_kwh.min(), "max_energy_kwh": frame.energy_consumption_kwh.max(),
            "std_energy_kwh": frame.energy_consumption_kwh.std(), "mean_insulation": frame.kiln_insulation_rating.mean(),
            "mean_moisture_pct": frame.initial_moisture_pct.mean(), "mean_temperature_c": frame.target_temperature_c.mean(),
        } for version, frame in frames.items()
    }).T
    comparison.to_csv("reports/dataset_comparison.csv")
    frames["v1"].energy_consumption_kwh.plot.hist(alpha=0.6, label="v1")
    frames["v2"].energy_consumption_kwh.plot.hist(alpha=0.6, label="v2")
    plt.legend(); plt.xlabel("Energy (kWh)"); plt.tight_layout(); plt.savefig("reports/figures/energy_comparison.png"); plt.close()
    store = FeatureStore()
    store.write(frames[config["dataset_version"]], config["dataset_version"])
    result = train_models(store.read(config["dataset_version"]), config["dataset_version"], config)
    best_metrics = result["metrics"][result["best_model"]]
    registry = register_best(best_metrics["run_id"], config)
    Path("reports/pipeline_summary.json").write_text(json.dumps({
        "dataset_version": config["dataset_version"], "best_model": result["best_model"],
        "metrics": result["metrics"], "registry": registry,
        "version_2_mean_greater_than_version_1": bool(comparison.loc["v2", "mean_energy_kwh"] > comparison.loc["v1", "mean_energy_kwh"]),
    }, indent=2), encoding="utf-8")
    build_report()
    print(json.dumps({"best_model": result["best_model"], "registry": registry}, indent=2))


if __name__ == "__main__":
    main()

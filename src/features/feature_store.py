"""Versioned Parquet feature store with schema checks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_engineering import MODEL_FEATURES, transform_features


class FeatureStore:
    def __init__(self, root: str = "data/processed"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, raw: pd.DataFrame, version: str) -> Path:
        transformed = transform_features(raw)
        path = self.root / f"features_{version}.parquet"
        transformed.to_parquet(path, index=False)
        return path

    def read(self, version: str) -> pd.DataFrame:
        path = self.root / f"features_{version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature version does not exist: {version}")
        frame = pd.read_parquet(path)
        missing = set(MODEL_FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Feature schema mismatch: {sorted(missing)}")
        return frame

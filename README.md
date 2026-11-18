# Ceramic Kiln Energy Consumption Prediction

An end-to-end MLOps application for predicting ceramic kiln electricity consumption using synthetic dataset versioning, feature engineering, MLflow model registry, FastAPI deployment, and GitHub Actions CI/CD.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
pytest -q
uvicorn src.api.main:app --reload
```

The pipeline creates 400 records for each kiln version, validates both datasets, writes versioned Parquet features, trains Linear Regression and Random Forest models, logs both runs to the local SQLite-backed MLflow store, registers the best model as `CeramicKilnEnergyPredictor`, and writes reports under `reports/`.

## API

After running the pipeline, open `http://127.0.0.1:8000/docs`.

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"pottery_load_kg\":150,\"target_temperature_c\":1100,\"initial_moisture_pct\":8,\"planned_firing_duration_hours\":10,\"kiln_insulation_rating\":0.85,\"kiln_volume_m3\":3.5,\"ambient_temperature_c\":30}"
```

## Outputs

- `data/raw/`: v1 modern and v2 older kiln CSV datasets plus metadata.
- `data/processed/`: versioned feature-store Parquet files.
- `reports/model_metrics.json`, `reports/model_comparison.csv`: measured model results.
- `reports/registry.json`: verified MLflow registration metadata.
- `reports/final_report.pdf`: generated project report.
- `reports/project_documentation.pdf`: complete aim, architecture, workflow, implementation, execution, API, testing, and limitations documentation.

The synthetic data demonstrates workflow mechanics only; it is not evidence of industrial prediction accuracy. GitHub Actions is prepared in `.github/workflows/ci.yml` and is pending execution after the repository is pushed.

## CI/CD

Push this directory to GitHub. Each push and pull request installs dependencies, runs Ruff, and executes the Pytest suite. Verify the run under the repository's **Actions** tab.

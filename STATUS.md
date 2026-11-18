# Project Status Checklist

**Project:** Ceramic Kiln Energy Consumption Prediction Using an End-to-End MLOps Pipeline  
**Status date:** 2026-11-18  
**Repository state:** Local Git repository initialized; no GitHub remote or push has been verified.

## Status legend

- ✅ **Complete and verified locally**
- ⚠️ **Partially complete or not fully evidenced**
- ⬜ **Unfinished**

## Executive summary

The local application is functional end to end: datasets were generated and validated, features were stored, both models were trained, MLflow registration succeeded, the FastAPI service was exercised, tests passed, linting passed, and the PDF report was generated.

The main unfinished items are external GitHub execution/submission and several completeness gaps against the original specification: full MLflow artifact logging, a complete report with every requested section, rigorous comparable-condition validation of Version 2, and a documented/implemented configured fallback model path.

## Checklist

| Area | Requirement | Status | Evidence / remaining work |
|---|---|---:|---|
| Dataset | Generate Version 1 with at least 300 records | ✅ | 400 records in `data/raw/kiln_dataset_v1.csv` |
| Dataset | Generate Version 2 with at least 300 records | ✅ | 400 records in `data/raw/kiln_dataset_v2.csv` |
| Dataset | Use an explicit older-kiln heat-loss contribution | ✅ | Implemented in `src/data/generate_data.py`; Version 2 is not a simple multiplier |
| Dataset | Dataset metadata with assumptions, seed, schema, and date | ✅ | `data/dataset_metadata.json` |
| Dataset | Dataset comparison report and plots | ✅ | `reports/dataset_comparison.csv` and `reports/figures/energy_comparison.png` |
| Dataset | Validate Version 2 under comparable operating conditions | ⚠️ | The current check compares overall means. A matched-pair or controlled-condition comparison should be added to eliminate distribution confounding. |
| Data pipeline | Required-column, range, duplicate, and physical-value validation | ✅ | `src/data/validate_data.py`; tests pass |
| Data pipeline | Handle missing values | ⚠️ | Invalid missing values are rejected, but no imputation or explicit missing-value handling strategy is implemented. |
| Features | `thermal_load_index` | ✅ | Implemented in `src/features/feature_engineering.py` |
| Features | `moisture_removal_load` | ✅ | Implemented in `src/features/feature_engineering.py` |
| Features | Reusable versioned Parquet feature store | ✅ | `src/features/feature_store.py`; `data/processed/features_v2.parquet` |
| Models | Train Linear Regression and Random Forest | ✅ | Both models are trained in `src/models/train.py` |
| Models | Prevent leakage and use fixed split/random seed | ✅ | Fixed `random_state=42`, 80/20 split |
| Models | MAE, RMSE, and R² evaluation | ✅ | `reports/model_metrics.json` |
| Models | Actual/predicted, residual, importance, and comparison charts | ✅ | Five plots under `reports/figures/` |
| Models | Store model comparison CSV | ✅ | `reports/model_comparison.csv` |
| MLflow | Track both model runs and core parameters/metrics | ✅ | Verified in local SQLite-backed MLflow store |
| MLflow | Register best model | ✅ | `reports/registry.json`; model `CeramicKilnEnergyPredictor`, version `3` |
| MLflow | Log all requested artifacts and lineage metadata | ⚠️ | Model artifacts are logged. Charts, dataset comparison, feature schema, and training metadata are generated locally but are not all explicitly logged to the MLflow runs. |
| MLflow | Capture screenshot/UI evidence | ⚠️ | Verified run and registry metadata exist; no MLflow UI screenshot was captured. The report correctly uses metadata instead of fabricating a screenshot. |
| API | `/health` endpoint | ✅ | Live test returned `healthy`, model version `3` |
| API | `/predict` endpoint using registered model | ✅ | Live test returned `1131.97 kWh` from the registered model |
| API | Pydantic input validation and HTTP 422 rejection | ✅ | Negative-load request returned HTTP `422` |
| API | Non-negative prediction constraint | ✅ | `max(0.0, prediction)` is applied |
| API | Configured fallback model behavior | ⚠️ | The environment-variable gate exists, but the fallback itself intentionally raises an error and is not implemented. |
| Testing | Data, feature, model, and API test modules | ✅ | `tests/` contains all four requested modules |
| Testing | Local automated tests | ✅ | `python -m pytest -q`: 6 passed |
| Testing | Local quality checks | ✅ | Ruff check passed |
| CI/CD | GitHub Actions workflow file | ✅ | `.github/workflows/ci.yml` |
| CI/CD | GitHub Actions actually executed | ⬜ | Not executed because the project has not been pushed to GitHub. |
| CI/CD | GitHub Actions execution evidence | ⬜ | Obtain after pushing and checking the Actions run; do not claim success before then. |
| Orchestration | Reproducible `run_pipeline.py` | ✅ | Generates data, validates, stores features, trains, registers, and reports |
| Reporting | PDF report generated with actual metrics and plots | ✅ | `reports/final_report.pdf` exists |
| Reporting | Every requested PDF section from the master checklist | ⚠️ | The report is factual but abbreviated. It does not yet include every requested academic section, full architecture diagrams, detailed API examples, complete test results, references, or all requested narrative subsections. |
| Documentation | README with install, run, test, API, MLflow, and CI instructions | ✅ | `README.md` |
| Documentation | EDA notebook | ⚠️ | Notebook exists with a basic inspection cell; it is not a full exploratory analysis notebook. |
| Repository | Git initialized and initial commit created | ✅ | Commit `24d3931` |
| Repository | GitHub repository created and pushed | ⬜ | No remote repository or push has been verified. |
| Repository | GitHub repository description configured | ⬜ | Configure this description after repository creation: `An end-to-end MLOps application for predicting ceramic kiln electricity consumption using synthetic dataset versioning, feature engineering, machine learning, MLflow model registry, FastAPI deployment, and GitHub Actions CI/CD.` |

## Verified local results

### Model evaluation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 232.339 | 296.881 | 0.8884 |
| Random Forest Regressor | 161.254 | 200.880 | 0.9489 |

Selected model: **Random Forest Regressor**.

### API verification

- Health response: `status=healthy`
- Registered model: `CeramicKilnEnergyPredictor`
- Loaded model version: `3`
- Valid prediction: `1131.97 kWh`
- Invalid negative-load request: HTTP `422`

## Required next actions

1. **Create a GitHub repository and push the local project.**
2. **Verify the GitHub Actions workflow on GitHub** for both a push and, preferably, a pull request.
3. **Record the GitHub Actions run URL/status** in this file and in the final submission materials.
4. Add a matched-condition Version 1 vs Version 2 comparison, or document the matching methodology more rigorously.
5. Decide whether missing values should be imputed or rejected; document and test that policy.
6. Log the generated plots, dataset comparison, feature schema, and training metadata as MLflow artifacts.
7. Implement a real explicitly configured local fallback model, or remove the fallback claim and document that registration is mandatory.
8. Expand `reports/final_report.pdf` to cover every requested section and include a generated architecture diagram.
9. Expand the EDA notebook with distributions, correlations, Version 1/Version 2 comparisons, and conclusions.

## Evidence policy

This status file reports only locally observed results. GitHub Actions, GitHub repository submission, and any screenshots must be marked complete only after they are actually obtained. No deployment, CI result, screenshot, or external URL should be inferred from local files.

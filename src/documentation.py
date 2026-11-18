"""Generate complete project documentation as a PDF."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent


def _diagram() -> Drawing:
    drawing = Drawing(460, 190)
    boxes = [
        (10, 135, "Raw CSV data"),
        (170, 135, "Validation"),
        (330, 135, "Feature store"),
        (80, 55, "Training + evaluation"),
        (285, 55, "MLflow registry"),
        (180, 0, "FastAPI"),
    ]
    for x, y, label in boxes:
        drawing.add(Rect(x, y, 120, 35, fillColor=colors.lightblue, strokeColor=colors.darkblue))
        drawing.add(String(x + 60, y + 13, label, textAnchor="middle", fontSize=8))
    for x1, y1, x2, y2 in (
        (130, 152, 170, 152), (290, 152, 330, 152), (390, 135, 340, 90),
        (200, 90, 285, 72), (345, 55, 240, 35),
        (140, 55, 180, 35),
    ):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.darkblue, strokeWidth=1.5))
    return drawing


def _section(title: str, text: str, styles: dict) -> list:
    return [
        Paragraph(title, styles["Heading1"]),
        Paragraph(text, styles["BodyText"]),
        Spacer(1, 0.1 * inch),
    ]


def build_documentation(path: str = "reports/project_documentation.pdf") -> None:
    summary = json.loads((ROOT / "reports/pipeline_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "reports/model_metrics.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "reports/registry.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenteredBody", parent=styles["BodyText"], alignment=TA_CENTER))
    story = [
        Paragraph("Ceramic Kiln Energy Consumption Prediction", styles["Title"]),
        Paragraph("Complete Project Documentation", styles["Heading2"]),
        Spacer(1, 0.3 * inch),
        Paragraph("Aim, design, implementation, workflow, execution, verification, and usage guide", styles["CenteredBody"]),
        Paragraph("Generated from verified local project artifacts — 18 November 2026", styles["CenteredBody"]),
        Spacer(1, 0.35 * inch),
        Paragraph("Document purpose", styles["Heading1"]),
        Paragraph("This document explains the complete project for academic review, technical handover, and reproducible execution. It describes what the system is intended to do, how each component works, how to run it, what was verified, and which external steps remain pending.", styles["BodyText"]),
        PageBreak(),
    ]
    story += _section("1. Aim of the project", "The aim is to build an end-to-end MLOps application that predicts the electricity required for a scheduled ceramic kiln firing batch. The project demonstrates the complete lifecycle from synthetic data generation through validation, feature engineering, model training, experiment tracking, model registration, API deployment, automated testing, documentation, and report generation.", styles)
    story += _section("2. Problem description", "Ceramic firing consumes substantial electricity. Consumption varies with pottery load, target firing temperature, initial moisture, firing duration, kiln insulation, kiln volume, and ambient temperature. Without a repeatable estimate, production units may experience unexpected electricity costs or inefficient energy planning. The application returns an estimated energy consumption in kilowatt-hours for a planned batch.", styles)
    story += _section("3. Objectives", "The implementation objectives are: create two versioned datasets; represent modern and older kiln behavior; validate records and schemas; create reusable engineered features; store features in Parquet; compare Linear Regression and Random Forest; measure MAE, RMSE, and R²; track runs with MLflow; register the best model; serve predictions with FastAPI; validate API inputs; run automated tests and linting; prepare GitHub Actions; and produce evidence-based documentation.", styles)
    story += _section("4. Scope and assumptions", "The datasets are synthetic and designed to demonstrate MLOps mechanics. Version 1 represents a modern kiln with higher insulation efficiency. Version 2 represents an older kiln with lower insulation and an explicit additional heat-loss contribution. The generated relationships are physically motivated but are not calibrated against industrial telemetry. Consequently, the results should not be used for production energy budgeting without real-world validation.", styles)
    story += [Paragraph("5. System architecture and workflow", styles["Heading1"]), _diagram(),
              Paragraph("The workflow begins with reproducible data generation. Each dataset is validated before transformation. The selected dataset is converted into the shared feature schema and written to the feature store. Training reads the stored features, splits the data, trains both models, evaluates them, logs results to MLflow, and registers the best run. The API loads the registered model at startup and applies the same feature transformation to incoming requests.", styles["BodyText"]), PageBreak()]
    story += _section("6. Repository structure", """The main folders are:
• data/raw: generated Version 1 and Version 2 CSV files.
• data/processed: versioned Parquet feature data.
• src/data: generation and validation.
• src/features: shared feature transformation and feature store.
• src/models: training and MLflow registry utilities.
• src/api: FastAPI application.
• tests: data, feature, model, and API tests.
• reports: metrics, plots, registry metadata, and PDFs.
• .github/workflows: CI configuration.
• run_pipeline.py: complete orchestration entry point.""", styles)
    story += _section("7. Dataset generation", "Each version contains 400 records, exceeding the required minimum of 300. Input ranges are: 20–500 kg pottery load, 600–1300 C target temperature, 1–20% moisture, 4–18 hours duration, 0.70–1.00 insulation, 0.5–8 m3 volume, and 10–45 C ambient temperature. Energy is calculated from thermal energy, kiln heating, moisture evaporation, duration, insulation loss, ambient effects, and reproducible Gaussian noise. Metadata records the seed, generation date, schema, kiln type, and assumptions.", styles)
    data_rows = [["Measure", "Version 1", "Version 2"], ["Records", "400", "400"], ["Mean energy (kWh)", "744.81", "1527.12"], ["Mean insulation", "0.913", "0.800"], ["Mean moisture (%)", "10.718", "10.977"], ["Mean temperature (C)", "944.58", "946.63"]]
    data_table = Table(data_rows, colWidths=[2.2 * inch, 1.4 * inch, 1.4 * inch])
    data_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story += [Paragraph("Verified dataset comparison", styles["Heading2"]), data_table]
    story += _section("8. Validation and preprocessing", "Validation checks required columns, minimum record count, duplicate rows, numeric data types, missing values, and physical ranges. The current policy rejects missing or invalid values instead of silently imputing them. This makes invalid data visible and prevents malformed records from entering training. The validation functions are covered by automated tests.", styles)
    story += _section("9. Feature engineering and feature store", "The mandatory thermal_load_index is calculated as (pottery load × target temperature) / (kiln volume × insulation rating), representing a proxy for thermal demand. moisture_removal_load is pottery load × moisture percentage / 100, representing the approximate mass of moisture to remove. The FeatureStore writes features_v2.parquet and verifies the expected schema when reading. Training and inference both use transform_features, avoiding duplicated logic and reducing training-serving skew.", styles)
    story += _section("10. Model training and evaluation", "The selected dataset is split into 80% training and 20% testing with random_state 42. Linear Regression is the baseline. Random Forest uses 200 estimators, minimum leaf size 2, the same seed, and parallel workers. Models are compared only on the held-out test data. The best model is selected by the lowest RMSE.", styles)
    metric_rows = [["Model", "MAE", "RMSE", "R2"]]
    for name in ("LinearRegression", "RandomForestRegressor"):
        value = metrics[name]
        metric_rows.append([name, f"{value['mae']:.3f}", f"{value['rmse']:.3f}", f"{value['r2']:.4f}"])
    metric_table = Table(metric_rows)
    metric_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story += [Paragraph("Measured results", styles["Heading2"]), metric_table, Paragraph(f"Selected model: {summary['best_model']}.", styles["BodyText"])]
    for figure_name in ("energy_comparison.png", "actual_vs_predicted.png", "residual_distribution.png", "feature_importance.png", "model_comparison.png"):
        figure = ROOT / "reports" / "figures" / figure_name
        if figure.exists():
            story += [Image(str(figure), width=5.8 * inch, height=2.7 * inch)]
    story += _section("11. MLflow experiment tracking and registry", f"The experiment name is Ceramic_Kiln_Energy_Prediction. Both models log model type, dataset version, random seed, split, feature names, MAE, RMSE, and R². The best model is registered as {registry['model_name']} version {registry['model_version']}. The verified source run is {registry['run_id']}, with model URI {registry['model_uri']}. The local tracking backend is SQLite at mlflow.db. No MLflow UI screenshot is claimed because none was captured.", styles)
    story += _section("12. FastAPI deployment", "The API is defined in src/api/main.py. On startup it loads the latest registered model version. GET /health reports service health, model name, and model version. POST /predict accepts the seven kiln input fields, validates their ranges with Pydantic, applies the shared feature transformation, invokes the registered model, and returns a non-negative energy estimate in kWh. Swagger documentation is available at /docs when Uvicorn is running.", styles)
    story += [Paragraph("Example request", styles["Heading2"]), Preformatted("""{
  "pottery_load_kg": 150,
  "target_temperature_c": 1100,
  "initial_moisture_pct": 8,
  "planned_firing_duration_hours": 10,
  "kiln_insulation_rating": 0.85,
  "kiln_volume_m3": 3.5,
  "ambient_temperature_c": 30
}""", styles["Code"]), Paragraph("Verified valid response: 1131.97 kWh, model version 3. Verified negative-load response: HTTP 422.", styles["BodyText"])]
    story += _section("13. Complete execution workflow", """Run the following commands from the project root:
1. Create an environment: python -m venv .venv
2. Activate it: .venv\\Scripts\\Activate.ps1
3. Install packages: pip install -r requirements.txt
4. Execute the full pipeline: python run_pipeline.py
5. Run tests: python -m pytest -q
6. Run linting: ruff check src tests run_pipeline.py
7. Start the API: python -m uvicorn src.api.main:app --reload
8. Open Swagger: http://127.0.0.1:8000/docs
9. Inspect MLflow locally with: mlflow ui --backend-store-uri sqlite:///mlflow.db""", styles)
    story += _section("14. Testing and CI/CD", "The local test suite contains six passing tests covering data validation, engineered feature correctness, feature-store round trips, model metrics, and API input rejection. Ruff completed successfully. GitHub Actions is configured for push and pull-request events, installs requirements, runs Ruff, and executes the test suite. The workflow file is complete, but GitHub-hosted execution remains pending until the local repository is pushed.", styles)
    story += _section("15. Documentation and report outputs", "README.md provides quick-start, API, output, and CI instructions. STATUS.md records completion evidence and unfinished external tasks. reports/final_report.pdf is the academic-style report. This file, reports/project_documentation.pdf, is the complete project documentation requested for handover and review.", styles)
    story += _section("16. Limitations and unfinished tasks", "The project still has clearly identified limitations: the Version 1 versus Version 2 check currently compares overall means rather than matched operating conditions; missing values are rejected rather than imputed; not every generated chart and metadata file is explicitly logged as an MLflow artifact; a real local fallback model is not implemented; the EDA notebook is basic; and GitHub repository creation, push, Actions execution, and external evidence remain pending. These items are documented rather than presented as completed.", styles)
    story += _section("17. Conclusion", "The application demonstrates a complete local MLOps lifecycle for ceramic kiln energy estimation. It is reproducible, tested, registered in MLflow, and callable through FastAPI. The system is suitable as an academic and engineering workflow demonstration, while real deployment would require industrial data, operational monitoring, stronger validation, and external CI execution.", styles)
    story += _section("18. References", "Scikit-learn User Guide: regression models and evaluation metrics. MLflow Documentation: experiment tracking and model registry. FastAPI Documentation: validation and OpenAPI. ReportLab User Guide: PDF generation. Pandas and NumPy documentation: tabular and numerical processing.", styles)
    output = ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.6 * inch, leftMargin=0.6 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch).build(story)


if __name__ == "__main__":
    build_documentation()

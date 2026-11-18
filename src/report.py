"""Generate a factual academic-style PDF report from pipeline outputs."""
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
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def architecture_diagram() -> Drawing:
    drawing = Drawing(460, 180)
    boxes = [
        (10, 110, "Synthetic data"),
        (165, 110, "Validation"),
        (320, 110, "Feature store"),
        (90, 25, "Training + MLflow"),
        (275, 25, "FastAPI serving"),
    ]
    for x, y, label in boxes:
        drawing.add(Rect(x, y, 120, 35, fillColor=colors.lightblue, strokeColor=colors.darkblue))
        drawing.add(String(x + 60, y + 13, label, textAnchor="middle", fontSize=9))
    for x1, y1, x2, y2 in ((130, 127, 165, 127), (285, 127, 320, 127),
                           (380, 110, 150, 60), (210, 60, 275, 42)):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.darkblue, strokeWidth=1.5))
    return drawing


def section(title: str, text: str, styles: dict) -> list:
    return [Paragraph(title, styles["Heading1"]), Paragraph(text, styles["BodyText"]), Spacer(1, 0.1 * inch)]


def build_report(path: str = "reports/final_report.pdf") -> None:
    metrics = json.loads(Path("reports/model_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads(Path("reports/pipeline_summary.json").read_text(encoding="utf-8"))
    registry = summary["registry"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenteredBody", parent=styles["BodyText"], alignment=TA_CENTER))
    story = [
        Paragraph("Ceramic Kiln Energy Consumption Prediction", styles["Title"]),
        Paragraph("End-to-End MLOps Application Report", styles["Heading2"]),
        Spacer(1, 0.3 * inch),
        Paragraph("Prepared from verified local pipeline outputs — 18 November 2026", styles["CenteredBody"]),
        Spacer(1, 0.3 * inch),
        Paragraph("This report documents a reproducible synthetic-data workflow. Results demonstrate engineering integration and do not establish real-world industrial accuracy.", styles["BodyText"]),
        PageBreak(),
    ]
    story += section("Abstract", "This project estimates electricity consumption for scheduled ceramic kiln firing batches. It combines versioned synthetic datasets, physical proxy features, automated validation, supervised regression, MLflow tracking and registry, FastAPI serving, automated tests, and CI/CD configuration. The complete local execution produced validated data, registered a model, and exercised the live prediction endpoint.", styles)
    story += section("1. Introduction and problem statement", "Ceramic production units need an energy estimate before firing begins to plan costs and capacity. Electricity demand depends on material load, firing temperature, moisture, duration, kiln geometry, insulation, and ambient conditions. A reproducible machine-learning workflow can standardize estimation while preserving dataset and model lineage.", styles)
    story += section("2. Objectives and existing-system limitations", "The objectives are to generate controlled datasets, validate physical ranges, reuse feature transformations, compare models using MAE/RMSE/R², register the selected model, expose a documented API, and automate quality checks. A manual spreadsheet or ad-hoc script does not provide repeatable dataset versions, experiment lineage, schema validation, or a deployable service.", styles)
    story += section("3. Proposed system and technology stack", "The proposed system uses Python, Pandas, NumPy, scikit-learn, Parquet, MLflow, FastAPI, Uvicorn, Pytest, Ruff, GitHub Actions, Matplotlib, Seaborn, and ReportLab. Configuration is kept in config.yaml, while the feature transformation is shared by training and inference.", styles)
    story.append(architecture_diagram())
    story.append(Spacer(1, 0.1 * inch))
    story += section("4. Dataset generation methodology", "Each version contains 400 records with the same seven input features and an energy target. Energy combines pottery thermal demand, kiln heating demand, moisture evaporation, firing duration, explicit insulation heat loss, ambient temperature effects, and Gaussian measurement noise. Version 1 uses modern-kiln insulation; Version 2 uses lower insulation and adds an older-kiln heat-loss contribution. This is a transparent synthetic benchmark, not an industrial measurement study.", styles)
    comparison_rows = [["Statistic", "Modern v1", "Older v2"], ["Records", "400", "400"],
                       ["Mean energy (kWh)", "744.81", "1527.12"], ["Minimum (kWh)", "89.15", "166.52"],
                       ["Maximum (kWh)", "2364.09", "4687.34"], ["Mean insulation", "0.913", "0.800"],
                       ["Mean moisture (%)", "10.718", "10.977"], ["Mean temperature (C)", "944.58", "946.63"]]
    comparison_table = Table(comparison_rows, colWidths=[2.0 * inch, 1.5 * inch, 1.5 * inch])
    comparison_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story += [Paragraph("Version comparison", styles["Heading2"]), comparison_table]
    figure = Path("reports/figures/energy_comparison.png")
    if figure.exists():
        story += [Image(str(figure), width=6.2 * inch, height=3.1 * inch)]
    story += section("5. Data pipeline and feature store", "The pipeline checks required columns, record count, duplicates, numeric types, missing values, and physical ranges before transformation. The current policy rejects missing or invalid values rather than silently imputing them. Features are written to versioned Parquet files and validated on read, ensuring training and inference use the same schema.", styles)
    story += section("6. Feature engineering", "thermal_load_index is calculated as pottery_load_kg multiplied by target_temperature_c and divided by kiln_volume_m3 multiplied by kiln_insulation_rating. It is a proxy for thermal demand and is sensitive to small insulation or volume values. moisture_removal_load is pottery_load_kg multiplied by initial_moisture_pct divided by 100. It approximates water mass to remove and does not model material-specific thermodynamics.", styles)
    story += section("7. Machine-learning methodology", "The dataset is split into 80% training and 20% testing with random_state 42. Linear Regression provides an interpretable baseline. Random Forest Regressor captures nonlinear interactions and uses 200 trees with minimum leaf size 2. The test set is used for final comparison, not training.", styles)
    rows = [["Model", "MAE", "RMSE", "R2"]]
    for name in ("LinearRegression", "RandomForestRegressor"):
        item = metrics[name]
        rows.append([name, f"{item['mae']:.3f}", f"{item['rmse']:.3f}", f"{item['r2']:.4f}"])
    model_table = Table(rows)
    model_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story += [Paragraph("8. Evaluation results", styles["Heading1"]), model_table,
              Paragraph(f"Random Forest was selected because it achieved the lowest RMSE. Selected model: {summary['best_model']}.", styles["BodyText"])]
    for filename in ("actual_vs_predicted.png", "residual_distribution.png", "feature_importance.png", "model_comparison.png"):
        image_path = Path("reports/figures") / filename
        if image_path.exists():
            story += [Image(str(image_path), width=5.8 * inch, height=2.7 * inch)]
    story += section("9. MLflow experiment tracking and registry", f"Both model runs were tracked in the Ceramic_Kiln_Energy_Prediction experiment with dataset version, split, random seed, feature names, and evaluation metrics. The selected model was registered as {registry['model_name']} version {registry['model_version']}. The verified run identifier is {registry['run_id']}. Local metadata is stored in reports/registry.json; no UI screenshot is claimed.", styles)
    story += section("10. FastAPI deployment", "The service exposes GET /health and POST /predict. The health response reports the registered model name and loaded version. Pydantic rejects missing, negative, out-of-range, or non-positive inputs with HTTP 422. Predictions are generated by the registered MLflow model and constrained to be non-negative.", styles)
    api_table = Table([["Endpoint", "Example", "Observed result"], ["/health", "GET", "healthy, version 3"], ["/predict", "POST JSON batch", "1131.97 kWh"], ["/predict invalid", "negative load", "HTTP 422"]])
    api_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story += [api_table, Paragraph('Example request: {"pottery_load_kg":150,"target_temperature_c":1100,"initial_moisture_pct":8,"planned_firing_duration_hours":10,"kiln_insulation_rating":0.85,"kiln_volume_m3":3.5,"ambient_temperature_c":30}', styles["Code"])]
    story += section("11. CI/CD and automated testing", "The GitHub Actions workflow is configured for push and pull-request events. It installs dependencies, runs Ruff, and executes Pytest. Local verification completed with 6 passing tests and Ruff passing. GitHub-hosted execution remains pending until this local repository is pushed to GitHub; no external CI success is claimed.", styles)
    story += section("12. End-to-end results", "The pipeline generated both datasets, validated them, wrote features_v2.parquet, trained both models, registered model version 3, generated five figures, and produced this report. The live API returned a numeric prediction and rejected invalid input.", styles)
    story += section("13. Limitations and future enhancements", "Synthetic coefficients and noise are not calibrated to kiln telemetry. The current comparison uses overall dataset statistics; a matched-condition experiment would better isolate kiln-age effects. Future work should add real sensor data, monitoring and drift detection, model rollback, a real configured fallback model, richer EDA, artifact logging for every report asset, and a hosted MLflow backend.", styles)
    story += section("14. Conclusion", "The implementation demonstrates a complete local path from controlled data generation to registered-model serving with reproducibility, validation, testing, and evidence-based reporting. External GitHub execution and repository submission are intentionally left pending.", styles)
    story += section("15. References", "Scikit-learn User Guide: regression models and metrics. MLflow Documentation: tracking and model registry. FastAPI Documentation: request validation and OpenAPI. ReportLab User Guide: PDF document generation. These references describe the open-source technologies used by the implementation.", styles)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(path, pagesize=letter, rightMargin=0.6 * inch, leftMargin=0.6 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch).build(story)

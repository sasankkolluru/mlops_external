"""Generate a factual PDF report from pipeline outputs."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
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


def build_report(path: str = "reports/final_report.pdf") -> None:
    metrics = json.loads(Path("reports/model_metrics.json").read_text(encoding="utf-8"))
    comparison = json.loads(Path("reports/pipeline_summary.json").read_text(encoding="utf-8"))
    registry = comparison["registry"]
    styles = getSampleStyleSheet()
    story = [Paragraph("Ceramic Kiln Energy Consumption Prediction", styles["Title"]),
             Spacer(1, 0.3 * inch),
             Paragraph("End-to-end MLOps project report", styles["Heading2"]),
             Paragraph("This report documents a reproducible synthetic-data workflow. Results demonstrate engineering integration and do not establish real-world industrial accuracy.", styles["BodyText"]),
             PageBreak(),
             Paragraph("Abstract and objectives", styles["Heading1"]),
             Paragraph("The application predicts scheduled ceramic firing electricity consumption from load, temperature, moisture, duration, insulation, volume, and ambient temperature. It generates two versioned datasets, validates and transforms features, compares regression models, tracks experiments in MLflow, registers the selected model, and serves predictions through FastAPI.", styles["BodyText"]),
             Paragraph("Dataset and feature engineering", styles["Heading1"]),
             Paragraph("Version 1 represents a modern kiln; Version 2 explicitly adds an older-kiln heat-loss term rather than applying a blanket multiplier. thermal_load_index proxies thermal demand and moisture_removal_load proxies the mass of water to remove. Both are approximations and require calibration with industrial data.", styles["BodyText"])]
    rows = [["Model", "MAE", "RMSE", "R2"]]
    for name in ("LinearRegression", "RandomForestRegressor"):
        item = metrics[name]
        rows.append([name, f"{item['mae']:.3f}", f"{item['rmse']:.3f}", f"{item['r2']:.3f}"])
    table = Table(rows)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                               ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story += [Paragraph("Model evaluation", styles["Heading1"]), table, Spacer(1, 0.2 * inch),
              Paragraph(f"Selected model: {comparison['best_model']}. MLflow registered name: {registry['model_name']}; version: {registry['model_version']}.", styles["BodyText"])]
    figure = Path("reports/figures/energy_comparison.png")
    if figure.exists():
        story += [Spacer(1, 0.2 * inch), Image(str(figure), width=6.2 * inch, height=3.1 * inch)]
    story += [Paragraph("Serving, CI/CD, and limitations", styles["Heading1"]),
              Paragraph("FastAPI exposes /health and /predict, with Pydantic range validation and a non-negative output constraint. GitHub Actions is prepared for push and pull request validation but is marked pending until a GitHub-hosted run exists. Limitations include synthetic assumptions, local MLflow storage, and absence of industrial sensor drift monitoring.", styles["BodyText"]),
              Paragraph("Conclusion", styles["Heading1"]),
              Paragraph("The project provides a complete, reproducible path from data generation to registered-model serving, with actual generated artifacts and measured evaluation outputs.", styles["BodyText"])]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(path, pagesize=letter, rightMargin=0.6 * inch, leftMargin=0.6 * inch).build(story)

"""
Enterprise HR AI — Attrition Prediction Service
Wraps the ML predictor with logging and structured output.
"""
import pandas as pd

from app.ml.predictor import predict_employee as _predict
from app.utils.config import DATA_RAW
from app.utils.logger import api_logger


def predict(data: dict) -> dict:
    """Run attrition prediction with logging."""
    api_logger.info("Prediction request received for Age=%s, Role=%s", data.get("Age"), data.get("JobRole"))
    result = _predict(data)
    api_logger.info(
        "Prediction completed: %s (prob=%.4f, risk=%s)",
        result["prediction"],
        result["attrition_probability"],
        result["risk_level"],
    )
    return result


def get_department_summary() -> list[dict]:
    """Get attrition counts and rates by department from the raw data."""
    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    dept_stats = (
        df.groupby("Department")
        .agg(
            total=("Attrition", "count"),
            left=("Attrition", lambda x: (x == "Yes").sum()),
        )
        .reset_index()
    )
    dept_stats["attrition_rate"] = (dept_stats["left"] / dept_stats["total"] * 100).round(1)
    return dept_stats.to_dict(orient="records")


def get_risk_distribution() -> list[dict]:
    """Get count of employees by risk level from raw data."""
    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    counts = df["Attrition"].value_counts().to_dict()
    total = len(df)
    return [
        {"category": "Stay (No)", "count": counts.get("No", 0), "percentage": round(counts.get("No", 0) / total * 100, 1)},
        {"category": "Leave (Yes)", "count": counts.get("Yes", 0), "percentage": round(counts.get("Yes", 0) / total * 100, 1)},
    ]

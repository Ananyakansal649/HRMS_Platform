"""
Enterprise HR AI — FastAPI Backend
Refactored per spec tasks 17-21.
Endpoints: prediction, dashboard, employees, monitoring.
"""
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.validation.employee_schema import EmployeeInput
from app.ml.model_loader import ModelLoader
from app.ml.predictor import (
    predict_employee,
    DEPARTMENTS, EDUCATION_FIELDS, JOB_ROLES,
    BUSINESS_TRAVELS, MARITAL_STATUSES, GENDERS,
)
from app.services.attrition_service import (
    predict as attrition_predict,
    get_department_summary,
    get_risk_distribution,
)
from app.services.engagement_service import (
    get_engagement_by_department,
    get_lowest_engagement_employees,
    get_engagement_stats,
)
from app.services.skill_gap_service import (
    get_required_skills_by_role,
    get_organization_skill_gaps,
)
from app.services.recommendation_service import generate_recommendations
from app.monitoring import (
    log_prediction, compute_drift_stats, get_model_health,
    compute_feature_drift, compute_training_reference_stats,
    log_actual_outcome, compute_model_performance,
    check_retrain_conditions,
)
from app.utils.logger import api_logger


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    api_logger.info("Starting Enterprise HR AI API...")
    ModelLoader().load()
    api_logger.info("Model loaded successfully.")
    # Pre-load skill data to avoid first-request latency
    from app.services.skill_gap_service import get_required_skills_by_role
    role_skills = get_required_skills_by_role()
    api_logger.info("Skill data loaded: %d roles defined", len(role_skills))
    yield
    api_logger.info("Shutting down.")


app = FastAPI(
    title="Enterprise HR AI - Workforce Intelligence API",
    description="Attrition prediction, engagement analytics, skill gaps, and upskilling recommendations.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str
    prediction_encoded: int
    attrition_probability: float
    no_attrition_probability: float
    risk_level: str
    model_version: str
    model_algorithm: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float


start_time = time.time()


# ---------------------------------------------------------------------------
# Health & Model Info
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health / status endpoint."""
    try:
        model, meta, _ = ModelLoader().load()
        return HealthResponse(
            status="healthy",
            model_loaded=model is not None,
            model_version=meta.get("version"),
            uptime_seconds=round(time.time() - start_time, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Return model metadata and feature list."""
    _, metadata, feature_names = ModelLoader().load()
    return {
        "metadata": metadata,
        "n_features": len(feature_names),
        "features": feature_names,
    }


@app.get("/options")
async def get_options():
    """Return valid categorical options for form building."""
    return {
        "departments": DEPARTMENTS,
        "education_fields": EDUCATION_FIELDS,
        "job_roles": JOB_ROLES,
        "business_travels": BUSINESS_TRAVELS,
        "marital_statuses": MARITAL_STATUSES,
        "genders": GENDERS,
    }


# ---------------------------------------------------------------------------
# Prediction (spec task 18)
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: EmployeeInput):
    """Predict attrition risk for a single employee."""
    try:
        input_dict = data.model_dump()
        result = attrition_predict(input_dict)

        log_prediction(
            input_data=input_dict,
            prediction=result["prediction"],
            probability=result["attrition_probability"],
            risk_level=result["risk_level"],
            model_version=result["model_version"],
            endpoint="api",
        )

        return PredictionResponse(**result)
    except Exception as e:
        api_logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(employees: list[EmployeeInput]):
    """Batch prediction for multiple employees (max 100)."""
    if len(employees) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 employees")

    results = []
    for emp in employees:
        try:
            input_dict = emp.model_dump()
            result = attrition_predict(input_dict)
            log_prediction(
                input_data=input_dict,
                prediction=result["prediction"],
                probability=result["attrition_probability"],
                risk_level=result["risk_level"],
                model_version=result["model_version"],
                endpoint="api_batch",
            )
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})

    return {"predictions": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Dashboard endpoints (spec task 18)
# ---------------------------------------------------------------------------
@app.get("/dashboard/summary")
async def dashboard_summary():
    """KPI summary: total employees, high-risk count, average engagement."""
    from app.utils.config import DATA_RAW
    import pandas as pd

    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    total = len(df)
    high_risk = int((df["Attrition"] == "Yes").sum())

    eng = get_engagement_stats()
    return {
        "total_employees": total,
        "high_risk_employees": high_risk,
        "high_risk_rate": round(high_risk / total * 100, 1) if total > 0 else 0,
        "average_engagement": eng.get("avg", 0),
        "engagement_records": eng.get("count", 0),
    }


@app.get("/dashboard/attrition-by-department")
async def dashboard_attrition_by_department():
    """Attrition rates by department for chart data."""
    return get_department_summary()


@app.get("/dashboard/risk-distribution")
async def dashboard_risk_distribution():
    """Risk level distribution."""
    return get_risk_distribution()


@app.get("/dashboard/engagement-by-department")
async def dashboard_engagement_by_department():
    """Average engagement by department."""
    return get_engagement_by_department()


@app.get("/dashboard/low-engagement")
async def dashboard_low_engagement(n: int = 10):
    """Lowest engagement employees."""
    return get_lowest_engagement_employees(n)


@app.get("/dashboard/skill-gaps")
async def dashboard_skill_gaps():
    """Organization-wide skill gaps with severity."""
    # Use synthetic employee data for MVP (raw data lacks per-employee skills)
    from app.services.skill_gap_service import get_required_skills_by_role
    role_skills = get_required_skills_by_role()

    # Generate synthetic employee-role mapping from attrition data
    import pandas as pd
    from app.utils.config import DATA_RAW
    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df.iterrows()
    ]

    # All employees start with no skills (since raw data lacks skill info)
    employee_skills = {}

    org_gaps = get_organization_skill_gaps(employee_roles, employee_skills)
    return {
        "gaps": org_gaps[:20],  # Top 20 most critical
        "total_skills_gap_count": len(org_gaps),
        "note": "Based on synthetic skill assignments (raw data lacks per-employee skills)",
    }


@app.get("/dashboard/recommendations")
async def dashboard_recommendations(n: int = 10):
    """Top N upskilling recommendations."""
    import pandas as pd
    from app.utils.config import DATA_RAW
    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))

    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df.head(50).iterrows()
    ]
    employee_skills = {}

    recs = generate_recommendations(employee_roles, employee_skills)
    return {"recommendations": recs[:n], "total": len(recs)}


# ---------------------------------------------------------------------------
# Employee detail (spec task 18)
# ---------------------------------------------------------------------------
@app.get("/employees/{employee_id}")
async def employee_detail(employee_id: str):
    """Full intelligence record for one employee."""
    import pandas as pd
    from app.utils.config import DATA_RAW

    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    emp = df[df["EmployeeNumber"].astype(str) == employee_id]

    if emp.empty:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    row = emp.iloc[0]

    # Get prediction
    input_data = {
        "Age": int(row.get("Age", 35)),
        "Department": row.get("Department", "Research & Development"),
        "DistanceFromHome": int(row.get("DistanceFromHome", 0)),
        "Education": int(row.get("Education", 3)),
        "EducationField": row.get("EducationField", "Life Sciences"),
        "EnvironmentSatisfaction": int(row.get("EnvironmentSatisfaction", 3)),
        "Gender": row.get("Gender", "Male"),
        "JobInvolvement": int(row.get("JobInvolvement", 3)),
        "JobLevel": int(row.get("JobLevel", 2)),
        "JobRole": row.get("JobRole", "Research Scientist"),
        "JobSatisfaction": int(row.get("JobSatisfaction", 3)),
        "MaritalStatus": row.get("MaritalStatus", "Single"),
        "MonthlyIncome": int(row.get("MonthlyIncome", 5000)),
        "NumCompaniesWorked": int(row.get("NumCompaniesWorked", 2)),
        "OverTime": row.get("OverTime", "No"),
        "PercentSalaryHike": int(row.get("PercentSalaryHike", 15)),
        "PerformanceRating": int(row.get("PerformanceRating", 3)),
        "RelationshipSatisfaction": int(row.get("RelationshipSatisfaction", 3)),
        "StockOptionLevel": int(row.get("StockOptionLevel", 1)),
        "TotalWorkingYears": int(row.get("TotalWorkingYears", 5)),
        "TrainingTimesLastYear": int(row.get("TrainingTimesLastYear", 2)),
        "WorkLifeBalance": int(row.get("WorkLifeBalance", 3)),
        "YearsAtCompany": int(row.get("YearsAtCompany", 3)),
        "YearsInCurrentRole": int(row.get("YearsInCurrentRole", 2)),
        "YearsSinceLastPromotion": int(row.get("YearsSinceLastPromotion", 1)),
        "YearsWithCurrManager": int(row.get("YearsWithCurrManager", 2)),
        "HourlyRate": int(row.get("HourlyRate", 60)),
        "DailyRate": int(row.get("DailyRate", 800)),
        "MonthlyRate": int(row.get("MonthlyRate", 15000)),
        "BusinessTravel": row.get("BusinessTravel", "Travel_Rarely"),
    }

    prediction = predict_employee(input_data)

    # Skill gaps for this employee's role
    from app.services.skill_gap_service import compute_employee_skill_gaps
    gaps = compute_employee_skill_gaps(
        [{"employee_id": employee_id, "role": row.get("JobRole", "Unknown")}],
        {}  # no per-employee skill data available
    )
    skill_gap = gaps[0] if gaps else {}

    # Recommendations for missing skills
    from app.services.recommendation_service import generate_recommendations
    recs = generate_recommendations(
        [{"employee_id": employee_id, "role": row.get("JobRole", "Unknown")}],
        {}
    )
    recommendations = recs[0].get("recommendations", []) if recs else []

    return {
        "employee_id": employee_id,
        "department": row.get("Department"),
        "job_role": row.get("JobRole"),
        "age": int(row.get("Age", 0)),
        "monthly_income": int(row.get("MonthlyIncome", 0)),
        "years_at_company": int(row.get("YearsAtCompany", 0)),
        "actual_attrition": row.get("Attrition"),
        "prediction": prediction,
        "skill_gaps": skill_gap.get("gap_count", 0),
        "missing_skills": skill_gap.get("missing_skills", [])[:10],
        "recommendations": recommendations[:5],
    }


# ---------------------------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------------------------
@app.get("/monitoring/drift")
async def monitoring_drift():
    """Check for prediction distribution drift."""
    return compute_drift_stats()


@app.get("/monitoring/health")
async def monitoring_health():
    """Full model health check with monitoring stats."""
    return get_model_health()


@app.get("/monitoring/feature-drift")
async def monitoring_feature_drift():
    """Task 25: Compare production input distributions against training data."""
    return compute_feature_drift()


@app.get("/monitoring/training-reference")
async def monitoring_training_reference():
    """Task 25: Get training data reference statistics."""
    return compute_training_reference_stats()


@app.get("/monitoring/model-performance")
async def monitoring_model_performance():
    """Task 26: Recompute live precision/recall/F1 from actual outcomes."""
    return compute_model_performance()


@app.post("/monitoring/log-outcome")
async def monitoring_log_outcome(
    employee_id: str,
    predicted: str,
    predicted_probability: float,
    actual_outcome: str,
):
    """Task 26: Log an actual attrition outcome for performance tracking."""
    return log_actual_outcome(
        employee_id=employee_id,
        predicted=predicted,
        predicted_probability=predicted_probability,
        actual_outcome=actual_outcome,
    )


@app.get("/monitoring/retrain-check")
async def monitoring_retrain_check():
    """Task 27: Check if model should be retrained based on automated rules."""
    return check_retrain_conditions()


# ---------------------------------------------------------------------------
# Direct execution for development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

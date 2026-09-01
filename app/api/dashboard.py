"""
Enterprise HR AI — Dashboard API Routes
Route module for dashboard/summary endpoints.
"""
from fastapi import APIRouter

from app.services.attrition_service import get_department_summary, get_risk_distribution
from app.services.engagement_service import (
    get_engagement_by_department,
    get_engagement_stats,
    get_lowest_engagement_employees,
)
from app.services.skill_gap_service import (
    get_required_skills_by_role,
    get_organization_skill_gaps,
)
from app.services.recommendation_service import generate_recommendations

import pandas as pd
from app.utils.config import DATA_RAW

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
async def dashboard_summary():
    """KPI summary: total employees, high-risk count, average engagement."""
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


@router.get("/attrition-by-department")
async def dashboard_attrition_by_department():
    """Attrition rates by department for chart data."""
    return get_department_summary()


@router.get("/risk-distribution")
async def dashboard_risk_distribution():
    """Risk level distribution."""
    return get_risk_distribution()


@router.get("/engagement-by-department")
async def dashboard_engagement_by_department():
    """Average engagement by department."""
    return get_engagement_by_department()


@router.get("/low-engagement")
async def dashboard_low_engagement(n: int = 10):
    """Lowest engagement employees."""
    return get_lowest_engagement_employees(n)


@router.get("/skill-gaps")
async def dashboard_skill_gaps():
    """Organization-wide skill gaps with severity."""
    role_skills = get_required_skills_by_role()
    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df.iterrows()
    ]
    employee_skills = {}
    org_gaps = get_organization_skill_gaps(employee_roles, employee_skills)

    return {
        "gaps": org_gaps[:20],
        "total_skills_gap_count": len(org_gaps),
        "note": "Based on role requirements (raw data lacks per-employee skills)",
    }


@router.get("/recommendations")
async def dashboard_recommendations(n: int = 10):
    """Top N upskilling recommendations."""
    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df.head(50).iterrows()
    ]
    employee_skills = {}
    recs = generate_recommendations(employee_roles, employee_skills)

    return {"recommendations": recs[:n], "total": len(recs)}

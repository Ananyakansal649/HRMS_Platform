"""
Enterprise HR AI — Skills API Routes
Route module for skill-gap and recommendation endpoints.
"""
from fastapi import APIRouter

from app.services.skill_gap_service import (
    get_required_skills_by_role,
    get_organization_skill_gaps,
    compute_employee_skill_gaps,
)
from app.services.recommendation_service import generate_recommendations

router = APIRouter(tags=["skills"])


@router.get("/skills/roles")
async def list_skill_roles():
    """List all roles with defined skills and their skill counts."""
    role_skills = get_required_skills_by_role()
    return {
        "roles": [
            {"role": role, "skill_count": len(skills)}
            for role, skills in sorted(role_skills.items())
        ],
        "total_roles": len(role_skills),
    }


@router.get("/skills/roles/{role_name}")
async def get_role_skills(role_name: str):
    """Get required skills for a specific role."""
    role_skills = get_required_skills_by_role()
    # Case-insensitive search
    for role, skills in role_skills.items():
        if role.lower() == role_name.lower():
            return {
                "role": role,
                "required_skills": sorted(skills),
                "skill_count": len(skills),
            }
    return {"error": f"Role '{role_name}' not found", "available_roles": list(role_skills.keys())[:20]}


@router.post("/skills/gaps")
async def compute_skill_gaps(employee_roles: list[dict], employee_skills: dict = {}):
    """Compute skill gaps for given employees."""
    return compute_employee_skill_gaps(employee_roles, employee_skills)


@router.get("/skills/organization-gaps")
async def organization_skill_gaps():
    """Get organization-wide skill gaps."""
    import pandas as pd
    from app.utils.config import DATA_RAW

    df = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df.iterrows()
    ]
    employee_skills = {}
    return get_organization_skill_gaps(employee_roles, employee_skills)


@router.get("/skills/recommendations")
async def get_recommendations(n: int = 10):
    """Get upskilling recommendations for employees."""
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

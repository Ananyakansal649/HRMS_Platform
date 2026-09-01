"""
Enterprise HR AI — Skill Gap Service
Computes per-employee and organization-wide skill gaps
by comparing required role skills against employee skills.
"""
import pandas as pd

from app.utils.config import DATA_RAW

# Module-level cache to avoid re-reading large CSV files
_role_skills_cache: dict | None = None


def get_required_skills_by_role() -> dict[str, set]:
    """
    Load required skills per role from essential_skills + software_skills.
    Returns: {role_title: {skill1, skill2, ...}}
    """
    global _role_skills_cache
    if _role_skills_cache is not None:
        return _role_skills_cache

    ess = pd.read_csv(str(DATA_RAW / "essential_skills.csv"))
    soft = pd.read_csv(str(DATA_RAW / "software_skills.csv"))
    occ = pd.read_csv(str(DATA_RAW / "occupation_data.csv"))

    # Build occupation title lookup: SOC code -> title
    soc_to_title = {}
    for _, row in occ.iterrows():
        code = row.get("O*NET-SOC Code", "")
        title = row.get("Title", "")
        if code and title:
            soc_to_title[code] = title

    # Essential skills: SOC code -> set of skill names
    ess_skills: dict[str, set] = {}
    for _, row in ess.iterrows():
        soc = row.get("O*NET-SOC Code", "")
        skill = row.get("Element Name", "")
        if soc and skill:
            ess_skills.setdefault(soc, set()).add(skill)

    # Software skills: SOC code -> set of software names
    soft_skills: dict[str, set] = {}
    for _, row in soft.iterrows():
        soc = row.get("O*NET-SOC Code", "")
        sw = row.get("Element Name", "")
        if soc and sw:
            soft_skills.setdefault(soc, set()).add(sw)

    # Combine per title
    role_skills: dict[str, set] = {}
    for soc, title in soc_to_title.items():
        combined = set()
        if soc in ess_skills:
            combined |= ess_skills[soc]
        if soc in soft_skills:
            combined |= soft_skills[soc]
        if combined:
            role_skills[title] = combined

    _role_skills_cache = role_skills
    return role_skills


def compute_employee_skill_gaps(employee_roles: list[dict], employee_skills: dict[str, set]) -> list[dict]:
    """
    Compute skill gaps for a list of employees.

    Args:
        employee_roles: [{"employee_id": "101", "role": "Data Analyst"}, ...]
        employee_skills: {"101": {"Python", "SQL"}, ...}

    Returns:
        [{"employee_id": "101", "role": "Data Analyst", "required": 15, "have": 8, "gap": ["MLOps", "Docker"]}, ...]
    """
    role_skills = get_required_skills_by_role()

    results = []
    for emp in employee_roles:
        eid = str(emp["employee_id"])
        role = emp.get("role", "Unknown")
        required = set()
        for role_title, skills in role_skills.items():
            if role.lower() in role_title.lower() or role_title.lower() in role.lower():
                required |= skills
        if not required:
            # Try partial match
            for role_title, skills in role_skills.items():
                for word in role.lower().split():
                    if word in role_title.lower():
                        required |= skills
                        break

        have = employee_skills.get(eid, set())
        gap = sorted(required - have)

        results.append({
            "employee_id": eid,
            "role": role,
            "required_skills": len(required),
            "current_skills": len(have),
            "gap_count": len(gap),
            "missing_skills": gap,
        })

    return results


def get_organization_skill_gaps(employee_roles: list[dict], employee_skills: dict[str, set]) -> list[dict]:
    """
    Aggregate skill gaps across all employees to find organization-wide gaps.

    Returns: [{"skill": "MLOps", "missing_count": 120, "severity": "HIGH"}, ...]
    """
    role_skills = get_required_skills_by_role()
    skill_missing_count: dict[str, int] = {}

    for emp in employee_roles:
        eid = str(emp["employee_id"])
        role = emp.get("role", "Unknown")
        required = set()
        for role_title, skills in role_skills.items():
            if role.lower() in role_title.lower() or role_title.lower() in role.lower():
                required |= skills
        if not required:
            for role_title, skills in role_skills.items():
                for word in role.lower().split():
                    if word in role_title.lower():
                        required |= skills
                        break
        have = employee_skills.get(eid, set())
        for skill in (required - have):
            skill_missing_count[skill] = skill_missing_count.get(skill, 0) + 1

    # Sort by count descending and add severity
    result = []
    for skill, count in sorted(skill_missing_count.items(), key=lambda x: -x[1]):
        if count >= 100:
            severity = "HIGH"
        elif count >= 50:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        result.append({"skill": skill, "missing_count": count, "severity": severity})

    return result

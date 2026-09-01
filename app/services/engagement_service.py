"""
Enterprise HR AI — Engagement Analytics Service
Provides engagement intelligence from the performance/engagement dataset.
"""
import pandas as pd

from app.utils.config import DATA_RAW


def get_engagement_by_department() -> list[dict]:
    """Average engagement score broken down by department."""
    df = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
    eng_col = _find_engagement_col(df)
    dept_col = _find_dept_col(df)
    if eng_col is None or dept_col is None:
        return []
    result = (
        df.groupby(dept_col)[eng_col]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )
    result.columns = ["department", "avg_engagement"]
    return result.to_dict(orient="records")


def get_lowest_engagement_employees(n: int = 10) -> list[dict]:
    """Return the N employees with the lowest engagement scores."""
    df = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
    eng_col = _find_engagement_col(df)
    id_col = _find_id_col(df)
    dept_col = _find_dept_col(df)
    if eng_col is None or id_col is None:
        return []
    cols = [id_col, eng_col]
    if dept_col:
        cols.append(dept_col)
    top = df.nsmallest(n, eng_col)[cols].to_dict(orient="records")
    return top


def get_engagement_stats() -> dict:
    """Overall engagement statistics."""
    df = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
    eng_col = _find_engagement_col(df)
    if eng_col is None:
        return {"avg": 0, "min": 0, "max": 0, "count": 0}
    return {
        "avg": round(float(df[eng_col].mean()), 2),
        "min": round(float(df[eng_col].min()), 2),
        "max": round(float(df[eng_col].max()), 2),
        "count": int(len(df)),
    }


def _find_engagement_col(df: pd.DataFrame):
    """Find the engagement score column by name patterns."""
    for col in df.columns:
        if "engagement" in col.lower() and "score" in col.lower():
            return col
    for col in df.columns:
        if "engagement" in col.lower():
            return col
    return None


def _find_dept_col(df: pd.DataFrame):
    for col in df.columns:
        if col.lower() in ("department", "departmenttype", "department_type"):
            return col
    return None


def _find_id_col(df: pd.DataFrame):
    for col in df.columns:
        if "id" in col.lower() and "employee" in col.lower():
            return col
    return None

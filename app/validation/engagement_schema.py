"""
Enterprise HR AI — Engagement Input Validation
Pydantic models for validating engagement data inputs.
"""
from pydantic import BaseModel, Field


class EngagementInput(BaseModel):
    """Employee engagement data for analysis."""
    employee_id: str = Field(..., description="Employee identifier")
    department: str = Field(..., description="Department name")
    engagement_score: float = Field(..., ge=0, le=100, description="Engagement score (0-100)")
    satisfaction_score: float = Field(None, ge=0, le=100, description="Satisfaction score (0-100)")
    work_life_balance: float = Field(None, ge=0, le=100, description="Work-life balance score (0-100)")

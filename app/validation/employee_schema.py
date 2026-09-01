"""
Enterprise HR AI — Employee Input Validation
Pydantic models for validating prediction requests (spec task 19).
"""
from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    """Raw employee attributes for attrition prediction."""
    Age: int = Field(..., ge=18, le=100, description="Employee age (18-100)")
    MonthlyIncome: int = Field(..., ge=0, description="Monthly income")
    TotalWorkingYears: int = Field(..., ge=0, description="Total years of work experience")
    YearsAtCompany: int = Field(0, ge=0, description="Years at current company")
    YearsInCurrentRole: int = Field(0, ge=0, description="Years in current role")
    YearsSinceLastPromotion: int = Field(0, ge=0, description="Years since last promotion")
    YearsWithCurrManager: int = Field(0, ge=0, description="Years with current manager")
    DistanceFromHome: int = Field(0, ge=0, description="Distance from home in km")
    Education: int = Field(3, ge=1, le=5, description="Education level (1-5)")
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4, description="Environment satisfaction (1-4)")
    JobInvolvement: int = Field(3, ge=1, le=4, description="Job involvement (1-4)")
    JobLevel: int = Field(2, ge=1, le=5, description="Job level (1-5)")
    JobSatisfaction: int = Field(3, ge=1, le=4, description="Job satisfaction (1-4)")
    NumCompaniesWorked: int = Field(3, ge=0, description="Number of companies worked at")
    OverTime: str = Field("No", description="Overtime: Yes or No")
    PercentSalaryHike: int = Field(14, ge=0, le=25, description="Salary hike %")
    PerformanceRating: int = Field(3, ge=3, le=4, description="Performance rating (3 or 4)")
    RelationshipSatisfaction: int = Field(3, ge=1, le=4, description="Relationship satisfaction (1-4)")
    StockOptionLevel: int = Field(1, ge=0, le=3, description="Stock option level (0-3)")
    TrainingTimesLastYear: int = Field(2, ge=0, description="Training sessions last year")
    WorkLifeBalance: int = Field(3, ge=1, le=4, description="Work-life balance (1-4)")
    Department: str = Field("Research & Development", description="Department name")
    EducationField: str = Field("Life Sciences", description="Education field")
    JobRole: str = Field("Research Scientist", description="Job role")
    BusinessTravel: str = Field("Travel_Rarely", description="Business travel frequency")
    MaritalStatus: str = Field("Single", description="Marital status")
    Gender: str = Field("Male", description="Gender: Female or Male")

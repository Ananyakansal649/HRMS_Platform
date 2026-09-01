"""
Enterprise HR AI — Predictor
Encodes raw employee input and returns predictions using the trained model.
"""
import numpy as np
import pandas as pd

from app.ml.model_loader import ModelLoader

# Categorical options (before encoding)
DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
EDUCATION_FIELDS = [
    "Human Resources", "Life Sciences", "Marketing", "Medical",
    "Other", "Technical Degree",
]
JOB_ROLES = [
    "Healthcare Representative", "Human Resources", "Laboratory Technician",
    "Manager", "Manufacturing Director", "Research Director",
    "Research Scientist", "Sales Executive", "Sales Representative",
]
BUSINESS_TRAVELS = ["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
MARITAL_STATUSES = ["Divorced", "Married", "Single"]
GENDERS = ["Female", "Male"]


def encode_employee_input(data: dict) -> pd.DataFrame:
    """
    Convert raw employee attributes into the 44-feature vector
    expected by the model. Mirrors 05_feature_engineering.ipynb logic.
    """
    record = {}

    # Direct numeric features
    numeric_fields = [
        "Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
        "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    ]
    for field in numeric_fields:
        if field in data and data[field] is not None:
            record[field] = int(data[field])
        else:
            record[field] = 0

    # Binary: OverTime
    overtime_val = data.get("OverTime", "No")
    record["OverTime"] = 1 if overtime_val in ("Yes", "yes", 1, True) else 0

    # Engineered features
    total_working = record["TotalWorkingYears"] if record["TotalWorkingYears"] > 0 else 1
    monthly_income = record["MonthlyIncome"]

    record["income_per_year"] = monthly_income * 12 / total_working
    record["years_since_promotion_ratio"] = (
        record["YearsSinceLastPromotion"] / total_working
    )
    record["satisfaction_score"] = (
        record["JobSatisfaction"]
        + record["EnvironmentSatisfaction"]
        + record["WorkLifeBalance"]
    ) / 3.0

    # One-hot encode categoricals (drop_first=True)

    # BusinessTravel
    bt = data.get("BusinessTravel", "Travel_Rarely")
    record["BusinessTravel_Travel_Frequently"] = 1 if bt == "Travel_Frequently" else 0
    record["BusinessTravel_Travel_Rarely"] = 1 if bt == "Travel_Rarely" else 0

    # Department
    dept = data.get("Department", "Research & Development")
    record["Department_Research & Development"] = 1 if dept == "Research & Development" else 0
    record["Department_Sales"] = 1 if dept == "Sales" else 0

    # EducationField (drop_first = "Human Resources")
    ef = data.get("EducationField", "Life Sciences")
    for ef_val in ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]:
        record[f"EducationField_{ef_val}"] = 1 if ef == ef_val else 0

    # Gender (drop_first = "Female")
    gender = data.get("Gender", "Female")
    record["Gender_Male"] = 1 if gender == "Male" else 0

    # JobRole (drop_first = "Healthcare Representative")
    jr = data.get("JobRole", "Research Scientist")
    for jr_val in [
        "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative",
    ]:
        record[f"JobRole_{jr_val}"] = 1 if jr == jr_val else 0

    # MaritalStatus (drop_first = "Divorced")
    ms = data.get("MaritalStatus", "Single")
    record["MaritalStatus_Married"] = 1 if ms == "Married" else 0
    record["MaritalStatus_Single"] = 1 if ms == "Single" else 0

    # Build DataFrame with features in exact model order
    _, _, feature_names = ModelLoader().load()
    df = pd.DataFrame([record])
    df = df[feature_names]

    return df


def predict_employee(data: dict) -> dict:
    """
    Full prediction pipeline:
    1. Load model
    2. Encode input
    3. Get prediction and probability
    4. Return structured result
    """
    model, metadata, feature_names = ModelLoader().load()
    df = encode_employee_input(data)

    prediction = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    attrition_prob = float(proba[1])
    no_attrition_prob = float(proba[0])

    risk_level = "Low"
    if attrition_prob >= 0.7:
        risk_level = "High"
    elif attrition_prob >= 0.4:
        risk_level = "Medium"

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "prediction_encoded": prediction,
        "attrition_probability": round(attrition_prob, 4),
        "no_attrition_probability": round(no_attrition_prob, 4),
        "risk_level": risk_level,
        "model_version": metadata.get("version", "unknown"),
        "model_algorithm": metadata.get("algorithm", "unknown"),
    }

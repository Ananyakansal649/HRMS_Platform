"""
Enterprise HR AI — Attrition API Routes
Route module for attrition prediction endpoints.
"""
from fastapi import APIRouter, HTTPException

from app.validation.employee_schema import EmployeeInput
from app.services.attrition_service import predict as attrition_predict
from app.monitoring import log_prediction
from app.utils.logger import api_logger

router = APIRouter(tags=["attrition"])


@router.post("/predict/attrition")
async def predict_attrition(data: EmployeeInput):
    """
    Predict employee attrition risk using the trained XGBoost model.

    Accepts validated employee attributes and returns:
    - prediction (Yes/No)
    - attrition_probability
    - risk_level (Low/Medium/High)
    - model_version
    """
    try:
        input_dict = data.model_dump()
        result = attrition_predict(input_dict)

        log_prediction(
            input_data=input_dict,
            prediction=result["prediction"],
            probability=result["attrition_probability"],
            risk_level=result["risk_level"],
            model_version=result["model_version"],
            endpoint="api_predict_attrition",
        )

        return result
    except Exception as e:
        api_logger.error("Prediction failed at /predict/attrition: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

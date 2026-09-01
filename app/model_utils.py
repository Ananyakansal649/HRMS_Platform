"""
Enterprise HR AI — Model Utilities (Legacy Wrapper)
Re-exports from the new module structure for backward compatibility.
New code should import from app.ml.model_loader and app.ml.predictor directly.
"""
from app.ml.model_loader import ModelLoader
from app.ml.predictor import (
    encode_employee_input,
    predict_employee,
    DEPARTMENTS,
    EDUCATION_FIELDS,
    JOB_ROLES,
    BUSINESS_TRAVELS,
    MARITAL_STATUSES,
    GENDERS,
)
from app.utils.config import MODEL_PATH, METADATA_PATH

__all__ = [
    "ModelLoader",
    "encode_employee_input",
    "predict_employee",
    "MODEL_PATH",
    "METADATA_PATH",
    "DEPARTMENTS",
    "EDUCATION_FIELDS",
    "JOB_ROLES",
    "BUSINESS_TRAVELS",
    "MARITAL_STATUSES",
    "GENDERS",
]

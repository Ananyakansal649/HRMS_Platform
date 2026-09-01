"""
Enterprise HR AI — Central Configuration
All paths, constants, and settings in one place.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PREDICTIONS = PROJECT_ROOT / "data" / "predictions"
DATA_MONITORING = PROJECT_ROOT / "data" / "monitoring"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_VERSIONED = MODELS_DIR / "v1"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model files
MODEL_PATH = MODELS_DIR / "attrition_pipeline.joblib"
METADATA_PATH = MODELS_VERSIONED / "metadata.json"

# Raw dataset files
RAW_FILES = {
    "employee_attrition": DATA_RAW / "employee_attrition.csv",
    "hr_performance_engagement": DATA_RAW / "hr_performance_engagement.csv",
    "occupation_data": DATA_RAW / "occupation_data.csv",
    "essential_skills": DATA_RAW / "essential_skills.csv",
    "software_skills": DATA_RAW / "software_skills.csv",
}

# Processed dataset files
PROCESSED_FILES = {
    "employee_attrition": DATA_PROCESSED / "employee_attrition_processed.csv",
    "engagement": DATA_PROCESSED / "engagement_processed.csv",
    "essential_skills": DATA_PROCESSED / "essential_skills_processed.csv",
    "occupation_master": DATA_PROCESSED / "occupation_master.csv",
    "software_skills": DATA_PROCESSED / "software_skills_processed.csv",
    "attrition_features": DATA_PROCESSED / "attrition_features.csv",
}

# Ensure directories exist
for d in [DATA_PREDICTIONS, DATA_MONITORING]:
    d.mkdir(parents=True, exist_ok=True)

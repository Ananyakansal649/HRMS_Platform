# Enterprise HR AI — Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
cd enterprise_hr_ai
pip install -r requirements.txt
```

### 2. Run FastAPI Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 3. Run Streamlit Dashboard

```bash
streamlit run app/dashboard.py --server.port 8501
```

Dashboard available at: http://localhost:8501

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Run with Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

---

## API Endpoints

### Health & Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health, model status, uptime |
| `/model/info` | GET | Model metadata, feature list |
| `/options` | GET | Valid categorical values for forms |

### Prediction

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single employee attrition prediction |
| `/predict/batch` | POST | Batch prediction (max 100) |

**POST /predict — Request body:**

```json
{
  "Age": 35,
  "Department": "Research & Development",
  "DistanceFromHome": 10,
  "Education": 3,
  "EducationField": "Life Sciences",
  "EnvironmentSatisfaction": 3,
  "Gender": "Male",
  "JobInvolvement": 3,
  "JobLevel": 2,
  "JobRole": "Research Scientist",
  "JobSatisfaction": 3,
  "MaritalStatus": "Married",
  "MonthlyIncome": 6500,
  "NumCompaniesWorked": 3,
  "OverTime": "Yes",
  "PercentSalaryHike": 14,
  "PerformanceRating": 3,
  "RelationshipSatisfaction": 3,
  "StockOptionLevel": 1,
  "TotalWorkingYears": 10,
  "TrainingTimesLastYear": 2,
  "WorkLifeBalance": 3,
  "YearsAtCompany": 5,
  "YearsInCurrentRole": 3,
  "YearsSinceLastPromotion": 1,
  "YearsWithCurrManager": 3,
  "HourlyRate": 65,
  "DailyRate": 800,
  "MonthlyRate": 15000,
  "BusinessTravel": "Travel_Rarely"
}
```

**Response:**

```json
{
  "prediction": "No",
  "prediction_encoded": 0,
  "attrition_probability": 0.0003,
  "no_attrition_probability": 0.9997,
  "risk_level": "Low",
  "model_version": "v1.0",
  "model_algorithm": "XGBoost"
}
```

### Dashboard

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard/summary` | GET | KPI summary (total, high-risk, engagement) |
| `/dashboard/attrition-by-department` | GET | Attrition rates by department |
| `/dashboard/risk-distribution` | GET | Stay vs Leave counts |
| `/dashboard/engagement-by-department` | GET | Average engagement by department |
| `/dashboard/low-engagement` | GET | Lowest engagement employees |
| `/dashboard/skill-gaps` | GET | Organization-wide skill gaps |
| `/dashboard/recommendations` | GET | Upskilling recommendations |

### Employee Detail

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/employees/{id}` | GET | Full intelligence record for one employee |

### Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/monitoring/drift` | GET | Prediction distribution drift stats |
| `/monitoring/health` | GET | Model file health + prediction throughput |

---

## Project Structure

```
enterprise_hr_ai/
├── data/
│   ├── raw/                    (5 original CSVs — DO NOT MODIFY)
│   ├── processed/              (6 cleaned/transformed CSVs)
│   └── monitoring/             (prediction logs)
├── models/
│   ├── attrition_pipeline.joblib   (production model)
│   ├── v1/                         (versioned copy + metadata)
│   └── *.png                       (SHAP + comparison plots)
├── notebooks/                  (9 .ipynb + 9 .html + index)
├── app/
│   ├── main.py                 (FastAPI — 14 endpoints)
│   ├── dashboard.py            (Streamlit — 7 pages)
│   ├── monitoring.py           (prediction logging + drift)
│   ├── model_utils.py          (backward-compatible wrapper)
│   ├── ml/
│   │   ├── model_loader.py     (singleton model loading)
│   │   └── predictor.py        (encoding + inference)
│   ├── services/
│   │   ├── attrition_service.py
│   │   ├── engagement_service.py
│   │   ├── skill_gap_service.py
│   │   └── recommendation_service.py
│   ├── validation/
│   │   └── employee_schema.py  (Pydantic input validation)
│   └── utils/
│       ├── config.py           (central paths/settings)
│       └── logger.py           (structured logging)
├── tests/
│   └── test_app.py             (28 unit tests)
├── docs/
│   ├── data_relationships.md
│   └── day3_usage.md           (this file)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model Information

- **Algorithm:** XGBoost
- **Version:** v1.0
- **Features:** 44
- **ROC-AUC:** 0.798
- **F1 Score:** 0.475
- **Training Date:** 2026-09-01

## Raw Datasets (DO NOT MODIFY)

| File | Rows | Columns |
|------|------|---------|
| employee_attrition.csv | 1,470 | 35 |
| hr_performance_engagement.csv | 2,845 | 28 |
| occupation_data.csv | 1,016 | 3 |
| essential_skills.csv | 18,200 | 15 |
| software_skills.csv | 31,821 | 7 |

# Enterprise HR AI — Workforce Intelligence & Upskilling Platform

## Problem Statement

Employee attrition costs organizations significantly in recruitment, training, and lost productivity. This platform predicts attrition risk, identifies skill gaps, and recommends upskilling paths — turning HR data into actionable intelligence.

## Architecture

```
                         USER
                          |
                 Streamlit UI (frontend)
                          |
                 FastAPI Backend
                          |
      +-------------------+-------------------+
 ML Prediction      Skill Engine         Analytics
      +-------------------+-------------------+
               Employee Intelligence
                          |
               Logging + Monitoring
                          |
                   Model Registry
```

## Project Structure

```
enterprise_hr_ai/
├── data/
│   ├── raw/                    # 5 original CSV datasets (DO NOT MODIFY)
│   ├── processed/              # Cleaned/transformed CSVs
│   ├── monitoring/             # Prediction logs, drift stats
│   └── predictions/            # Prediction event logs
├── models/
│   ├── attrition_pipeline.joblib   # Production XGBoost model
│   └── v1/                         # Versioned model + metadata
├── app/
│   ├── main.py                 # FastAPI (18 endpoints)
│   ├── dashboard.py            # Streamlit (7 pages)
│   ├── monitoring.py           # Drift + performance monitoring
│   ├── ml/
│   │   ├── model_loader.py     # Model loading singleton
│   │   └── predictor.py        # Feature encoding + inference
│   ├── services/
│   │   ├── attrition_service.py
│   │   ├── engagement_service.py
│   │   ├── skill_gap_service.py
│   │   └── recommendation_service.py
│   ├── validation/
│   │   └── employee_schema.py  # Pydantic input validation
│   └── utils/
│       ├── config.py           # Central paths/settings
│       └── logger.py           # Structured logging
├── notebooks/                  # 9 Jupyter notebooks + HTML previews
├── tests/
│   └── test_app.py             # 28+ unit tests
├── docs/
│   ├── data_relationships.md   # Dataset join analysis
│   └── day3_usage.md           # API + dashboard usage guide
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```

## Datasets

| Dataset | Rows | Columns | Purpose |
|---------|------|---------|---------|
| employee_attrition.csv | 1,470 | 35 | Attrition prediction target |
| hr_performance_engagement.csv | 2,845 | 28 | Engagement analytics |
| occupation_data.csv | 1,016 | 3 | Role master reference |
| essential_skills.csv | 18,200 | 15 | Skill requirements per role |
| software_skills.csv | 31,821 | 7 | Software requirements per role |

## ML Model

- **Algorithm:** XGBoost Classifier
- **Target:** Employee Attrition (Yes/No)
- **Features:** 44 (21 numeric + 3 engineered + 18 one-hot encoded)
- **Train/Test Split:** 80/20 stratified (1,176 / 294 samples)
- **Evaluation Metrics:**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.798 |
| F1 Score | 0.475 |
| Precision | 0.576 |
| Recall | 0.404 |

## Setup

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
cd enterprise_hr_ai
pip install -r requirements.txt
```

### Running

**FastAPI Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
```

**Streamlit Dashboard:**
```bash
streamlit run app/dashboard.py --server.port 8501
# Dashboard: http://localhost:8501
```

**Tests:**
```bash
python -m pytest tests/ -v
```

**Docker:**
```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## API Endpoints

### Prediction
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single employee attrition prediction |
| POST | `/predict/batch` | Batch prediction (max 100) |

### Dashboard
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard/summary` | KPI: total, high-risk, engagement |
| GET | `/dashboard/attrition-by-department` | Attrition rates by dept |
| GET | `/dashboard/risk-distribution` | Stay vs Leave counts |
| GET | `/dashboard/engagement-by-department` | Avg engagement by dept |
| GET | `/dashboard/low-engagement` | Lowest engagement employees |
| GET | `/dashboard/skill-gaps` | Org-wide skill gaps |
| GET | `/dashboard/recommendations` | Upskilling recommendations |

### Employee
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/employees/{id}` | Full intelligence record |

### Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/monitoring/health` | Model file + prediction health |
| GET | `/monitoring/drift` | Prediction distribution drift |
| GET | `/monitoring/feature-drift` | Input feature drift vs training |
| GET | `/monitoring/model-performance` | Live precision/recall/F1 |
| GET | `/monitoring/retrain-check` | Automated retrain rules |
| POST | `/monitoring/log-outcome` | Log actual attrition outcome |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health + model status |
| GET | `/model/info` | Model metadata + features |
| GET | `/options` | Valid categorical values |

## Monitoring & Retraining

**Data Drift (Task 25):** Compares production input distributions (age, income, years at company) against training data using z-score analysis.

**Model Performance (Task 26):** Recomputes precision/recall/F1 from actual attrition outcomes logged via `/monitoring/log-outcome`.

**Retraining Rules (Task 27):** Automated triggers when:
- Feature drift z-score > 3.0
- Live F1 drops > 0.2 below training baseline
- (Cooldown: max once per 30 days)

## Dashboard Pages

1. **Dashboard Overview** — KPI cards, department charts
2. **Attrition Prediction** — Interactive prediction form
3. **Engagement Analytics** — Department engagement breakdown
4. **Skill Gaps** — Organization-wide gap analysis
5. **Recommendations** — AI upskilling suggestions
6. **Model Info** — Metrics, features, metadata
7. **Monitoring** — Drift detection, prediction logs

## Key Design Decisions

- **Model choice:** XGBoost selected for highest F1 (0.475) and recall (0.404) — recall prioritized because missing high-risk employees is costly
- **Feature engineering:** 3 derived features (income_per_year, years_since_promotion_ratio, satisfaction_score) with business rationale
- **Leakage prevention:** EmployeeNumber, MonthlyRate, DailyRate, HourlyRate dropped before training
- **Singleton model loading:** Model loaded once at startup, cached in memory
- **Backward compatibility:** `model_utils.py` re-exports from new `app.ml` modules

## License

Academic project — Enterprise HR AI Platform.

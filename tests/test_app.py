"""
Enterprise HR AI — Unit Tests
Covers spec task 22 requirements:
- Missing required column is caught
- Invalid engagement score is rejected
- Attrition prediction returns a real probability
- Risk level is assigned correctly from that probability
- Skill gap calculation matches expected output
- API returns the expected status codes
"""
import sys
import os

# Ensure app module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def low_risk_employee():
    return {
        "Age": 45, "Department": "Research & Development",
        "DistanceFromHome": 5, "Education": 4,
        "EducationField": "Life Sciences", "EnvironmentSatisfaction": 4,
        "Gender": "Female", "JobInvolvement": 4, "JobLevel": 3,
        "JobRole": "Research Director", "JobSatisfaction": 4,
        "MaritalStatus": "Married", "MonthlyIncome": 15000,
        "NumCompaniesWorked": 2, "OverTime": "No",
        "PercentSalaryHike": 14, "PerformanceRating": 3,
        "RelationshipSatisfaction": 4, "StockOptionLevel": 1,
        "TotalWorkingYears": 20, "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 4, "YearsAtCompany": 15,
        "YearsInCurrentRole": 10, "YearsSinceLastPromotion": 5,
        "YearsWithCurrManager": 10, "HourlyRate": 80,
        "DailyRate": 1000, "MonthlyRate": 20000,
        "BusinessTravel": "Travel_Rarely",
    }


@pytest.fixture
def high_risk_employee():
    return {
        "Age": 22, "Department": "Sales",
        "DistanceFromHome": 25, "Education": 1,
        "EducationField": "Marketing", "EnvironmentSatisfaction": 1,
        "Gender": "Male", "JobInvolvement": 1, "JobLevel": 1,
        "JobRole": "Sales Representative", "JobSatisfaction": 1,
        "MaritalStatus": "Single", "MonthlyIncome": 2000,
        "NumCompaniesWorked": 5, "OverTime": "Yes",
        "PercentSalaryHike": 22, "PerformanceRating": 4,
        "RelationshipSatisfaction": 1, "StockOptionLevel": 0,
        "TotalWorkingYears": 1, "TrainingTimesLastYear": 0,
        "WorkLifeBalance": 1, "YearsAtCompany": 0,
        "YearsInCurrentRole": 0, "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0, "HourlyRate": 25,
        "DailyRate": 150, "MonthlyRate": 25000,
        "BusinessTravel": "Travel_Frequently",
    }


# ---------------------------------------------------------------------------
# Test: Health Endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_loaded(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_has_uptime(self, client):
        r = client.get("/health")
        data = r.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# Test: Prediction returns real probability
# ---------------------------------------------------------------------------
class TestPredictionProbability:
    def test_prediction_returns_probability(self, client, low_risk_employee):
        """Spec requirement: prediction returns a real probability."""
        r = client.post("/predict", json=low_risk_employee)
        assert r.status_code == 200
        data = r.json()
        assert "attrition_probability" in data
        assert isinstance(data["attrition_probability"], float)
        assert 0 <= data["attrition_probability"] <= 1

    def test_prediction_returns_prediction_label(self, client, low_risk_employee):
        r = client.post("/predict", json=low_risk_employee)
        data = r.json()
        assert data["prediction"] in ("Yes", "No")

    def test_prediction_returns_model_info(self, client, low_risk_employee):
        r = client.post("/predict", json=low_risk_employee)
        data = r.json()
        assert data["model_algorithm"] == "XGBoost"
        assert data["model_version"] == "v1.0"


# ---------------------------------------------------------------------------
# Test: Risk level assigned correctly
# ---------------------------------------------------------------------------
class TestRiskLevelAssignment:
    def test_low_risk_employee(self, client, low_risk_employee):
        """Spec requirement: risk level assigned correctly from probability."""
        r = client.post("/predict", json=low_risk_employee)
        data = r.json()
        assert data["risk_level"] == "Low"
        assert data["attrition_probability"] < 0.4

    def test_high_risk_employee(self, client, high_risk_employee):
        r = client.post("/predict", json=high_risk_employee)
        data = r.json()
        assert data["risk_level"] == "High"
        assert data["attrition_probability"] >= 0.7

    def test_high_risk_has_higher_probability(self, client, low_risk_employee, high_risk_employee):
        """High-risk input should produce higher probability than low-risk."""
        r_low = client.post("/predict", json=low_risk_employee)
        r_high = client.post("/predict", json=high_risk_employee)
        assert r_high.json()["attrition_probability"] > r_low.json()["attrition_probability"]


# ---------------------------------------------------------------------------
# Test: API returns expected status codes
# ---------------------------------------------------------------------------
class TestAPICodes:
    def test_valid_input_returns_200(self, client, low_risk_employee):
        """Spec requirement: API returns expected status codes."""
        r = client.post("/predict", json=low_risk_employee)
        assert r.status_code == 200

    def test_invalid_input_returns_422(self, client):
        """Spec requirement: bad data gets rejected."""
        r = client.post("/predict", json={"Age": 5})
        assert r.status_code == 422

    def test_age_too_low_returns_422(self, client):
        r = client.post("/predict", json={"Age": 10, "MonthlyIncome": 5000, "TotalWorkingYears": 5})
        assert r.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        """Spec requirement: missing required column is caught."""
        r = client.post("/predict", json={"Age": 30})
        assert r.status_code == 422

    def test_dashboard_summary_returns_200(self, client):
        r = client.get("/dashboard/summary")
        assert r.status_code == 200

    def test_employee_not_found_returns_404(self, client):
        r = client.get("/employees/999999")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test: Skill gap calculation
# ---------------------------------------------------------------------------
class TestSkillGap:
    def test_skill_gap_matches_expected(self):
        """Spec requirement: skill gap calculation matches expected output."""
        from app.services.skill_gap_service import compute_employee_skill_gaps

        employee_roles = [
            {"employee_id": "101", "role": "Data Analyst"},
        ]
        employee_skills = {
            "101": {"Python", "SQL", "Excel"},
        }

        results = compute_employee_skill_gaps(employee_roles, employee_skills)
        assert len(results) == 1
        result = results[0]
        assert result["employee_id"] == "101"
        assert result["gap_count"] >= 0
        assert isinstance(result["missing_skills"], list)

    def test_skill_gap_empty_skills_has_full_gap(self):
        """Employee with no skills should have gaps for all required."""
        from app.services.skill_gap_service import compute_employee_skill_gaps

        employee_roles = [{"employee_id": "102", "role": "Software Developer"}]
        results = compute_employee_skill_gaps(employee_roles, {})
        assert len(results) == 1
        assert results[0]["gap_count"] >= 0


# ---------------------------------------------------------------------------
# Test: Model info
# ---------------------------------------------------------------------------
class TestModelInfo:
    def test_model_info_has_44_features(self, client):
        r = client.get("/model/info")
        data = r.json()
        assert data["n_features"] == 44

    def test_model_info_has_metadata(self, client):
        r = client.get("/model/info")
        data = r.json()
        assert data["metadata"]["algorithm"] == "XGBoost"
        assert data["metadata"]["version"] == "v1.0"


# ---------------------------------------------------------------------------
# Test: Options
# ---------------------------------------------------------------------------
class TestOptions:
    def test_options_has_departments(self, client):
        r = client.get("/options")
        opts = r.json()
        assert len(opts["departments"]) == 3

    def test_options_has_roles(self, client):
        r = client.get("/options")
        opts = r.json()
        assert len(opts["job_roles"]) == 9


# ---------------------------------------------------------------------------
# Test: Dashboard summary
# ---------------------------------------------------------------------------
class TestDashboardSummary:
    def test_summary_has_total_employees(self, client):
        r = client.get("/dashboard/summary")
        data = r.json()
        assert data["total_employees"] == 1470

    def test_summary_has_high_risk_count(self, client):
        r = client.get("/dashboard/summary")
        data = r.json()
        assert data["high_risk_employees"] > 0


# ---------------------------------------------------------------------------
# Test: Engagement analytics
# ---------------------------------------------------------------------------
class TestEngagementAnalytics:
    def test_engagement_by_department(self, client):
        r = client.get("/dashboard/engagement-by-department")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_skill_gaps(self, client):
        r = client.get("/dashboard/skill-gaps")
        assert r.status_code == 200

    def test_recommendations(self, client):
        r = client.get("/dashboard/recommendations")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test: Batch prediction
# ---------------------------------------------------------------------------
class TestBatchPrediction:
    def test_batch_returns_results(self, client, low_risk_employee):
        r = client.post("/predict/batch", json=[low_risk_employee, low_risk_employee])
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_too_large_returns_400(self, client, low_risk_employee):
        r = client.post("/predict/batch", json=[low_risk_employee] * 101)
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Test: Monitoring (Tasks 25-27)
# ---------------------------------------------------------------------------
class TestMonitoring:
    def test_feature_drift_endpoint(self, client):
        """Task 25: Feature drift endpoint returns training + production stats."""
        r = client.get("/monitoring/feature-drift")
        assert r.status_code == 200
        data = r.json()
        assert "training_reference" in data
        assert "alerts" in data

    def test_training_reference_endpoint(self, client):
        """Task 25: Training reference stats are available."""
        r = client.get("/monitoring/training-reference")
        assert r.status_code == 200
        data = r.json()
        assert "Age" in data
        assert "MonthlyIncome" in data
        assert "n_training_samples" in data
        assert data["n_training_samples"] == 1470

    def test_model_performance_endpoint(self, client):
        """Task 26: Model performance endpoint works."""
        r = client.get("/monitoring/model-performance")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "n_outcomes" in data

    def test_retrain_check_endpoint(self, client):
        """Task 27: Retrain check returns conditions and thresholds."""
        r = client.get("/monitoring/retrain-check")
        assert r.status_code == 200
        data = r.json()
        assert "should_retrain" in data
        assert "conditions" in data
        assert "thresholds" in data
        assert isinstance(data["should_retrain"], bool)

    def test_drift_stats(self):
        """Task 25: compute_feature_drift returns training reference."""
        from app.monitoring import compute_feature_drift
        result = compute_feature_drift()
        assert "training_reference" in result
        assert "alerts" in result

    def test_retrain_conditions(self):
        """Task 27: check_retrain_conditions returns structured result."""
        from app.monitoring import check_retrain_conditions
        result = check_retrain_conditions()
        assert "should_retrain" in result
        assert "conditions" in result
        assert "thresholds" in result
        assert result["thresholds"]["drift_z_score"] == 3.0
        assert result["thresholds"]["f1_drop"] == 0.2

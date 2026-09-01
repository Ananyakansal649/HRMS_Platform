"""
Enterprise HR AI — Streamlit Dashboard
AI Workforce Intelligence Platform (spec task 23).
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from app.ml.model_loader import ModelLoader
from app.ml.predictor import predict_employee as _predict
from app.monitoring import log_prediction
from app.utils.config import DATA_RAW

st.set_page_config(
    page_title="Enterprise HR AI — Workforce Intelligence",
    page_icon="🏢",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return ModelLoader().load()


model, metadata, feature_names = load_model()


# ---------------------------------------------------------------------------
# Helper: safe predict
# ---------------------------------------------------------------------------
def safe_predict(data: dict) -> dict:
    """Run prediction with error handling."""
    try:
        result = _predict(data)
        log_prediction(
            input_data=data,
            prediction=result["prediction"],
            probability=result["attrition_probability"],
            risk_level=result["risk_level"],
            model_version=result["model_version"],
            endpoint="streamlit",
        )
        return result
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏢 Enterprise HR AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard Overview", "👤 Attrition Prediction", "📊 Engagement Analytics",
     "⚠️ Skill Gaps", "🎯 Recommendations", "🤖 Model Info", "📈 Monitoring"],
)


# ---------------------------------------------------------------------------
# PAGE: Dashboard Overview (KPI cards + charts)
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard Overview":
    st.title("AI Workforce Intelligence Platform")

    # KPI cards
    try:
        df_attr = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
        total = len(df_attr)
        high_risk = int((df_attr["Attrition"] == "Yes").sum())
    except Exception:
        total, high_risk = 0, 0

    try:
        df_eng = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
        eng_col = [c for c in df_eng.columns if "engagement" in c.lower()]
        avg_eng = round(float(df_eng[eng_col[0]].mean()), 1) if eng_col else 0
    except Exception:
        avg_eng = 0

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Employees", f"{total:,}")
    col2.metric("⚠️ High Risk", f"{high_risk}", f"{round(high_risk/total*100, 1)}%")
    col3.metric("📊 Avg Engagement", f"{avg_eng}%")

    st.divider()

    # Attrition by department
    st.subheader("Attrition Risk by Department")
    dept_attr = df_attr.groupby("Department")["Attrition"].apply(
        lambda x: (x == "Yes").sum() / len(x) * 100
    ).round(1).sort_values(ascending=True)
    st.bar_chart(dept_attr)

    # Risk distribution
    st.subheader("Risk Distribution")
    risk_counts = df_attr["Attrition"].value_counts()
    risk_df = pd.DataFrame({
        "Category": ["Stay (No)", "Leave (Yes)"],
        "Count": [risk_counts.get("No", 0), risk_counts.get("Yes", 0)],
    })
    st.bar_chart(risk_df.set_index("Category"))


# ---------------------------------------------------------------------------
# PAGE: Attrition Prediction
# ---------------------------------------------------------------------------
elif page == "👤 Attrition Prediction":
    st.title("👤 Attrition Prediction")
    st.markdown("Enter employee details to predict attrition risk using the trained XGBoost model.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Research Scientist", "Laboratory Technician", "Sales Executive",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ])
        education = st.slider("Education Level", 1, 5, 3)
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources",
        ])
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 6000, step=500)
        total_working_years = st.number_input("Total Working Years", 0, 40, 8)
        years_at_company = st.number_input("Years at Company", 0, 40, 5)
        overtime = st.selectbox("OverTime", ["No", "Yes"])

    with col2:
        distance = st.number_input("Distance from Home (km)", 0, 30, 5)
        env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
        job_inv = st.slider("Job Involvement", 1, 4, 3)
        job_level = st.slider("Job Level", 1, 5, 2)
        job_sat = st.slider("Job Satisfaction", 1, 4, 3)
        work_life = st.slider("Work-Life Balance", 1, 4, 3)
        perf_rating = st.selectbox("Performance Rating", [3, 4])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    if st.button("🔍 Predict Attrition Risk", type="primary"):
        input_data = {
            "Age": age, "Department": department, "JobRole": job_role,
            "Education": education, "EducationField": education_field,
            "MonthlyIncome": monthly_income, "TotalWorkingYears": total_working_years,
            "YearsAtCompany": years_at_company, "OverTime": overtime,
            "DistanceFromHome": distance, "EnvironmentSatisfaction": env_sat,
            "JobInvolvement": job_inv, "JobLevel": job_level,
            "JobSatisfaction": job_sat, "WorkLifeBalance": work_life,
            "PerformanceRating": perf_rating, "Gender": gender,
            "MaritalStatus": marital, "BusinessTravel": business_travel,
            "NumCompaniesWorked": 3, "PercentSalaryHike": 14,
            "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
            "TrainingTimesLastYear": 2, "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
            "HourlyRate": 60, "DailyRate": 800, "MonthlyRate": 15000,
        }

        result = safe_predict(input_data)
        if result:
            st.divider()
            prob = result["attrition_probability"]

            # Color-coded risk display
            if result["risk_level"] == "High":
                st.error(f"🚨 **HIGH RISK** — Attrition probability: {prob:.1%}")
            elif result["risk_level"] == "Medium":
                st.warning(f"⚠️ **MEDIUM RISK** — Attrition probability: {prob:.1%}")
            else:
                st.success(f"✅ **LOW RISK** — Attrition probability: {prob:.1%}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", result["prediction"])
            col2.metric("Probability", f"{prob:.4f}")
            col3.metric("Risk Level", result["risk_level"])


# ---------------------------------------------------------------------------
# PAGE: Engagement Analytics
# ---------------------------------------------------------------------------
elif page == "📊 Engagement Analytics":
    st.title("📊 Engagement Analytics")

    try:
        df_eng = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
        eng_col = [c for c in df_eng.columns if "engagement" in c.lower()]
        dept_col = [c for c in df_eng.columns if "department" in c.lower()]

        if eng_col:
            st.subheader("Engagement by Department")
            if dept_col:
                dept_eng = df_eng.groupby(dept_col[0])[eng_col[0]].mean().round(2).sort_values(ascending=True)
                st.bar_chart(dept_eng)
            else:
                st.info("No department column found in engagement data")

            st.subheader("Engagement Distribution")
            eng_dist = df_eng[eng_col[0]].value_counts().sort_index()
            st.bar_chart(eng_dist)
        else:
            st.warning("No engagement score column found in the data")
    except Exception as e:
        st.error(f"Error loading engagement data: {e}")


# ---------------------------------------------------------------------------
# PAGE: Skill Gaps
# ---------------------------------------------------------------------------
elif page == "⚠️ Skill Gaps":
    st.title("⚠️ Critical Organisation Skill Gaps")

    from app.services.skill_gap_service import get_required_skills_by_role, get_organization_skill_gaps

    st.info("Note: The raw datasets do not contain per-employee current skills. "
            "Skill gaps are computed based on role requirements vs. assumed zero current skills. "
            "This will be addressed with real employee skill data in a future iteration.")

    role_skills = get_required_skills_by_role()
    st.metric("Roles with defined skills", len(role_skills))

    if role_skills:
        st.subheader("View Required Skills by Role")
        sorted_roles = sorted(role_skills.keys())
        selected_role = st.selectbox("Select a role", sorted_roles, index=0)
        if selected_role:
            skills = role_skills[selected_role]
            st.markdown(f"**{selected_role}** — {len(skills)} required skills")
            st.write(', '.join(sorted(skills)))

    st.divider()
    st.subheader("Top Organisation-Wide Skill Gaps")

    # Synthetic example for demonstration
    synthetic_roles = [{"employee_id": str(i), "role": list(role_skills.keys())[i % len(role_skills)]}
                       for i in range(100)]
    org_gaps = get_organization_skill_gaps(synthetic_roles, {})

    if org_gaps:
        gap_df = pd.DataFrame(org_gaps[:15])
        st.dataframe(gap_df, use_container_width=True)
    else:
        st.info("No skill gap data available")


# ---------------------------------------------------------------------------
# PAGE: Recommendations
# ---------------------------------------------------------------------------
elif page == "🎯 Recommendations":
    st.title("🎯 AI Upskilling Recommendations")

    from app.services.recommendation_service import generate_recommendations
    from app.services.skill_gap_service import get_required_skills_by_role

    role_skills = get_required_skills_by_role()
    df_attr = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))

    n_employees = st.slider("Number of employees to analyze", 5, 50, 10)
    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df_attr.head(n_employees).iterrows()
    ]

    recs = generate_recommendations(employee_roles, {})

    if recs:
        for rec in recs:
            if rec["recommendations"]:
                with st.expander(f"Employee {rec['employee_id']} — {rec['role']} ({rec['gap_count']} gaps)"):
                    for rec_text in rec["recommendations"]:
                        st.write(f"• {rec_text}")


# ---------------------------------------------------------------------------
# PAGE: Model Info
# ---------------------------------------------------------------------------
elif page == "🤖 Model Info":
    st.title("🤖 Model Information")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Algorithm", metadata.get("algorithm", "Unknown"))
        st.metric("Version", metadata.get("version", "Unknown"))
        st.metric("Training Date", metadata.get("training_date", "Unknown"))
        st.metric("Features", len(feature_names))

    with col2:
        metrics = metadata.get("metrics", {})
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")

    with st.expander("Feature Names"):
        st.write(feature_names)

    with st.expander("Training Data Details"):
        st.json(metadata.get("preprocessing", {}))
        st.json(metadata.get("class_distribution", {}))


# ---------------------------------------------------------------------------
# PAGE: Monitoring
# ---------------------------------------------------------------------------
elif page == "📈 Monitoring":
    st.title("📈 Prediction Monitoring")

    try:
        from app.monitoring import load_predictions, compute_drift_stats, get_model_health

        # Model health
        st.subheader("Model Health")
        health = get_model_health()
        col1, col2, col3 = st.columns(3)
        col1.metric("Model File", "✅ Exists" if health.get("model_file_exists") else "❌ Missing")
        col2.metric("Recent Predictions", health.get("recent_prediction_count", 0))
        col3.metric("Version", health.get("model_version", "Unknown"))

        # Drift stats
        st.subheader("Drift Detection")
        drift = compute_drift_stats()
        st.json(drift)

        # Recent predictions
        st.subheader("Recent Predictions")
        preds = load_predictions(limit=20)
        if preds:
            pred_df = pd.DataFrame(preds)
            st.dataframe(pred_df[["timestamp", "prediction", "attrition_probability",
                                   "risk_level", "model_version"]], use_container_width=True)
        else:
            st.info("No predictions logged yet. Make a prediction to see it here.")
    except Exception as e:
        st.error(f"Monitoring error: {e}")

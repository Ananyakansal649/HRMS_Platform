"""
Enterprise HR AI — Streamlit Dashboard
AI Workforce Intelligence Platform (spec task 23).

Professional enterprise-grade UI with modern SaaS design.
Fully theme-aware: works in Light, Dark, and System modes.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Import path fix — required for Streamlit Community Cloud
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.model_loader import ModelLoader
from app.ml.predictor import predict_employee as _predict
from app.monitoring import log_prediction
from app.utils.config import DATA_RAW

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise HR AI — Workforce Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional CSS — Enterprise SaaS theme (fully theme-aware)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.65;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text-color);
    line-height: 1.2;
}
.kpi-delta {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 4px;
}
.kpi-delta.negative { color: #ef4444; }
.kpi-delta.positive { color: #10b981; }
.kpi-icon {
    font-size: 1.4rem;
    margin-bottom: 6px;
}

/* ── Section Cards ─────────────────────────────────────────────────────── */
.section-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.section-card h3 {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-color);
    margin: 0 0 12px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
}

/* ── Risk Cards ────────────────────────────────────────────────────────── */
.risk-card {
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin-bottom: 20px;
}
.risk-low {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
}
.risk-medium {
    background: #fffbeb;
    border: 1px solid #fde68a;
}
.risk-high {
    background: #fef2f2;
    border: 1px solid #fecaca;
}
.risk-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.risk-low .risk-label { color: #059669; }
.risk-medium .risk-label { color: #d97706; }
.risk-high .risk-label { color: #dc2626; }
.risk-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-color);
}
.risk-detail {
    font-size: 0.85rem;
    color: var(--text-color);
    opacity: 0.6;
    margin-top: 4px;
}

/* ── Metric Grid ───────────────────────────────────────────────────────── */
.metric-item {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin-bottom: 4px;
}
.metric-item .label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.metric-item .value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-color);
}

/* ── Recommendation Cards ──────────────────────────────────────────────── */
.rec-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}

/* ── Model / Status Cards ──────────────────────────────────────────────── */
.model-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 4px;
}
.model-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.model-card .value {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-color);
}

/* ── Status Dot ────────────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.status-dot.green { background: #10b981; }
.status-dot.yellow { background: #f59e0b; }
.status-dot.red { background: #ef4444; }

/* ── Expander ──────────────────────────────────────────────────────────── */
.streamlit-expanderContent {
    background: var(--secondary-background-color) !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--border-color);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Load model once
# ===========================================================================
@st.cache_resource
def load_model():
    return ModelLoader().load()


model, metadata, feature_names = load_model()


# ===========================================================================
# Helper: safe predict
# ===========================================================================
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


# ===========================================================================
# Helper: render KPI card
# ===========================================================================
def kpi_card(icon: str, label: str, value, delta: str = "", delta_positive: bool = True):
    """Render a themed KPI card."""
    delta_cls = "positive" if delta_positive else "negative"
    delta_html = f'<div class="kpi-delta {delta_cls}">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Sidebar navigation
# ===========================================================================
with st.sidebar:
    st.markdown("## 🏢 Enterprise HR AI")
    st.caption("AI Workforce Intelligence")

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard Overview",
            "👤 Attrition Prediction",
            "📊 Engagement Analytics",
            "⚠️ Skill Gaps",
            "🎯 Recommendations",
            "🤖 Model Info",
            "📈 Monitoring",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Model version info — uses st.markdown for theme-adaptive text
    model_algo = metadata.get("algorithm", "XGBoost")
    model_ver = metadata.get("version", "v1.0")
    st.markdown(f"""
    <div style="font-size: 0.72rem; line-height: 1.7; opacity: 0.75;">
        <span class="status-dot green"></span> System Online<br>
        Model: <strong>{model_algo} {model_ver}</strong><br>
        Features: <strong>{len(feature_names)}</strong>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: Dashboard Overview
# ===========================================================================
if page == "🏠 Dashboard Overview":
    st.title("AI Workforce Intelligence Platform")
    st.caption("AI-driven workforce analytics and attrition prediction for enterprise HR.")

    # ── Load data ──
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

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("👥", "Total Employees", f"{total:,}")
    with c2:
        kpi_card("⚠️", "High Risk Employees", f"{high_risk:,}",
                 delta=f"{round(high_risk/total*100, 1)}% attrition rate" if total else "",
                 delta_positive=False)
    with c3:
        kpi_card("📊", "Avg Engagement", f"{avg_eng}%")
    with c4:
        retention = round((1 - high_risk / total) * 100, 1) if total else 0
        kpi_card("✅", "Retention Rate", f"{retention}%",
                 delta=f"{total - high_risk:,} employees retained",
                 delta_positive=True)

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    # ── Charts Row ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="section-card">
            <h3>Attrition Risk by Department</h3>
        </div>
        """, unsafe_allow_html=True)
        dept_attr = df_attr.groupby("Department")["Attrition"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).round(1).sort_values(ascending=True)
        st.bar_chart(dept_attr)

    with col_right:
        st.markdown("""
        <div class="section-card">
            <h3>Risk Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        risk_counts = df_attr["Attrition"].value_counts()
        risk_df = pd.DataFrame({
            "Category": ["Stay (No)", "Leave (Yes)"],
            "Count": [risk_counts.get("No", 0), risk_counts.get("Yes", 0)],
        })
        st.bar_chart(risk_df.set_index("Category"))


# ===========================================================================
# PAGE: Attrition Prediction
# ===========================================================================
elif page == "👤 Attrition Prediction":
    st.title("Attrition Prediction")
    st.caption("Enter employee details to predict attrition risk using the trained XGBoost model.")

    # ── Input Sections ──
    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="section-card">
            <h3>👤 Employee Profile</h3>
        </div>
        """, unsafe_allow_html=True)
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
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

        st.markdown("""
        <div class="section-card">
            <h3>🏢 Work Environment</h3>
        </div>
        """, unsafe_allow_html=True)
        distance = st.number_input("Distance from Home (km)", 0, 30, 5)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    with right:
        st.markdown("""
        <div class="section-card">
            <h3>💰 Career & Compensation</h3>
        </div>
        """, unsafe_allow_html=True)
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 6000, step=500)
        total_working_years = st.number_input("Total Working Years", 0, 40, 8)
        years_at_company = st.number_input("Years at Company", 0, 40, 5)
        job_level = st.slider("Job Level", 1, 5, 2)

        st.markdown("""
        <div class="section-card">
            <h3>⭐ Satisfaction & Performance</h3>
        </div>
        """, unsafe_allow_html=True)
        env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
        job_inv = st.slider("Job Involvement", 1, 4, 3)
        job_sat = st.slider("Job Satisfaction", 1, 4, 3)
        work_life = st.slider("Work-Life Balance", 1, 4, 3)
        perf_rating = st.selectbox("Performance Rating", [3, 4])

    # ── Predict Button ──
    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Predict Attrition Risk", type="primary", use_container_width=True)

    if predict_clicked:
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
            prob = result["attrition_probability"]
            risk = result["risk_level"]

            # ── Risk Result Card ──
            if risk == "High":
                risk_cls = "risk-high"
                risk_emoji = "🚨"
            elif risk == "Medium":
                risk_cls = "risk-medium"
                risk_emoji = "⚠️"
            else:
                risk_cls = "risk-low"
                risk_emoji = "✅"

            st.markdown(f"""
            <div class="risk-card {risk_cls}">
                <div class="risk-label">{risk_emoji} {risk} Risk</div>
                <div class="risk-value">{prob:.1%}</div>
                <div class="risk-detail">Attrition Probability</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="label">Prediction</div>
                    <div class="value">{result['prediction']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="label">Probability</div>
                    <div class="value">{prob:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="label">Model Version</div>
                    <div class="value">{result['model_version']}</div>
                </div>
                """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: Engagement Analytics
# ===========================================================================
elif page == "📊 Engagement Analytics":
    st.title("Engagement Analytics")
    st.caption("Workforce engagement scores, departmental comparisons, and distribution analysis.")

    try:
        df_eng = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
        eng_col = [c for c in df_eng.columns if "engagement" in c.lower()]
        dept_col = [c for c in df_eng.columns if "department" in c.lower()]

        if eng_col:
            # ── KPI Summary ──
            avg_score = round(float(df_eng[eng_col[0]].mean()), 2)
            min_score = round(float(df_eng[eng_col[0]].min()), 2)
            max_score = round(float(df_eng[eng_col[0]].max()), 2)
            total_records = len(df_eng)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("📊", "Avg Engagement", f"{avg_score}%")
            with c2:
                kpi_card("📉", "Min Score", f"{min_score}%")
            with c3:
                kpi_card("📈", "Max Score", f"{max_score}%")
            with c4:
                kpi_card("📋", "Total Records", f"{total_records:,}")

            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("""
                <div class="section-card">
                    <h3>Engagement by Department</h3>
                </div>
                """, unsafe_allow_html=True)
                if dept_col:
                    dept_eng = df_eng.groupby(dept_col[0])[eng_col[0]].mean().round(2).sort_values(ascending=True)
                    st.bar_chart(dept_eng)
                else:
                    st.info("No department column found in engagement data")

            with col_right:
                st.markdown("""
                <div class="section-card">
                    <h3>Engagement Distribution</h3>
                </div>
                """, unsafe_allow_html=True)
                eng_dist = df_eng[eng_col[0]].value_counts().sort_index()
                st.bar_chart(eng_dist)
        else:
            st.warning("No engagement score column found in the data")
    except Exception as e:
        st.error(f"Error loading engagement data: {e}")


# ===========================================================================
# PAGE: Skill Gaps
# ===========================================================================
elif page == "⚠️ Skill Gaps":
    from app.services.skill_gap_service import get_required_skills_by_role, get_organization_skill_gaps

    st.title("Critical Organisation Skill Gaps")
    st.caption("Identify skill gaps across roles and prioritise upskilling investments.")

    st.info(
        "Note: The raw datasets do not contain per-employee current skills. "
        "Skill gaps are computed based on role requirements vs. assumed zero current skills. "
        "This will be addressed with real employee skill data in a future iteration."
    )

    role_skills = get_required_skills_by_role()

    # ── KPI ──
    c1, c2 = st.columns(2)
    with c1:
        kpi_card("📋", "Roles with Defined Skills", f"{len(role_skills):,}")
    with c2:
        all_skills = set()
        for skills in role_skills.values():
            all_skills.update(skills)
        kpi_card("🔧", "Unique Skills Required", f"{len(all_skills):,}")

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    if role_skills:
        with st.expander("🔍 View Required Skills by Role", expanded=False):
            sorted_roles = sorted(role_skills.keys())
            selected_role = st.selectbox("Select a role", sorted_roles, index=0, label_visibility="collapsed")
            if selected_role:
                skills = role_skills[selected_role]
                st.markdown(f"**{selected_role}** — {len(skills)} required skills")
                st.write(", ".join(sorted(skills)))

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # ── Organisation-wide skill gaps ──
    st.markdown("""
    <div class="section-card">
        <h3>Top Organisation-Wide Skill Gaps</h3>
    </div>
    """, unsafe_allow_html=True)

    synthetic_roles = [
        {"employee_id": str(i), "role": list(role_skills.keys())[i % len(role_skills)]}
        for i in range(100)
    ]
    org_gaps = get_organization_skill_gaps(synthetic_roles, {})

    if org_gaps:
        gap_df = pd.DataFrame(org_gaps[:15])
        st.dataframe(gap_df, use_container_width=True)
    else:
        st.info("No skill gap data available")


# ===========================================================================
# PAGE: Recommendations
# ===========================================================================
elif page == "🎯 Recommendations":
    from app.services.recommendation_service import generate_recommendations
    from app.services.skill_gap_service import get_required_skills_by_role

    st.title("AI Upskilling Recommendations")
    st.caption("Personalised training recommendations based on each employee's role skill gaps.")

    role_skills = get_required_skills_by_role()
    df_attr = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))

    n_employees = st.slider("Number of employees to analyze", 5, 50, 10)
    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df_attr.head(n_employees).iterrows()
    ]

    recs = generate_recommendations(employee_roles, {})

    if recs:
        # Summary KPI
        total_gaps = sum(r["gap_count"] for r in recs)
        avg_gaps = round(total_gaps / len(recs), 1) if recs else 0
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("👥", "Employees Analyzed", f"{len(recs)}")
        with c2:
            kpi_card("⚠️", "Total Skill Gaps", f"{total_gaps:,}")
        with c3:
            kpi_card("📊", "Avg Gaps per Employee", f"{avg_gaps}")

        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

        for rec in recs:
            if rec["recommendations"]:
                with st.expander(
                    f"👤 Employee {rec['employee_id']}  —  {rec['role']}  ({rec['gap_count']} gaps)"
                ):
                    for rec_text in rec["recommendations"]:
                        st.markdown(f"""
                        <div class="rec-card">
                            <div style="font-size: 0.85rem; color: var(--text-color);">
                                {rec_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: Model Info
# ===========================================================================
elif page == "🤖 Model Info":
    st.title("Model Information")
    st.caption("Details of the trained attrition prediction model.")

    metrics = metadata.get("metrics", {})

    # ── Model Summary Cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="model-card">
            <div class="label">Algorithm</div>
            <div class="value">{metadata.get("algorithm", "Unknown")}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="model-card">
            <div class="label">Version</div>
            <div class="value">{metadata.get("version", "Unknown")}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="model-card">
            <div class="label">Training Date</div>
            <div class="value">{metadata.get("training_date", "Unknown")}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="model-card">
            <div class="label">Features</div>
            <div class="value">{len(feature_names)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # ── Performance Metrics ──
    st.markdown("""
    <div class="section-card">
        <h3>Performance Metrics</h3>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-item">
            <div class="label">ROC-AUC</div>
            <div class="value">{metrics.get('roc_auc', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="label">F1 Score</div>
            <div class="value">{metrics.get('f1_score', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-item">
            <div class="label">Precision</div>
            <div class="value">{metrics.get('precision', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-item">
            <div class="label">Recall</div>
            <div class="value">{metrics.get('recall', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # ── Details ──
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Feature Names"):
            st.write(feature_names)
    with col_b:
        with st.expander("Training Data Details"):
            st.json(metadata.get("preprocessing", {}))
            st.json(metadata.get("class_distribution", {}))


# ===========================================================================
# PAGE: Monitoring
# ===========================================================================
elif page == "📈 Monitoring":
    from app.monitoring import load_predictions, compute_drift_stats, get_model_health

    st.title("Prediction Monitoring")
    st.caption("Live model health, drift detection, and recent prediction activity.")

    try:
        health = get_model_health()
        drift = compute_drift_stats()

        # ── Model Health KPIs ──
        model_ok = health.get("model_file_exists", False)
        pred_count = health.get("recent_prediction_count", 0)
        model_ver = health.get("model_version", "Unknown")

        c1, c2, c3 = st.columns(3)
        with c1:
            status = "🟢 Healthy" if model_ok else "🔴 Missing"
            kpi_card("🤖", "Model Status", status)
        with c2:
            kpi_card("📊", "Recent Predictions", f"{pred_count:,}")
        with c3:
            kpi_card("🏷️", "Model Version", model_ver)

        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("""
            <div class="section-card">
                <h3>Drift Detection</h3>
            </div>
            """, unsafe_allow_html=True)
            st.json(drift)

        with col_right:
            st.markdown("""
            <div class="section-card">
                <h3>Recent Predictions</h3>
            </div>
            """, unsafe_allow_html=True)
            preds = load_predictions(limit=20)
            if preds:
                pred_df = pd.DataFrame(preds)
                st.dataframe(
                    pred_df[["timestamp", "prediction", "attrition_probability",
                              "risk_level", "model_version"]],
                    use_container_width=True,
                )
            else:
                st.info("No predictions logged yet. Make a prediction to see it here.")

    except Exception as e:
        st.error(f"Monitoring error: {e}")

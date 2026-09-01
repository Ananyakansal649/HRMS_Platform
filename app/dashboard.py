"""
Enterprise HR AI — Streamlit Dashboard
AI Workforce Intelligence Platform (spec task 23).

Redesigned for a professional enterprise SaaS appearance.
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
# Professional CSS — Enterprise HR AI dashboard redesign
#
# Design principles:
#   1. ALL text uses var(--text-color) or opacity modifiers — never hardcoded
#   2. ALL backgrounds use var(--secondary-background-color) — theme-adaptive
#   3. Semantic accent colors (green/amber/red) are intentional and fixed
#   4. Typography via system font stack — no external loading needed
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Reset & Base ──────────────────────────────────────────────────────── */
.stApp { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

/* ── Hero / Page Header ────────────────────────────────────────────────── */
.dash-hero {
    padding: 4px 0 20px 0;
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 24px;
}
.dash-hero h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-color) !important;
    margin: 0 !important;
    padding: 0 !important;
    letter-spacing: -0.02em;
}
.dash-hero p {
    color: var(--text-color) !important;
    opacity: 0.55 !important;
    font-size: 0.88rem !important;
    margin: 4px 0 0 0 !important;
}

/* ── Accent Chips (tiny colored dots) ──────────────────────────────────── */
.chip { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.chip-blue   { background: #3b82f6; }
.chip-green  { background: #10b981; }
.chip-amber  { background: #f59e0b; }
.chip-red    { background: #ef4444; }

/* ── KPI Tiles ─────────────────────────────────────────────────────────── */
.kpi-tile {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 22px 20px 18px 20px;
    position: relative;
    overflow: hidden;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-tile::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    border-radius: 14px 0 0 14px;
}
.kpi-tile.accent-blue::before  { background: #3b82f6; }
.kpi-tile.accent-red::before   { background: #ef4444; }
.kpi-tile.accent-green::before { background: #10b981; }
.kpi-tile.accent-amber::before { background: #f59e0b; }
.kpi-tile .kpi-icon { font-size: 1.3rem; margin-bottom: 6px; }
.kpi-tile .kpi-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-color);
    opacity: 0.5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.kpi-tile .kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--text-color);
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.kpi-tile .kpi-sub {
    font-size: 0.75rem;
    color: var(--text-color);
    opacity: 0.5;
    margin-top: 6px;
    line-height: 1.3;
}
.kpi-tile .kpi-sub.neg { color: #ef4444; opacity: 0.85; }
.kpi-tile .kpi-sub.pos { color: #10b981; opacity: 0.85; }

/* ── Card System ───────────────────────────────────────────────────────── */
.dash-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.dash-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 12px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-color);
}
.dash-card-header h3 {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-color);
    margin: 0 !important;
    letter-spacing: -0.01em;
}
.dash-card-icon {
    font-size: 1rem;
}

/* ── Section Divider ───────────────────────────────────────────────────── */
.dash-divider {
    border: none;
    border-top: 1px solid var(--border-color);
    margin: 20px 0;
}

/* ── Risk Display ──────────────────────────────────────────────────────── */
.risk-banner {
    border-radius: 14px;
    padding: 32px;
    text-align: center;
    margin-bottom: 20px;
    position: relative;
}
.risk-banner.risk-high   { background: var(--secondary-background-color); border: 2px solid #dc2626; }
.risk-banner.risk-medium { background: var(--secondary-background-color); border: 2px solid #d97706; }
.risk-banner.risk-low    { background: var(--secondary-background-color); border: 2px solid #059669; }
.risk-banner .risk-status {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.risk-banner.risk-high   .risk-status { color: #dc2626; }
.risk-banner.risk-medium .risk-status { color: #d97706; }
.risk-banner.risk-low    .risk-status { color: #059669; }
.risk-banner .risk-pct {
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text-color);
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.risk-banner .risk-sub {
    font-size: 0.82rem;
    color: var(--text-color);
    opacity: 0.55;
    margin-top: 6px;
}

/* ── Metric Pill ───────────────────────────────────────────────────────── */
.metric-pill {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
}
.metric-pill .mp-label {
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--text-color);
    opacity: 0.45;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.metric-pill .mp-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-color);
}

/* ── Recommendation Item ───────────────────────────────────────────────── */
.rec-item {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: var(--text-color);
    line-height: 1.5;
}

/* ── Model / Info Card ─────────────────────────────────────────────────── */
.info-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
}
.info-card .ic-label {
    font-size: 0.6rem;
    font-weight: 700;
    color: var(--text-color);
    opacity: 0.45;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.info-card .ic-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-color);
    letter-spacing: -0.01em;
}

/* ── Section Title (inside cards) ──────────────────────────────────────── */
.sec-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-color);
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 14px;
}

/* ── Expander polish ───────────────────────────────────────────────────── */
.streamlit-expanderContent {
    background: var(--secondary-background-color) !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Sidebar status block ──────────────────────────────────────────────── */
.sb-status {
    font-size: 0.68rem;
    line-height: 1.8;
    color: var(--text-color);
    opacity: 0.65;
}
.sb-status strong { opacity: 1; }
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
# Helper: render a KPI tile
# ===========================================================================
def kpi(icon: str, label: str, value, sub: str = "", accent: str = "blue", sub_cls: str = ""):
    cls = f"kpi-tile accent-{accent}"
    sub_html = f'<div class="kpi-sub {sub_cls}">{sub}</div>' if sub else ''
    st.markdown(f"""
    <div class="{cls}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Helper: page hero header
# ===========================================================================
def hero(title: str, subtitle: str):
    st.markdown(f"""
    <div class="dash-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Helper: card with header
# ===========================================================================
def card(icon: str, title: str):
    st.markdown(f"""
    <div class="dash-card">
        <div class="dash-card-header">
            <span class="dash-card-icon">{icon}</span>
            <h3>{title}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    # ── Brand ──
    st.markdown("### 🏢 Enterprise HR AI")
    st.caption("AI Workforce Intelligence")

    st.markdown('<hr class="dash-divider" style="margin:10px 0">', unsafe_allow_html=True)

    # ── Navigation ──
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

    # ── Status footer ──
    st.markdown('<hr class="dash-divider" style="margin:10px 0">', unsafe_allow_html=True)

    algo = metadata.get("algorithm", "XGBoost")
    ver  = metadata.get("version", "v1.0")
    nfeat = len(feature_names)
    st.markdown(f"""
    <div class="sb-status">
        <span class="chip chip-green"></span> System Online<br>
        Model: <strong>{algo} {ver}</strong><br>
        Features: <strong>{nfeat}</strong>
    </div>
    """, unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: Dashboard Overview                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if page == "🏠 Dashboard Overview":
    hero("AI Workforce Intelligence Platform",
         "AI-driven workforce analytics and attrition prediction for enterprise HR.")

    # ── Load data ──
    try:
        df_attr = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))
        total   = len(df_attr)
        high_risk = int((df_attr["Attrition"] == "Yes").sum())
    except Exception:
        total, high_risk = 0, 0

    try:
        df_eng = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
        eng_col = [c for c in df_eng.columns if "engagement" in c.lower()]
        avg_eng = round(float(df_eng[eng_col[0]].mean()), 1) if eng_col else 0
    except Exception:
        avg_eng = 0

    retention = round((1 - high_risk / total) * 100, 1) if total else 0
    pct = round(high_risk / total * 100, 1) if total else 0

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("👥", "Total Employees", f"{total:,}", accent="blue")
    with c2:
        kpi("⚠️", "High Risk", f"{high_risk:,}", sub=f"{pct}% attrition rate", accent="red", sub_cls="neg")
    with c3:
        kpi("📊", "Avg Engagement", f"{avg_eng}%", accent="amber")
    with c4:
        kpi("✅", "Retention Rate", f"{retention}%", sub=f"{total - high_risk:,} retained", accent="green", sub_cls="pos")

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── Charts ──
    left, right = st.columns(2)

    with left:
        card("📈", "Attrition Risk by Department")
        dept_attr = df_attr.groupby("Department")["Attrition"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).round(1).sort_values(ascending=True)
        st.bar_chart(dept_attr)

    with right:
        card("📊", "Risk Distribution")
        risk_counts = df_attr["Attrition"].value_counts()
        risk_df = pd.DataFrame({
            "Category": ["Stay (No)", "Leave (Yes)"],
            "Count": [risk_counts.get("No", 0), risk_counts.get("Yes", 0)],
        })
        st.bar_chart(risk_df.set_index("Category"))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: Attrition Prediction                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "👤 Attrition Prediction":
    hero("Employee Attrition Prediction",
         "Predict employee attrition risk using the trained XGBoost model.")

    left, right = st.columns(2)

    with left:
        card("👤", "Employee Profile")
        age            = st.number_input("Age", 18, 100, 35)
        department     = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        job_role       = st.selectbox("Job Role", [
            "Research Scientist", "Laboratory Technician", "Sales Executive",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ])
        education      = st.slider("Education Level", 1, 5, 3)
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources",
        ])
        gender         = st.selectbox("Gender", ["Male", "Female"])
        marital        = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

        card("🏢", "Work Environment")
        distance       = st.number_input("Distance from Home (km)", 0, 30, 5)
        overtime       = st.selectbox("OverTime", ["No", "Yes"])
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    with right:
        card("💰", "Compensation & Experience")
        monthly_income     = st.number_input("Monthly Income", 1000, 20000, 6000, step=500)
        total_working_years = st.number_input("Total Working Years", 0, 40, 8)
        years_at_company   = st.number_input("Years at Company", 0, 40, 5)
        job_level          = st.slider("Job Level", 1, 5, 2)

        card("⭐", "Satisfaction & Performance")
        env_sat     = st.slider("Environment Satisfaction", 1, 4, 3)
        job_inv     = st.slider("Job Involvement", 1, 4, 3)
        job_sat     = st.slider("Job Satisfaction", 1, 4, 3)
        work_life   = st.slider("Work-Life Balance", 1, 4, 3)
        perf_rating = st.selectbox("Performance Rating", [3, 4])

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Run Prediction", type="primary", use_container_width=True)

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

            if risk == "High":
                cls, emoji = "risk-high", "🚨"
            elif risk == "Medium":
                cls, emoji = "risk-medium", "⚠️"
            else:
                cls, emoji = "risk-low", "✅"

            st.markdown(f"""
            <div class="risk-banner {cls}">
                <div class="risk-status">{emoji} {risk} Risk</div>
                <div class="risk-pct">{prob:.1%}</div>
                <div class="risk-sub">Attrition Probability</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="mp-label">Prediction</div>
                    <div class="mp-value">{result['prediction']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="mp-label">Probability</div>
                    <div class="mp-value">{prob:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="mp-label">Model Version</div>
                    <div class="mp-value">{result['model_version']}</div>
                </div>
                """, unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: Engagement Analytics                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "📊 Engagement Analytics":
    hero("Engagement Analytics",
         "Workforce engagement scores, departmental comparisons, and distribution analysis.")

    try:
        df_eng  = pd.read_csv(str(DATA_RAW / "hr_performance_engagement.csv"))
        eng_col = [c for c in df_eng.columns if "engagement" in c.lower()]
        dept_col = [c for c in df_eng.columns if "department" in c.lower()]

        if eng_col:
            avg_s = round(float(df_eng[eng_col[0]].mean()), 2)
            min_s = round(float(df_eng[eng_col[0]].min()), 2)
            max_s = round(float(df_eng[eng_col[0]].max()), 2)
            total_r = len(df_eng)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi("📊", "Avg Engagement", f"{avg_s}%", accent="blue")
            with c2:
                kpi("📉", "Min Score", f"{min_s}%", accent="red")
            with c3:
                kpi("📈", "Max Score", f"{max_s}%", accent="green")
            with c4:
                kpi("📋", "Total Records", f"{total_r:,}", accent="amber")

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                card("📊", "Engagement by Department")
                if dept_col:
                    dept_eng = df_eng.groupby(dept_col[0])[eng_col[0]].mean().round(2).sort_values(ascending=True)
                    st.bar_chart(dept_eng)
                else:
                    st.info("No department column found in engagement data")
            with col_r:
                card("📈", "Engagement Distribution")
                eng_dist = df_eng[eng_col[0]].value_counts().sort_index()
                st.bar_chart(eng_dist)
        else:
            st.warning("No engagement score column found in the data")
    except Exception as e:
        st.error(f"Error loading engagement data: {e}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: Skill Gaps                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "⚠️ Skill Gaps":
    from app.services.skill_gap_service import get_required_skills_by_role, get_organization_skill_gaps

    hero("Critical Organisation Skill Gaps",
         "Identify skill gaps across roles and prioritise upskilling investments.")

    st.info(
        "Note: The raw datasets do not contain per-employee current skills. "
        "Skill gaps are computed based on role requirements vs. assumed zero current skills. "
        "This will be addressed with real employee skill data in a future iteration."
    )

    role_skills = get_required_skills_by_role()

    c1, c2 = st.columns(2)
    with c1:
        kpi("📋", "Roles with Defined Skills", f"{len(role_skills):,}", accent="blue")
    with c2:
        all_skills = set()
        for skills in role_skills.values():
            all_skills.update(skills)
        kpi("🔧", "Unique Skills Required", f"{len(all_skills):,}", accent="amber")

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    if role_skills:
        with st.expander("🔍  View Required Skills by Role", expanded=False):
            sorted_roles = sorted(role_skills.keys())
            selected_role = st.selectbox("Select a role", sorted_roles, index=0, label_visibility="collapsed")
            if selected_role:
                skills = role_skills[selected_role]
                st.markdown(f"**{selected_role}** — {len(skills)} required skills")
                st.write(", ".join(sorted(skills)))

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    card("⚠️", "Top Organisation-Wide Skill Gaps")

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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: Recommendations                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "🎯 Recommendations":
    from app.services.recommendation_service import generate_recommendations
    from app.services.skill_gap_service import get_required_skills_by_role

    hero("AI Upskilling Recommendations",
         "Personalised training recommendations based on each employee's role skill gaps.")

    role_skills = get_required_skills_by_role()
    df_attr     = pd.read_csv(str(DATA_RAW / "employee_attrition.csv"))

    n_employees = st.slider("Number of employees to analyze", 5, 50, 10)
    employee_roles = [
        {"employee_id": str(row["EmployeeNumber"]), "role": row["JobRole"]}
        for _, row in df_attr.head(n_employees).iterrows()
    ]

    recs = generate_recommendations(employee_roles, {})

    if recs:
        total_gaps = sum(r["gap_count"] for r in recs)
        avg_gaps   = round(total_gaps / len(recs), 1) if recs else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("👥", "Employees Analyzed", f"{len(recs)}", accent="blue")
        with c2:
            kpi("⚠️", "Total Skill Gaps", f"{total_gaps:,}", accent="red")
        with c3:
            kpi("📊", "Avg Gaps / Employee", f"{avg_gaps}", accent="amber")

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        for rec in recs:
            if rec["recommendations"]:
                with st.expander(
                    f"👤  Employee {rec['employee_id']}  —  {rec['role']}  ({rec['gap_count']} gaps)"
                ):
                    for rec_text in rec["recommendations"]:
                        st.markdown(f'<div class="rec-item">{rec_text}</div>', unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: Model Info                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "🤖 Model Info":
    hero("Model Information",
         "Details of the trained attrition prediction model.")

    metrics = metadata.get("metrics", {})

    # ── Model metadata ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="info-card">
            <div class="ic-label">Algorithm</div>
            <div class="ic-value">{metadata.get("algorithm", "Unknown")}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-card">
            <div class="ic-label">Version</div>
            <div class="ic-value">{metadata.get("version", "Unknown")}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="info-card">
            <div class="ic-label">Training Date</div>
            <div class="ic-value">{metadata.get("training_date", "Unknown")}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="info-card">
            <div class="ic-label">Features</div>
            <div class="ic-value">{len(feature_names)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # ── Performance ──
    card("📈", "Performance Metrics")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="mp-label">ROC-AUC</div>
            <div class="mp-value">{metrics.get('roc_auc', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="mp-label">F1 Score</div>
            <div class="mp-value">{metrics.get('f1_score', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="mp-label">Precision</div>
            <div class="mp-value">{metrics.get('precision', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="mp-label">Recall</div>
            <div class="mp-value">{metrics.get('recall', 0):.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Feature Names"):
            st.write(feature_names)
    with col_b:
        with st.expander("Training Data Details"):
            st.json(metadata.get("preprocessing", {}))
            st.json(metadata.get("class_distribution", {}))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: Monitoring                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif page == "📈 Monitoring":
    from app.monitoring import load_predictions, compute_drift_stats, get_model_health

    hero("Prediction Monitoring",
         "Live model health, drift detection, and recent prediction activity.")

    try:
        health = get_model_health()
        drift  = compute_drift_stats()

        model_ok    = health.get("model_file_exists", False)
        pred_count  = health.get("recent_prediction_count", 0)
        model_ver   = health.get("model_version", "Unknown")

        c1, c2, c3 = st.columns(3)
        with c1:
            status = "🟢 Healthy" if model_ok else "🔴 Missing"
            kpi("🤖", "Model Status", status, accent="green" if model_ok else "red")
        with c2:
            kpi("📊", "Recent Predictions", f"{pred_count:,}", accent="blue")
        with c3:
            kpi("🏷️", "Model Version", model_ver, accent="amber")

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            card("📉", "Drift Detection")
            st.json(drift)

        with col_r:
            card("📋", "Recent Predictions")
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

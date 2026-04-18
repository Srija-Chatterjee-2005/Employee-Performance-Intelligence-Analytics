from __future__ import annotations
import os
from io import BytesIO
import pandas as pd
import streamlit as st

from src.data_generator import save_dataset
from src.insights import add_recommendations
from src.model import predict_single, save_model, train_and_evaluate, CLASS_ORDER
from src.visuals import (
    confusion_matrix_fig,
    department_performance_fig,
    experience_boxplot_fig,
    feature_importance_fig,
    heatmap_fig,
    performance_distribution_fig,
    scatter_salary_vs_performance_fig,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "data", "employee_data.csv")
MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.pkl")

st.set_page_config(page_title="Employee Performance Intelligence System", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
.main {
    background: linear-gradient(135deg, #06141f 0%, #0a2535 45%, #103c53 100%);
    color: #eef5f9;
}
.block-container {padding-top: 1rem; padding-bottom: 1.2rem;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061018 0%, #0b1c28 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.stTabs [data-baseweb="tab-list"] {gap: 10px;}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(31, 201, 169, 0.18) !important;
    border: 1px solid rgba(31, 201, 169, 0.45) !important;
}
.kpi-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.18);
    min-height: 112px;
}
.kpi-label {font-size: 0.95rem; color: #cce0eb; margin-bottom: 8px;}
.kpi-value {font-size: 1.85rem; font-weight: 700; color: #ffffff;}
.hero {
    padding: 18px 22px;
    border-radius: 20px;
    background: linear-gradient(90deg, rgba(255,255,255,0.09), rgba(31,201,169,0.10));
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 16px;
}
.small-note {color: #d6e5ee; font-size: 0.92rem;}
.section-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 16px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_or_generate_data():
    if not os.path.exists(DATA_PATH):
        save_dataset(DATA_PATH, n_samples=1400, random_state=42)
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_model_and_metrics(df: pd.DataFrame):
    model, metrics = train_and_evaluate(df)
    save_model(model, MODEL_PATH)
    return model, metrics


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="employees")
    return output.getvalue()


def apply_filters(df: pd.DataFrame):
    st.sidebar.title("🎛️ Smart Filters")
    dept = st.sidebar.multiselect(
        "Department",
        sorted(df["department"].dropna().unique()),
        default=sorted(df["department"].dropna().unique()),
    )
    gender = st.sidebar.multiselect(
        "Gender",
        sorted(df["gender"].dropna().unique()),
        default=sorted(df["gender"].dropna().unique()),
    )
    education = st.sidebar.multiselect(
        "Education",
        sorted(df["education"].dropna().unique()),
        default=sorted(df["education"].dropna().unique()),
    )
    perf_classes = st.sidebar.multiselect(
        "Performance Class",
        sorted(df["performance_label"].dropna().unique()),
        default=sorted(df["performance_label"].dropna().unique()),
    )
    exp_range = st.sidebar.slider(
        "Experience Range",
        int(df["years_experience"].min()),
        int(df["years_experience"].max()),
        (0, int(df["years_experience"].max())),
    )
    salary_range = st.sidebar.slider(
        "Salary Range",
        int(df["monthly_salary"].min()),
        int(df["monthly_salary"].max()),
        (int(df["monthly_salary"].min()), int(df["monthly_salary"].max())),
    )
    satisfaction_range = st.sidebar.slider(
        "Satisfaction Score",
        int(df["satisfaction_score"].fillna(df["satisfaction_score"].median()).min()),
        int(df["satisfaction_score"].fillna(df["satisfaction_score"].median()).max()),
        (1, 10),
    )

    filtered = df[
        (df["department"].isin(dept))
        & (df["gender"].isin(gender))
        & (df["education"].isin(education))
        & (df["performance_label"].isin(perf_classes))
        & (df["years_experience"].between(exp_range[0], exp_range[1]))
        & (df["monthly_salary"].between(salary_range[0], salary_range[1]))
        & (df["satisfaction_score"].fillna(df["satisfaction_score"].median()).between(satisfaction_range[0], satisfaction_range[1]))
    ].copy()
    return filtered


def hero_section(df: pd.DataFrame, metrics: dict):
    st.markdown(
        f"""
        <div class='hero'>
            <h2 style='margin:0;'>📊 Employee Performance Intelligence System</h2>
            <p style='margin:8px 0 0 0;'>Corporate-style HR analytics dashboard with performance prediction, smart filtering, model comparison, and action-ready recommendations.</p>
            <p class='small-note' style='margin-top:8px;'>Current best model: <b>{metrics['best_model_name']}</b> | Dataset size: <b>{len(df)}</b> employees</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(df: pd.DataFrame):
    total = len(df)
    high = int((df["performance_label"] == "High Performer").sum())
    avg_score = float(df["performance_score"].mean()) if total else 0
    avg_attrition = float(df["attrition_risk"].mean()) if total else 0
    cols = st.columns(4)
    metrics = [
        ("👥 Employees", f"{total}"),
        ("⭐ High Performers", f"{high}"),
        ("📈 Avg Performance", f"{avg_score:.1f}"),
        ("⚠️ Avg Attrition Risk", f"{avg_attrition:.1f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )


def overview_tab(df: pd.DataFrame):
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("### Department Snapshot")
        dept_summary = (
            df.groupby("department", as_index=False)
            .agg(
                employees=("employee_id", "count"),
                avg_performance=("performance_score", "mean"),
                avg_attrition=("attrition_risk", "mean"),
            )
            .sort_values("avg_performance", ascending=False)
        )
        st.dataframe(dept_summary.round(2), use_container_width=True)
    with c2:
        st.markdown("### Quick Observations")
        top_dept = dept_summary.iloc[0]["department"] if not dept_summary.empty else "N/A"
        highest_attrition = dept_summary.sort_values("avg_attrition", ascending=False).iloc[0]["department"] if not dept_summary.empty else "N/A"
        low_training = int((df["training_hours"].fillna(0) < 20).sum())
        st.markdown(
            f"""
            <div class='section-box'>
            <p>• <b>{top_dept}</b> currently leads in average performance.</p>
            <p>• <b>{highest_attrition}</b> has the highest average attrition risk.</p>
            <p>• <b>{low_training}</b> employees may benefit from additional training.</p>
            <p>• This dashboard helps HR spot promotion candidates, performance gaps, and retention priorities.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    a1, a2 = st.columns(2)
    with a1:
        st.pyplot(performance_distribution_fig(df), use_container_width=True)
    with a2:
        st.pyplot(department_performance_fig(df), use_container_width=True)


def employee_explorer(df: pd.DataFrame):
    st.subheader("🧾 Employee Explorer")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        keyword = st.text_input("Search by employee ID or job role")
    with c2:
        sort_col = st.selectbox("Sort by", ["performance_score", "monthly_salary", "years_experience", "attrition_risk"])
    show_df = df.copy()
    if keyword:
        mask = show_df["employee_id"].str.contains(keyword, case=False, na=False) | show_df["job_role"].str.contains(keyword, case=False, na=False)
        show_df = show_df[mask]
    show_df = add_recommendations(show_df).sort_values(sort_col, ascending=False)
    st.dataframe(show_df, use_container_width=True, height=430)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Download filtered CSV",
            data=show_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_employees.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "⬇️ Download filtered Excel",
            data=to_excel_bytes(show_df),
            file_name="filtered_employees.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def analytics_tab(df: pd.DataFrame):
    st.subheader("📊 Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(scatter_salary_vs_performance_fig(df), use_container_width=True)
        st.pyplot(experience_boxplot_fig(df), use_container_width=True)
    with c2:
        st.pyplot(heatmap_fig(df), use_container_width=True)
        training_summary = (
            df.groupby("performance_label", as_index=False)["training_hours"]
            .mean()
            .rename(columns={"training_hours": "avg_training_hours"})
        )
        st.markdown("### Training Hours by Performance Class")
        st.dataframe(training_summary.round(2), use_container_width=True)


def prediction_tab(model, df: pd.DataFrame):
    st.subheader("🧠 Prediction Center")
    st.markdown("<p class='small-note'>Use this form to simulate a new employee profile and predict the likely performance class.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 21, 60, 29)
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
        department = st.selectbox("Department", sorted(df["department"].dropna().unique()))
        education = st.selectbox("Education", sorted(df["education"].dropna().unique()))
        years_experience = st.number_input("Years of Experience", 0, 35, 4)
    with col2:
        monthly_salary = st.number_input("Monthly Salary", 25000, 200000, 65000, step=1000)
        training_hours = st.number_input("Training Hours", 0, 100, 28)
        attendance_percent = st.slider("Attendance %", 50.0, 100.0, 89.5)
        projects_completed = st.number_input("Projects Completed", 0, 20, 6)
        overtime_hours = st.number_input("Overtime Hours", 0, 50, 12)
    with col3:
        satisfaction_score = st.slider("Satisfaction Score", 1, 10, 7)
        work_life_balance = st.slider("Work-Life Balance", 1, 10, 7)
        manager_rating = st.slider("Manager Rating", 1.0, 5.0, 3.8, step=0.1)
        last_promotion_years = st.number_input("Years Since Last Promotion", 0, 12, 2)
        remote_days_per_month = st.number_input("Remote Days/Month", 0, 20, 5)
        skill_score = st.slider("Skill Score", 30, 100, 72)

    job_role_options = sorted(df[df["department"] == department]["job_role"].dropna().unique()) or sorted(df["job_role"].dropna().unique())
    job_role = st.selectbox("Job Role", job_role_options)

    attrition_risk = max(5.0, min(95.0, (10 - satisfaction_score) * 4 + overtime_hours * 0.9 + max(last_promotion_years - 3, 0) * 4))

    input_df = pd.DataFrame([
        {
            "age": age,
            "gender": gender,
            "department": department,
            "job_role": job_role,
            "education": education,
            "years_experience": years_experience,
            "monthly_salary": monthly_salary,
            "training_hours": training_hours,
            "attendance_percent": attendance_percent,
            "projects_completed": projects_completed,
            "overtime_hours": overtime_hours,
            "satisfaction_score": satisfaction_score,
            "work_life_balance": work_life_balance,
            "manager_rating": manager_rating,
            "last_promotion_years": last_promotion_years,
            "remote_days_per_month": remote_days_per_month,
            "skill_score": skill_score,
            "attrition_risk": round(attrition_risk, 1),
            "performance_score": 0.0,
        }
    ])

    if st.button("Predict Performance", use_container_width=True):
        pred, proba = predict_single(model, input_df)
        p1, p2 = st.columns([0.9, 1.1])
        with p1:
            st.success(f"Predicted Class: {pred}")
            st.metric("Estimated Attrition Risk", f"{attrition_risk:.1f}")
        with p2:
            if proba:
                proba_df = pd.DataFrame({"Class": list(proba.keys()), "Probability": list(proba.values())}).sort_values("Probability", ascending=False)
                st.dataframe(proba_df, use_container_width=True)

        recommendations = []
        if attendance_percent < 80:
            recommendations.append("Improve attendance consistency")
        if training_hours < 20:
            recommendations.append("Complete additional training")
        if satisfaction_score <= 5:
            recommendations.append("Schedule a manager feedback session")
        if overtime_hours > 25:
            recommendations.append("Reduce burnout risk through workload balancing")
        if years_experience > 5 and pred == "High Performer":
            recommendations.append("Consider promotion review")
        st.info("HR Action Suggestions: " + (", ".join(recommendations) if recommendations else "Continue standard coaching and monitoring."))


def recommendations_tab(df: pd.DataFrame):
    st.subheader("💡 Recommendations & Reports")
    df = add_recommendations(df)
    needs_training = df[(df["training_hours"].fillna(0) < 20) | (df["performance_label"] == "Needs Improvement")]
    promotion_ready = df[(df["performance_label"] == "High Performer") & (df["years_experience"] >= 4)]
    retention_risk = df[df["attrition_risk"] >= 60]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📚 Training Focus")
        st.dataframe(needs_training[["employee_id", "department", "training_hours", "performance_label"]].head(10), use_container_width=True)
    with c2:
        st.markdown("### 🚀 Promotion Shortlist")
        st.dataframe(promotion_ready[["employee_id", "department", "years_experience", "performance_label"]].head(10), use_container_width=True)
    with c3:
        st.markdown("### ⚠️ Retention Watchlist")
        st.dataframe(retention_risk[["employee_id", "department", "attrition_risk", "satisfaction_score"]].head(10), use_container_width=True)

    st.markdown("### Full HR Recommendation Table")
    st.dataframe(df[["employee_id", "department", "performance_label", "attrition_risk", "hr_recommendation"]], use_container_width=True, height=350)


def model_lab_tab(metrics: dict):
    st.subheader("🧪 Model Lab")
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.markdown("### Model Comparison")
        comparison_df = metrics["results_df"].copy()
        for col in ["accuracy", "precision", "recall", "weighted_f1"]:
            comparison_df[col] = comparison_df[col].round(4)
        st.dataframe(comparison_df, use_container_width=True)
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(metrics["classification_report"]).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
    with c2:
        st.pyplot(confusion_matrix_fig(metrics["confusion_matrix"], CLASS_ORDER), use_container_width=True)
        feature_importance_df = metrics.get("feature_importance", pd.DataFrame())
        if not feature_importance_df.empty:
            st.pyplot(feature_importance_fig(feature_importance_df), use_container_width=True)


def main():
    df = load_or_generate_data()
    model, metrics = get_model_and_metrics(df)
    filtered_df = apply_filters(df)

    hero_section(filtered_df, metrics)
    kpi_row(filtered_df)
    st.markdown("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview",
        "🧾 Employee Explorer",
        "📊 Analytics",
        "🧠 Prediction Center",
        "💡 Recommendations",
    ])
    with tab1:
        overview_tab(filtered_df)
    with tab2:
        employee_explorer(filtered_df)
    with tab3:
        analytics_tab(filtered_df)
    with tab4:
        prediction_tab(model, df)
    with tab5:
        recommendations_tab(filtered_df)

    st.markdown("---")
    model_lab_tab(metrics)


if __name__ == "__main__":
    main()

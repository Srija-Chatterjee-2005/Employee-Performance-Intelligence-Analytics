from __future__ import annotations
import pandas as pd


def generate_recommendation(row: pd.Series) -> str:
    recommendations = []
    if row.get("attendance_percent", 100) < 80:
        recommendations.append("attendance improvement plan")
    if row.get("training_hours", 100) < 20:
        recommendations.append("skill development training")
    if row.get("satisfaction_score", 10) <= 5:
        recommendations.append("manager check-in and engagement support")
    if row.get("overtime_hours", 0) > 25:
        recommendations.append("workload rebalance")
    if row.get("last_promotion_years", 0) > 5 and row.get("performance_score", 0) >= 70:
        recommendations.append("promotion review")
    if row.get("attrition_risk", 0) > 60:
        recommendations.append("retention action")

    if not recommendations:
        if row.get("performance_label") == "High Performer":
            return "Strong performer. Consider leadership grooming or fast-track opportunities."
        return "Performance is stable. Continue regular coaching and monitoring."

    return "Recommend: " + ", ".join(recommendations).capitalize() + "."


def add_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hr_recommendation"] = out.apply(generate_recommendation, axis=1)
    return out

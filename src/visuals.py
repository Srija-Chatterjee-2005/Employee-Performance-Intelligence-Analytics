from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")


def performance_distribution_fig(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.countplot(
        data=df,
        x="performance_label",
        order=["Needs Improvement", "Average Performer", "High Performer"],
        ax=ax,
    )
    ax.set_title("Performance Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Performance Label")
    ax.set_ylabel("Employee Count")
    plt.xticks(rotation=10)
    plt.tight_layout()
    return fig


def department_performance_fig(df: pd.DataFrame):
    dept_avg = (
        df.groupby("department", as_index=False)["performance_score"]
        .mean()
        .sort_values("performance_score", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=dept_avg, x="performance_score", y="department", ax=ax)
    ax.set_title("Average Performance Score by Department", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Score")
    ax.set_ylabel("Department")
    plt.tight_layout()
    return fig


def scatter_salary_vs_performance_fig(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.scatterplot(
        data=df,
        x="monthly_salary",
        y="performance_score",
        hue="performance_label",
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Salary vs Performance Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Monthly Salary")
    ax.set_ylabel("Performance Score")
    plt.tight_layout()
    return fig


def heatmap_fig(df: pd.DataFrame):
    corr_cols = [
        "years_experience", "monthly_salary", "training_hours", "attendance_percent",
        "projects_completed", "overtime_hours", "satisfaction_score", "manager_rating",
        "skill_score", "attrition_risk", "performance_score"
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def experience_boxplot_fig(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.boxplot(data=df, x="performance_label", y="years_experience", ax=ax)
    ax.set_title("Experience Spread Across Performance Classes", fontsize=13, fontweight="bold")
    ax.set_xlabel("Performance Label")
    ax.set_ylabel("Years of Experience")
    plt.xticks(rotation=10)
    plt.tight_layout()
    return fig


def confusion_matrix_fig(confusion_matrix: np.ndarray, class_order: list[str]):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_order, yticklabels=class_order, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def feature_importance_fig(feature_importance_df: pd.DataFrame, top_n: int = 12):
    plot_df = feature_importance_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.barplot(data=plot_df, x="importance", y="feature", ax=ax)
    ax.set_title(f"Top {top_n} Important Features", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig

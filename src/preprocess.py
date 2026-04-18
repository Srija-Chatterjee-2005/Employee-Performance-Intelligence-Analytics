from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "years_experience",
    "monthly_salary",
    "training_hours",
    "attendance_percent",
    "projects_completed",
    "overtime_hours",
    "satisfaction_score",
    "work_life_balance",
    "manager_rating",
    "last_promotion_years",
    "remote_days_per_month",
    "skill_score",
    "attrition_risk",
]

CATEGORICAL_FEATURES = ["gender", "department", "job_role", "education"]
TARGET = "performance_label"


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["productivity_index"] = (
        data["projects_completed"].fillna(data["projects_completed"].median()) * 4
        + data["attendance_percent"].fillna(data["attendance_percent"].median()) * 0.4
        + data["skill_score"].fillna(data["skill_score"].median()) * 0.3
        - data["overtime_hours"].fillna(data["overtime_hours"].median()) * 0.5
    )
    data["training_effectiveness"] = (
        data["training_hours"].fillna(data["training_hours"].median())
        * data["manager_rating"].fillna(data["manager_rating"].median())
    )
    data["salary_per_experience"] = data["monthly_salary"] / (data["years_experience"].fillna(0) + 1)
    return data


def get_feature_lists():
    engineered_numeric = NUMERIC_FEATURES + [
        "productivity_index",
        "training_effectiveness",
        "salary_per_experience",
    ]
    return engineered_numeric, CATEGORICAL_FEATURES, TARGET


def build_preprocessor() -> ColumnTransformer:
    numeric_features, categorical_features, _ = get_feature_lists()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

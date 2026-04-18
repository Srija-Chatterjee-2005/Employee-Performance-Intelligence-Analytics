from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, feature_engineering, get_feature_lists


MODEL_CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

CLASS_ORDER = ["Needs Improvement", "Average Performer", "High Performer"]


def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(100)]

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 2:
            importance_values = np.mean(np.abs(coef), axis=0)
        else:
            importance_values = np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_importance = pd.DataFrame(
        {"feature": feature_names[: len(importance_values)], "importance": importance_values}
    ).sort_values("importance", ascending=False)
    return feature_importance


def train_and_evaluate(df: pd.DataFrame):
    data = feature_engineering(df)
    numeric_features, categorical_features, target = get_feature_lists()
    X = data[numeric_features + categorical_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    trained_models = {}

    for model_name, model in MODEL_CANDIDATES.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)

        results.append(
            {
                "model": model_name,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "weighted_f1": f1,
            }
        )
        trained_models[model_name] = {
            "pipeline": pipeline,
            "predictions": preds,
        }

    results_df = pd.DataFrame(results).sort_values(by=["weighted_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_models[best_model_name]["pipeline"]
    best_preds = trained_models[best_model_name]["predictions"]

    metrics = {
        "best_model_name": best_model_name,
        "results_df": results_df,
        "accuracy": accuracy_score(y_test, best_preds),
        "f1": f1_score(y_test, best_preds, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, best_preds, labels=CLASS_ORDER),
        "classification_report": classification_report(y_test, best_preds, output_dict=True, zero_division=0),
        "X_test": X_test,
        "y_test": y_test,
        "predictions": best_preds,
        "feature_importance": _extract_feature_importance(best_pipeline),
    }
    return best_pipeline, metrics


def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    return joblib.load(filepath)


from src.preprocess import feature_engineering

def predict_single(model, input_df):
    input_df = feature_engineering(input_df.copy())

    pred = model.predict(input_df)[0]
    proba = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = list(model.named_steps["model"].classes_)
        proba = dict(zip(classes, probs))

    return pred, proba
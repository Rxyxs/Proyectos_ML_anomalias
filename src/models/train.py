"""Entrenamiento y validación de modelos de detección de fraude."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "isFraud"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model.joblib"


def split_data(df: pd.DataFrame, target: str = TARGET_COLUMN, test_size: float = 0.2, random_state: int = 42):
    """Divide el dataset en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_models(y_train: pd.Series, random_state: int = 42) -> dict:
    """Define los modelos candidatos, cada uno con manejo de clases desbalanceadas."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": _build_xgboost(scale_pos_weight, random_state),
    }


def _build_xgboost(scale_pos_weight: float, random_state: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=random_state,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un único modelo XGBoost sobre los datos de entrenamiento."""
    model = _build_xgboost(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de evaluación relevantes para un problema desbalanceado."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "report": classification_report(y_test, y_pred, target_names=["No fraude", "Fraude"], digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def train_and_compare(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[str, object, dict]:
    """Entrena varios modelos candidatos y los compara por PR-AUC (más informativo que ROC-AUC en fraude)."""
    models = get_models(y_train)
    results = {}

    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = model
        results[name] = metrics

        print(f"--- {name} ---")
        print(metrics["report"])
        print(f"Matriz de confusión:\n{metrics['confusion_matrix']}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    best_name = max(results, key=lambda name: results[name]["pr_auc"])
    return best_name, results[best_name]["model"], results


def save_model(model, path: Path = MODEL_OUTPUT_PATH) -> None:
    """Serializa el modelo entrenado en disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.preprocessing import clean_data
    from src.features.build_features import build_features

    df = load_raw_data()
    df = clean_data(df)
    df = build_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    best_name, best_model, results = train_and_compare(X_train, y_train, X_test, y_test)

    print(f"\nMejor modelo: {best_name} (PR-AUC={results[best_name]['pr_auc']:.4f})")

    save_model(best_model)
    print(f"Modelo guardado en {MODEL_OUTPUT_PATH}")

    from src.models.visualize import generate_comparison_report

    figure_paths = generate_comparison_report(results, X_test, y_test)
    print("\nFiguras comparativas guardadas en:")
    for name, path in figure_paths.items():
        print(f"  {name}: {path}")

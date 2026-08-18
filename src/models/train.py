"""Entrenamiento y validación de modelos de detección de fraude."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "isFraud"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model.joblib"


def split_data(df: pd.DataFrame, target: str = TARGET_COLUMN, test_size: float = 0.2, random_state: int = 42):
    """Divide el dataset en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un modelo XGBoost sobre los datos de entrenamiento."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path: Path = MODEL_OUTPUT_PATH) -> None:
    """Serializa el modelo entrenado en disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

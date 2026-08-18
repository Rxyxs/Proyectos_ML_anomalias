"""Inferencia sobre datos nuevos utilizando un modelo ya entrenado."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model.joblib"


def load_model(path: Path = MODEL_PATH):
    """Carga un modelo previamente entrenado y serializado con joblib."""
    return joblib.load(path)


def predict(model, X: pd.DataFrame) -> pd.Series:
    """Genera predicciones binarias de fraude para las filas de X."""
    return pd.Series(model.predict(X), index=X.index, name="isFraudPred")


def predict_proba(model, X: pd.DataFrame) -> pd.Series:
    """Genera la probabilidad de fraude estimada para las filas de X."""
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="fraudProbability")

"""Modelos no supervisados de detección de anomalías (fraude desconocido / zero-day)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Tasa de contaminación esperada: estimación previa del negocio (no derivada del set de
# prueba), aproximada a la tasa histórica de fraude observada en PaySim (~0.13%).
DEFAULT_CONTAMINATION = 0.001


def build_isolation_forest(
    contamination: float = DEFAULT_CONTAMINATION, n_estimators: int = 200, random_state: int = 42
) -> IsolationForest:
    """Isolation Forest: aísla puntos con particiones aleatorias; las anomalías requieren menos particiones."""
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )


def build_lof(contamination: float = DEFAULT_CONTAMINATION, n_neighbors: int = 20) -> LocalOutlierFactor:
    """Local Outlier Factor en modo novelty: compara la densidad local de un punto contra la de sus vecinos."""
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )


def anomaly_score(model, X) -> np.ndarray:
    """Anomaly score continuo y homogéneo entre modelos: valores más altos = más anómalo.

    Ambos modelos exponen `score_samples`, donde el valor es más bajo cuanto más anómalo
    es el punto; se invierte el signo para tener una única convención de score.
    """
    return -model.score_samples(X)

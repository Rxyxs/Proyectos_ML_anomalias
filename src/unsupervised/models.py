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


class MADBaseline:
    """Baseline estadístico robusto: score = distancia máxima en MAD-z entre las columnas.

    Para cada feature se calcula el z-score robusto (mediana / Median Absolute Deviation en
    lugar de media / desviación estándar, insensible a la cola de outliers de montos y
    saldos). El score final de una fila es el máximo de |z| entre sus features, siguiendo la
    misma lógica que Isolation Forest/LOF: un solo feature muy desviado ya hace anómala la
    transacción. No requiere entrenamiento iterativo: solo memoriza mediana y MAD por columna.
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def fit(self, X) -> "MADBaseline":
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        # Constante 1.4826: hace que el MAD sea consistente con la desviación estándar
        # bajo una distribución normal, convención estándar del z-score robusto.
        self.mad_ = 1.4826 * np.median(np.abs(X - self.median_), axis=0) + self.eps
        return self

    def score_samples(self, X) -> np.ndarray:
        """Devuelve -score (convención sklearn: más bajo = más anómalo) para reusar `anomaly_score`."""
        X = np.asarray(X, dtype=float)
        z = np.abs(X - self.median_) / self.mad_
        max_abs_z = z.max(axis=1)
        return -max_abs_z


def build_mad_baseline() -> MADBaseline:
    """Baseline estadístico (MAD-z score) — sin hiperparámetros de contaminación."""
    return MADBaseline()

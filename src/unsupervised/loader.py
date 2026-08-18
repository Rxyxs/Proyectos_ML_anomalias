"""Carga de datos para detección no supervisada de anomalías (fraude zero-day).

Reutiliza el dataset PaySim ya descargado por src.data.loader en vez de descargar
'mlg-ulb/creditcardfraud' vía kagglehub: mantiene un único dataset fuente para todo
el repositorio y no depende de credenciales de Kaggle adicionales.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessing import clean_data
from src.features.build_features import build_features

TARGET_COLUMN = "isFraud"

# Entrenamiento: muestra de transacciones normales (suficiente para que Isolation Forest
# y Local Outlier Factor estimen densidad/aislamiento; LOF en modo novelty no escala a
# millones de puntos porque cada score_samples requiere una búsqueda de vecinos contra
# todo el set de entrenamiento).
TRAIN_NORMAL_SIZE = 30_000

# Prueba: muestra de normales + TODAS las transacciones fraudulentas disponibles, para
# tener suficientes anomalías reales con las que medir Precision@k/Recall@k sin heredar
# el costo de puntuar millones de filas contra el índice de vecinos de LOF.
TEST_NORMAL_SIZE = 50_000


def get_unsupervised_data(
    train_normal_size: int = TRAIN_NORMAL_SIZE,
    test_normal_size: int = TEST_NORMAL_SIZE,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Prepara datos para detección no supervisada.

    Devuelve (X_train, X_test, y_test) donde X_train contiene únicamente transacciones
    normales (isFraud == 0) y (X_test, y_test) es un conjunto de prueba mixto con
    normales y anomalías, usado solo para evaluar — nunca para ajustar los modelos.
    """
    df = load_raw_data()
    df = clean_data(df)
    df = build_features(df)

    normal = df[df[TARGET_COLUMN] == 0]
    fraud = df[df[TARGET_COLUMN] == 1]

    train_normal = normal.sample(n=train_normal_size, random_state=random_state)
    remaining_normal = normal.drop(train_normal.index)
    test_normal = remaining_normal.sample(n=test_normal_size, random_state=random_state)

    test_df = pd.concat([test_normal, fraud]).sample(frac=1, random_state=random_state)

    X_train = train_normal.drop(columns=[TARGET_COLUMN])
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_test

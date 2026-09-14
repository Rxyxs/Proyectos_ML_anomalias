"""Módulo 8 — por qué se disparó esta alerta.

Un analista que recibe una transacción marcada necesita saber qué la marcó. Sin eso la alerta
no es accionable: no se puede confirmar, no se puede descartar rápido y no se puede explicar
a un cliente que reclama.

De los dieciséis detectores, solo LODA trae atribución propia (Módulo 6) y HBOS y ECOD la
admitirían por construcción. Los otros trece no. Esta implementación es **agnóstica al
modelo**: no mira el interior del detector, solo lo consulta.

El método es oclusión. Se reemplaza una feature por su valor típico —la mediana del
entrenamiento— y se vuelve a puntuar. Si el score se desploma, esa feature era la que
sostenía la anomalía; si no se mueve, no estaba aportando nada. Repetido sobre las quince
columnas da un ranking de responsabilidad.

Dos advertencias sobre cómo leer el resultado, porque el método tiene límites reales:

- **Mide contribución marginal, no causalidad.** Si dos features están correlacionadas y
  juntas hacen anómala a la transacción, ocluir una sola puede no mover el score, y las dos
  aparecerían como irrelevantes. Es el mismo problema que tienen los métodos de permutación.
- **La línea de base importa.** Reemplazar por la mediana pregunta "¿qué pasa si esta
  transacción fuera típica en esta columna?", que es la pregunta correcta para explicar una
  alerta, pero no la única posible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def occlusion_attribution(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Cuánto cae el anomaly score al reemplazar cada feature por su valor típico.

    Devuelve una matriz (n_filas, n_features) donde el valor positivo significa que esa
    columna estaba **sosteniendo** la anomalía. Un valor negativo significa lo contrario: sin
    esa columna la transacción se vería aún más rara.

    `score_fn` recibe una matriz escalada y devuelve un score donde más alto = más anómalo.
    `baseline` es el vector de valores típicos, normalmente la mediana del entrenamiento ya
    escalada.
    """
    X_scaled = np.asarray(X_scaled, dtype=float)
    baseline = np.asarray(baseline, dtype=float).ravel()

    if baseline.shape[0] != X_scaled.shape[1]:
        raise ValueError(
            f"La línea de base tiene {baseline.shape[0]} valores y los datos "
            f"{X_scaled.shape[1]} columnas."
        )
    if feature_names is not None and len(feature_names) != X_scaled.shape[1]:
        raise ValueError(
            f"Se recibieron {len(feature_names)} nombres para {X_scaled.shape[1]} columnas."
        )

    score_original = np.asarray(score_fn(X_scaled), dtype=float)
    atribucion = np.empty_like(X_scaled)

    for j in range(X_scaled.shape[1]):
        ocluido = X_scaled.copy()
        ocluido[:, j] = baseline[j]
        atribucion[:, j] = score_original - np.asarray(score_fn(ocluido), dtype=float)

    return atribucion


def explain_rows(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> list[list[tuple[str, float]]]:
    """Las `top_k` features más responsables de cada fila, ordenadas de mayor a menor aporte."""
    atribucion = occlusion_attribution(score_fn, X_scaled, baseline, feature_names)
    nombres = np.asarray(feature_names)

    explicaciones = []
    for fila in atribucion:
        orden = np.argsort(fila)[::-1][:top_k]
        explicaciones.append([(str(nombres[j]), float(fila[j])) for j in orden])
    return explicaciones


def attribution_frame(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """La matriz de atribución como DataFrame, con las features como columnas."""
    return pd.DataFrame(
        occlusion_attribution(score_fn, X_scaled, baseline, feature_names),
        columns=list(feature_names),
    )


def global_importance(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Aporte promedio de cada feature sobre un conjunto de transacciones.

    Promediar la atribución absoluta sobre muchas alertas da una lectura global: en qué se
    apoya el detector en general, no solo en un caso. Útil para detectar que un detector
    depende de una única columna, que es una fragilidad operativa aunque la métrica sea buena.
    """
    atribucion = occlusion_attribution(score_fn, X_scaled, baseline, feature_names)
    tabla = pd.DataFrame({
        "feature": list(feature_names),
        "aporte_medio": atribucion.mean(axis=0),
        "aporte_absoluto_medio": np.abs(atribucion).mean(axis=0),
    })
    return tabla.sort_values("aporte_absoluto_medio", ascending=False).reset_index(drop=True)

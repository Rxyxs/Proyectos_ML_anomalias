"""Módulo 7 — aprendizaje activo: en qué gastar la capacidad de revisión.

El Módulo 5 estableció que el cuello de botella no es económico sino de capacidad: por
debajo del costo de equilibrio conviene revisar todo, así que lo que limita es cuántos casos
alcanza a mirar el equipo. El Módulo 7 agrega que esas revisiones **producen etiquetas**, y
las etiquetas mejoran el modelo. Entonces la pregunta operativa cambia: ya no es solo "qué
alertas reviso hoy" sino "qué alertas conviene revisar para detectar mejor mañana".

Son dos objetivos que compiten. Revisar siempre las de score más alto maximiza el fraude
atrapado hoy —explotación— pero devuelve etiquetas todas parecidas entre sí, de las que el
modelo aprende poco. Revisar donde el modelo duda maximiza lo aprendido —exploración— pero
gasta capacidad en casos que probablemente sean legítimos.

Este módulo simula el circuito completo, ronda a ronda, con cuatro estrategias:

- `random`: línea de base. Sobre una prevalencia del 0,08% casi no encuentra positivos.
- `top_score`: la cola de alertas clásica, ordenada por el detector no supervisado. Es una
  cola **estática**: no aprende nada de las etiquetas que va produciendo.
- `top_model`: la misma idea de explotación, pero ordenando por el modelo supervisado que se
  reentrena en cada ronda. La cola **se actualiza**.
- `uncertainty`: pura exploración, los casos con probabilidad predicha más cercana a 0,5.
- `hybrid`: mitad `top_model` y mitad `uncertainty`, que es lo que un equipo hace en la
  práctica sin llamarlo así.

`top_score` y `top_model` están las dos a propósito. Sin la segunda, comparar la cola clásica
contra `uncertainty` mezcla dos variables —explotar contra explorar, y ranker estático contra
ranker que aprende— y cualquier diferencia sería inatribuible. `top_model` aísla la primera.

La primera ronda no tiene modelo del que extraer incertidumbre, así que todas las estrategias
arrancan desde el score no supervisado. Ese arranque en frío no es un detalle de
implementación: es la razón por la que tener quince detectores sin etiquetas vale la pena.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.adaptive.stacking import build_training_set, fit_classifier

STRATEGIES = ("random", "top_score", "top_model", "uncertainty", "hybrid")


def select_batch(
    strategy: str,
    batch_size: int,
    unsupervised_scores: np.ndarray,
    already_labeled: np.ndarray,
    predicted_proba: np.ndarray | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Elige las próximas `batch_size` transacciones a revisar, sin repetir las ya revisadas."""
    rng = np.random.default_rng(random_state)
    disponible = np.setdiff1d(np.arange(len(unsupervised_scores)), already_labeled)
    if len(disponible) <= batch_size:
        return disponible

    if strategy == "random":
        return rng.choice(disponible, size=batch_size, replace=False)

    # Sin modelo todavía, cualquier estrategia que dependa de él cae al score no supervisado:
    # es el arranque en frío, y es exactamente lo que los Módulos 2 a 6 dejan disponible.
    if predicted_proba is None or strategy == "top_score":
        orden = disponible[np.argsort(unsupervised_scores[disponible])[::-1]]
        return orden[:batch_size]

    if strategy == "top_model":
        return disponible[np.argsort(predicted_proba[disponible])[::-1]][:batch_size]

    if strategy == "uncertainty":
        incertidumbre = -np.abs(predicted_proba[disponible] - 0.5)
        return disponible[np.argsort(incertidumbre)[::-1]][:batch_size]

    if strategy == "hybrid":
        mitad = batch_size // 2
        por_score = disponible[np.argsort(predicted_proba[disponible])[::-1]][:mitad]
        resto = np.setdiff1d(disponible, por_score)
        incertidumbre = -np.abs(predicted_proba[resto] - 0.5)
        por_duda = resto[np.argsort(incertidumbre)[::-1]][: batch_size - mitad]
        return np.concatenate([por_score, por_duda])

    raise ValueError(f"Estrategia desconocida: {strategy}. Usar una de {list(STRATEGIES)}")


def run_active_learning(
    X_pool: np.ndarray,
    y_pool,
    X_test: np.ndarray,
    y_test,
    unsupervised_scores: np.ndarray,
    strategy: str,
    n_rounds: int = 8,
    batch_size: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simula el circuito de revisión ronda a ronda y mide qué se gana en cada una.

    Devuelve una fila por ronda con las etiquetas acumuladas, cuántas resultaron fraude y el
    PR-AUC alcanzado sobre el período tardío.
    """
    y_arr = np.asarray(y_pool)
    labeled = np.array([], dtype=int)
    proba_pool = None

    rows = []
    for ronda in range(1, n_rounds + 1):
        batch = select_batch(strategy, batch_size, unsupervised_scores, labeled,
                             proba_pool, random_state + ronda)
        labeled = np.concatenate([labeled, batch])

        X_train, y_train = build_training_set(X_pool, y_arr, labeled, random_state=random_state)
        model = fit_classifier(X_train, y_train, random_state)

        if model is None:
            rows.append({
                "estrategia": strategy, "ronda": ronda, "etiquetas": len(labeled),
                "fraudes_encontrados": int(y_arr[labeled].sum()), "pr_auc": np.nan,
            })
            continue

        proba_pool = model.predict_proba(X_pool)[:, 1]
        rows.append({
            "estrategia": strategy, "ronda": ronda, "etiquetas": len(labeled),
            "fraudes_encontrados": int(y_arr[labeled].sum()),
            "pr_auc": average_precision_score(y_test, model.predict_proba(X_test)[:, 1]),
        })
    return pd.DataFrame(rows)


def compare_strategies(
    X_pool: np.ndarray,
    y_pool,
    X_test: np.ndarray,
    y_test,
    unsupervised_scores: np.ndarray,
    strategies=STRATEGIES,
    **kwargs,
) -> pd.DataFrame:
    """Corre el circuito completo con cada estrategia y apila los resultados."""
    return pd.concat(
        [
            run_active_learning(X_pool, y_pool, X_test, y_test, unsupervised_scores, s, **kwargs)
            for s in strategies
        ],
        ignore_index=True,
    )

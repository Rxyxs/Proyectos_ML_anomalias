"""Combinación de detectores: ensembles de anomaly scores.

Ningún detector no supervisado gana en todos los tipos de anomalía —cada familia mira una
propiedad distinta de los datos—, y sin etiquetas no hay forma de elegir el mejor *antes*
de desplegarlo. La respuesta estándar en la literatura de outlier ensembles (Aggarwal &
Sathe) es no elegir: combinar los scores de varios detectores para quedarse con la señal
que comparten y diluir el ruido particular de cada uno.

El problema al combinar es que los scores no son comparables entre sí: la distancia de
Mahalanobis vive en otra escala que un error de reconstrucción o una log-verosimilitud.
Cada estrategia de aquí resuelve esa incompatibilidad de una forma distinta:

- `rank_average`: reemplaza cada score por su rango. Descarta la magnitud y conserva solo
  el orden, así que una cola pesada en un detector no arrastra el consenso.
- `zscore_average`: estandariza cada detector (media 0, desviación 1) y promedia. Conserva
  *cuánto* de anómalo es un punto, no solo su posición.
- `zscore_max`: estandariza y toma el máximo. Basta con que **un** detector grite fuerte
  para levantar la alerta — útil cuando cada tipo de fraude solo lo ve una familia.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

EPS = 1e-12


def _stack(scores_by_model: dict[str, np.ndarray]) -> np.ndarray:
    """Apila los scores en una matriz (n_muestras, n_detectores), validando longitudes."""
    if not scores_by_model:
        raise ValueError("Se requiere al menos un detector para combinar scores.")

    arrays = [np.asarray(s, dtype=float) for s in scores_by_model.values()]
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"Todos los detectores deben puntuar las mismas filas; largos={sorted(lengths)}")
    return np.column_stack(arrays)


def rank_average(scores_by_model: dict[str, np.ndarray]) -> np.ndarray:
    """Promedio de rangos normalizados a [0, 1]. Más alto = más anómalo."""
    stacked = _stack(scores_by_model)
    n = stacked.shape[0]
    # rankdata asigna el rango promedio a los empates, importante en detectores como HBOS
    # que producen scores discretos (todas las filas del mismo bin comparten valor).
    ranks = np.apply_along_axis(rankdata, 0, stacked) / n
    return ranks.mean(axis=1)


def zscore_average(scores_by_model: dict[str, np.ndarray]) -> np.ndarray:
    """Promedio de scores estandarizados por detector. Más alto = más anómalo."""
    return _standardize(_stack(scores_by_model)).mean(axis=1)


def zscore_max(scores_by_model: dict[str, np.ndarray]) -> np.ndarray:
    """Máximo de los scores estandarizados: basta un detector convencido. Más alto = más anómalo."""
    return _standardize(_stack(scores_by_model)).max(axis=1)


def _standardize(stacked: np.ndarray) -> np.ndarray:
    """Estandariza cada columna (detector) a media 0 y desviación 1."""
    return (stacked - stacked.mean(axis=0)) / (stacked.std(axis=0) + EPS)


ENSEMBLE_STRATEGIES = {
    "ensemble_rank_avg": rank_average,
    "ensemble_z_avg": zscore_average,
    "ensemble_z_max": zscore_max,
}


def build_ensembles(scores_by_model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Aplica las tres estrategias de combinación sobre el mismo conjunto de detectores."""
    return {name: strategy(scores_by_model) for name, strategy in ENSEMBLE_STRATEGIES.items()}

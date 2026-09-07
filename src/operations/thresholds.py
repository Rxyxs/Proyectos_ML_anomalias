"""Módulo 5 — dónde cortar el anomaly score.

Los módulos 2 a 4 producen un ranking y lo evalúan con PR-AUC y Precision@k. Ninguna de las
dos cosas es desplegable: en producción no llega un conjunto de prueba para ordenar, llega
una transacción y hay que decir *sí* o *no*. Eso exige un umbral, y elegirlo es un problema
distinto del de elegir el modelo.

Tres reglas, en orden creciente de información disponible:

1. `quantile_threshold` — no necesita ni una etiqueta. Como el ajuste se hace solo con
   transacciones normales, el cuantil (1-α) de los scores de entrenamiento deja fuera una
   fracción α de lo normal: **α es directamente la tasa de falsos positivos esperada**. Es
   la única regla aplicable el día que se despliega el sistema.
2. `capacity_threshold` — parte de la restricción real de un equipo de análisis: cuántas
   alertas por día se pueden revisar. Traduce esa capacidad al umbral que la produce.
3. `cost_optimal_threshold` — necesita etiquetas y montos. Busca el corte que maximiza el
   dinero neto salvado, que es el criterio que un negocio realmente optimiza.

La distancia entre lo que promete (1) y lo que rinde (3) es el resultado interesante: si el
cuantil de entrenamiento no reproduce la FPR prometida sobre datos posteriores, el sistema
está mal calibrado y hay que recalibrarlo periódicamente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.operations.costs import net_savings


def quantile_threshold(train_scores: np.ndarray, alpha: float = 0.01) -> float:
    """Umbral sin etiquetas: cuantil (1-alpha) de los scores de entrenamiento.

    `alpha` es la tasa de falsos positivos que se acepta sobre tráfico normal. El
    entrenamiento contiene solo transacciones normales, así que el cuantil empírico es una
    estimación directa de esa tasa — siempre que la distribución no cambie, que es
    exactamente lo que hay que verificar después.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha debe estar en (0, 1); se recibió {alpha}")
    return float(np.quantile(np.asarray(train_scores, dtype=float), 1 - alpha))


def empirical_alarm_rate(scores: np.ndarray, threshold: float) -> float:
    """Fracción de transacciones que superan el umbral: el volumen de alertas que genera."""
    return float((np.asarray(scores, dtype=float) >= threshold).mean())


def capacity_threshold(scores: np.ndarray, alerts_per_period: int, n_periods: int) -> float:
    """Umbral que produce, en promedio, `alerts_per_period` alertas por período.

    Invierte la restricción operativa: un equipo que revisa 100 casos por día no quiere
    saber qué FPR tolera, quiere el corte que le entrega 100 casos.
    """
    scores = np.asarray(scores, dtype=float)
    total_alerts = max(1, int(alerts_per_period * n_periods))
    if total_alerts >= len(scores):
        return float(scores.min())
    return float(np.partition(scores, -total_alerts)[-total_alerts])


def cost_optimal_threshold(
    y_true,
    scores: np.ndarray,
    amounts,
    review_cost: float,
    recovery_rate: float = 1.0,
    n_candidates: int = 200,
) -> dict:
    """Umbral que maximiza el ahorro neto, evaluando una grilla de cortes candidatos.

    Los candidatos son cuantiles del score en vez de valores equiespaciados: la distribución
    de scores es de cola muy pesada y una grilla lineal gastaría casi todos sus puntos en
    una región donde no hay ninguna transacción.
    """
    scores = np.asarray(scores, dtype=float)
    quantiles = np.linspace(0.5, 1.0, n_candidates, endpoint=False)
    candidates = np.unique(np.quantile(scores, quantiles))

    best = {"threshold": float(candidates[-1]), "net_savings": -np.inf, "n_alerts": 0}
    for threshold in candidates:
        flagged = scores >= threshold
        savings = net_savings(y_true, flagged, amounts, review_cost, recovery_rate)
        if savings > best["net_savings"]:
            best = {
                "threshold": float(threshold),
                "net_savings": float(savings),
                "n_alerts": int(flagged.sum()),
            }
    return best


def threshold_report(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test,
    alphas=(0.001, 0.005, 0.01, 0.05),
) -> pd.DataFrame:
    """Compara la FPR prometida por el cuantil de entrenamiento contra la observada en prueba.

    `alpha_observado` se mide **solo sobre las transacciones legítimas** de la prueba: es la
    definición de tasa de falsos positivos, y mezclar el fraude la contaminaría.
    """
    y_arr = np.asarray(y_test)
    test_scores = np.asarray(test_scores, dtype=float)
    legit_scores = test_scores[y_arr == 0]

    rows = []
    for alpha in alphas:
        threshold = quantile_threshold(train_scores, alpha)
        flagged = test_scores >= threshold
        detected = int(flagged[y_arr == 1].sum())
        rows.append({
            "alpha": alpha,
            "threshold": threshold,
            "alpha_observado": float((legit_scores >= threshold).mean()),
            "alertas": int(flagged.sum()),
            "fraude_detectado": detected,
            "recall": detected / max(1, int(y_arr.sum())),
            "precision": detected / max(1, int(flagged.sum())),
        })
    return pd.DataFrame(rows)

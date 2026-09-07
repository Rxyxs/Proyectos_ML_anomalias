"""Módulo 5 — métricas en pesos, no en conteo de transacciones.

Todo el repositorio evalúa hasta acá contando transacciones: PR-AUC, Precision@k, recall.
Esas métricas tratan por igual a un fraude de 10 mil y a uno de 10 millones, y sobre PaySim
esa equivalencia es indefendible: **el 10% de los fraudes más caros concentra el 53,7% del
monto defraudado**, y el fraude tiene un monto mediano ~6x el de una transacción legítima.

Un detector puede ganar en PR-AUC y perder en dinero salvado si acierta muchos casos
baratos y se le escapan los caros. Estas funciones miden lo segundo.

`recovery_rate` modela que detectar no es recuperar: una alerta que se dispara después de
que el dinero salió del sistema evita parte de la pérdida, no toda. Se deja explícito como
parámetro en vez de asumir 1.0 porque es una decisión de negocio, no del modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true, amounts) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y_true).astype(int)
    amount_arr = np.asarray(amounts, dtype=float)
    if y_arr.shape != amount_arr.shape:
        raise ValueError(
            f"y_true y amounts deben tener el mismo largo; {y_arr.shape} vs {amount_arr.shape}"
        )
    return y_arr, amount_arr


def value_weighted_recall(y_true, flagged, amounts) -> float:
    """Fracción del **monto** defraudado que cae dentro de las alertas.

    Es el análogo en pesos del recall: en vez de "cuántos fraudes atrapé", responde "cuánta
    de la plata en juego atrapé".
    """
    y_arr, amount_arr = _as_arrays(y_true, amounts)
    flagged_arr = np.asarray(flagged).astype(bool)

    total_fraud_amount = amount_arr[y_arr == 1].sum()
    if total_fraud_amount == 0:
        return 0.0
    caught = amount_arr[(y_arr == 1) & flagged_arr].sum()
    return float(caught / total_fraud_amount)


def net_savings(y_true, flagged, amounts, review_cost: float, recovery_rate: float = 1.0) -> float:
    """Dinero recuperado por las alertas correctas menos el costo de revisarlas todas.

    Cada alerta cuesta `review_cost` se confirme o no —el analista dedica el mismo tiempo a
    descartar un falso positivo— y cada fraude detectado recupera `recovery_rate` del monto.
    """
    y_arr, amount_arr = _as_arrays(y_true, amounts)
    flagged_arr = np.asarray(flagged).astype(bool)

    recovered = recovery_rate * amount_arr[(y_arr == 1) & flagged_arr].sum()
    return float(recovered - review_cost * flagged_arr.sum())


def savings_curve(
    y_true,
    scores: np.ndarray,
    amounts,
    review_cost: float,
    recovery_rate: float = 1.0,
    n_points: int = 60,
) -> pd.DataFrame:
    """Ahorro neto y recall en pesos a lo largo del ranking, de la alerta 1 a la N.

    Recorre presupuestos de revisión crecientes (cuántas de las transacciones más anómalas
    se revisan) y devuelve, para cada uno, qué se gana. El máximo de esa curva es el punto
    de operación óptimo; su forma dice cuán sensible es a equivocarse en la elección.
    """
    y_arr, amount_arr = _as_arrays(y_true, amounts)
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(scores)[::-1]
    budgets = np.unique(np.geomspace(1, len(scores), n_points).astype(int))

    total_fraud_amount = amount_arr[y_arr == 1].sum()
    rows = []
    for budget in budgets:
        top = order[:budget]
        caught_mask = y_arr[top] == 1
        caught_amount = amount_arr[top][caught_mask].sum()
        rows.append({
            "revisadas": int(budget),
            "fraudes_detectados": int(caught_mask.sum()),
            "recall": float(caught_mask.sum() / max(1, int(y_arr.sum()))),
            "recall_en_monto": float(caught_amount / total_fraud_amount) if total_fraud_amount else 0.0,
            "ahorro_neto": float(recovery_rate * caught_amount - review_cost * budget),
        })
    return pd.DataFrame(rows)


def break_even_review_cost(y_true, amounts, recovery_rate: float = 1.0) -> float:
    """Costo de revisión por encima del cual deja de convenir revisar una transacción al azar.

    Es la pérdida esperada por transacción: prevalencia x monto medio del fraude x tasa de
    recuperación. Si revisar cuesta menos que eso, el óptimo económico degenera en "revisar
    todo" y el umbral deja de ser una decisión de modelado — pasa a estar limitado por la
    capacidad del equipo, no por la economía. Conviene calcularlo antes de interpretar
    cualquier óptimo de ahorro neto.
    """
    y_arr, amount_arr = _as_arrays(y_true, amounts)
    if amount_arr.size == 0:
        return 0.0
    return float(recovery_rate * amount_arr[y_arr == 1].sum() / len(amount_arr))


def amount_concentration(y_true, amounts, top_fraction: float = 0.1) -> dict:
    """Cuánto del monto defraudado concentra la fracción más cara de los fraudes.

    Justifica por qué las métricas por conteo son insuficientes: si un décimo de los casos
    explica la mitad del dinero, el orden que importa no es el de "más anómalo" sino el de
    "más caro entre los anómalos".
    """
    y_arr, amount_arr = _as_arrays(y_true, amounts)
    fraud_amounts = amount_arr[y_arr == 1]
    if fraud_amounts.size == 0:
        return {"top_fraction": top_fraction, "share_of_amount": 0.0, "n_fraud": 0}

    n_top = max(1, int(len(fraud_amounts) * top_fraction))
    share = np.sort(fraud_amounts)[-n_top:].sum() / fraud_amounts.sum()
    return {
        "top_fraction": top_fraction,
        "share_of_amount": float(share),
        "n_fraud": int(len(fraud_amounts)),
        "median_fraud_amount": float(np.median(fraud_amounts)),
    }

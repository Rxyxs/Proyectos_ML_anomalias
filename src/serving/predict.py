"""Módulo 8 — la API de scoring: de una transacción a una alerta explicada.

Esta es la superficie que consumiría un sistema en producción. Toma un `DetectorPackage` y
transacciones ya limpias, y devuelve por cada una lo que un analista necesita para actuar:

- **score**: el anomaly score crudo, útil solo para ordenar entre sí;
- **p_valor**: la fracción del tráfico legítimo de calibración al menos tan anómala. A
  diferencia del score, tiene unidades interpretables y se puede comparar entre detectores;
- **alerta**: la decisión binaria, tomada con el umbral que el paquete promete;
- **motivo**: las features que sostienen la anomalía, por oclusión.

La diferencia entre `score` y `p_valor` es la que vuelve accionable al resultado. Un score de
14,3 no significa nada por sí solo; un p-valor de 0,0004 dice que menos de una transacción
legítima de cada dos mil se ve así de rara.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.conformal.conformal import conformal_p_values
from src.serving.explain import explain_rows
from src.serving.package import DetectorPackage


def score_transactions(package: DetectorPackage, X, top_k: int = 3,
                       explain: bool = True) -> pd.DataFrame:
    """Puntúa transacciones y devuelve una fila por cada una, lista para revisar.

    `explain=False` saltea la atribución, que cuesta una pasada de scoring por feature. Para
    puntuar millones de transacciones conviene explicar solo las que se van a revisar.
    """
    # Se escala una sola vez y se reutiliza. Llamar a score() y p_values() con el DataFrame
    # crudo volvería a pasar por el escalador en cada uno —tres escalados y dos scorings para
    # el mismo resultado—, lo que además falsearía la latencia que mide este módulo.
    X_scaled = package.to_scaled_matrix(X)
    scores = package.score_from_scaled(X_scaled)
    p_values = conformal_p_values(package.calibration_scores, scores)
    alertas = p_values <= package.alpha

    salida = pd.DataFrame({
        "score": scores,
        "p_valor": p_values,
        "alerta": alertas,
    }, index=X.index if isinstance(X, pd.DataFrame) else None)

    if not explain:
        return salida

    motivos = explain_rows(
        package.score_from_scaled,
        X_scaled, _baseline_from_package(package), package.feature_names, top_k=top_k,
    )
    salida["motivo"] = [
        ", ".join(f"{nombre} ({aporte:+.2f})" for nombre, aporte in fila) for fila in motivos
    ]
    return salida


def _baseline_from_package(package: DetectorPackage) -> np.ndarray:
    """Vector de valores típicos en el espacio escalado.

    `RobustScaler` centra en la mediana del entrenamiento, así que la mediana escalada es el
    vector cero. Se calcula a partir del escalador en vez de hardcodear ceros para que el
    módulo siga siendo correcto si alguien cambia el escalador por otro.
    """
    centro = getattr(package.scaler, "center_", None)
    if centro is None:
        return np.zeros(len(package.feature_names))
    return package.scaler.transform(np.asarray(centro).reshape(1, -1)).ravel()


def top_alerts(package: DetectorPackage, X, n: int = 10) -> pd.DataFrame:
    """Las `n` transacciones más anómalas, explicadas.

    Es la cola de revisión del Módulo 5: se explica solo lo que efectivamente se va a mirar.
    """
    p_values = package.p_values(X)
    orden = np.argsort(p_values)[:n]

    seleccion = X.iloc[orden] if isinstance(X, pd.DataFrame) else np.asarray(X)[orden]
    return score_transactions(package, seleccion, explain=True).sort_values("p_valor")


def alert_rate(package: DetectorPackage, X) -> float:
    """Fracción de transacciones que el paquete alertaría: el volumen que genera."""
    return float(package.predict(X).mean())

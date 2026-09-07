"""Módulo 6 — detección conforme: p-valores con garantía de tasa de falsas alarmas.

El Módulo 5 dejó un problema abierto. El umbral por cuantil promete que una fracción α del
tráfico legítimo va a superar el corte, y sobre PaySim esa promesa se rompe: Deep SVDD
promete 0,1% y entrega 35%. La pregunta natural es si existe una forma de fijar el umbral
que **garantice** la tasa de falsas alarmas en vez de estimarla y cruzar los dedos.

La respuesta de la literatura es la detección conforme. En vez de comparar el score contra
un cuantil, se lo compara contra un conjunto de calibración y se devuelve un p-valor:

    p(x) = (1 + #{scores de calibración >= score de x}) / (n_calibración + 1)

El +1 en numerador y denominador no es un ajuste cosmético: es lo que vuelve la garantía
válida en muestra finita, sin supuestos asintóticos. Si los datos de calibración y el punto
nuevo son **intercambiables**, entonces para un punto legítimo

    P(p(x) <= α) <= α    para todo α, en muestra finita.

O sea: rechazar cuando p <= α produce una tasa de falsos positivos de a lo sumo α, sin
suponer nada sobre la distribución de los datos ni sobre el detector. Funciona envolviendo
cualquiera de los quince detectores del repositorio.

Un matiz que conviene tener presente al leer los números: la garantía es **marginal**, es
decir promedia sobre el sorteo del conjunto de calibración. Fijado un conjunto concreto, la
tasa observada fluctúa alrededor de α en vez de quedar siempre por debajo — con 30.000
puntos de calibración esa fluctuación es de un ~10-20% relativo. Por eso los reportes de
abajo informan la razón observada/α y no un veredicto binario: una razón de 1,2 es ruido
muestral, una de 350 —la que produce el umbral por cuantil sobre Deep SVDD— es otra cosa.

**La letra chica está en "intercambiables".** La garantía no es incondicional: vale mientras
la distribución de calibración y la de producción sean la misma. La deriva temporal rompe
exactamente ese supuesto, y `run_conformal.py` mide qué pasa cuando se rompe. El aporte real
del método no es eliminar el supuesto, es volverlo **explícito y verificable**: se pasa de
"ojalá el cuantil siga sirviendo" a "la garantía vale si y solo si hay intercambiabilidad, y
esto es lo que ocurre cuando no la hay".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_p_values(calibration_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """P-valores conformes para cada score de prueba. Más bajo = más anómalo.

    Ambos conjuntos usan la convención del repositorio (más alto = más anómalo). El p-valor
    de un punto es la proporción de la calibración que resulta al menos tan anómala como él.
    """
    calibration = np.sort(np.asarray(calibration_scores, dtype=float))
    test = np.asarray(test_scores, dtype=float)
    n = calibration.size
    if n == 0:
        raise ValueError("El conjunto de calibración no puede estar vacío.")

    # searchsorted con 'left' da cuántos elementos son estrictamente menores; el resto son
    # los >= que la definición del p-valor necesita contar.
    n_greater_equal = n - np.searchsorted(calibration, test, side="left")
    return (1.0 + n_greater_equal) / (n + 1.0)


def conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Score a partir del cual el p-valor conforme cae por debajo de `alpha`.

    Equivale a rechazar por p-valor, pero deja un corte fijo aplicable transacción a
    transacción sin recalcular nada — que es como se despliega en producción.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha debe estar en (0, 1); se recibió {alpha}")
    calibration = np.sort(np.asarray(calibration_scores, dtype=float))
    n = calibration.size

    # El menor k tal que (1 + k) / (n + 1) <= alpha admite a lo sumo k scores de calibración
    # por encima del corte; ese corte es el (n - k)-ésimo valor ordenado.
    k = int(np.floor(alpha * (n + 1) - 1))
    if k < 0:
        return float(np.inf)
    return float(calibration[max(0, n - 1 - k)])


def coverage_report(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test,
    alphas=(0.001, 0.005, 0.01, 0.05),
) -> pd.DataFrame:
    """Compara la tasa de falsas alarmas garantizada contra la observada, por cada alpha.

    `fpr_observada` se mide solo sobre las transacciones legítimas — es la definición de tasa
    de falsos positivos, y es también la cantidad que la garantía conforme acota.

    `razon_fpr_alpha` es el cociente entre ambas. Cerca de 1 significa que la garantía se
    sostiene; valores de dos o tres órdenes de magnitud significan que la intercambiabilidad
    entre calibración y prueba no se cumple.
    """
    y_arr = np.asarray(y_test)
    p_values = conformal_p_values(calibration_scores, test_scores)
    legit = p_values[y_arr == 0]
    fraud_p = p_values[y_arr == 1]

    rows = []
    for alpha in alphas:
        flagged = p_values <= alpha
        detected = int((y_arr[flagged] == 1).sum())
        rows.append({
            "alpha": alpha,
            "fpr_observada": float((legit <= alpha).mean()),
            # Razón observada/α en vez de un booleano: la garantía es marginal sobre el
            # sorteo de la calibración, así que una razón levemente mayor que 1 es lo
            # esperable y no una violación.
            "razon_fpr_alpha": float((legit <= alpha).mean() / alpha),
            "alertas": int(flagged.sum()),
            "fraude_detectado": detected,
            "recall": detected / max(1, int(y_arr.sum())),
            "precision": detected / max(1, int(flagged.sum())),
            "p_valor_mediano_fraude": float(np.median(fraud_p)) if fraud_p.size else float("nan"),
        })
    return pd.DataFrame(rows)


class ConformalDetector:
    """Envuelve cualquier detector del repositorio y devuelve p-valores en vez de scores.

    Espera un detector ya construido que exponga `fit` y `score_samples` con la convención
    de scikit-learn. El ajuste y la calibración usan conjuntos **disjuntos**: si se calibrara
    con los mismos datos del ajuste se perdería la intercambiabilidad entre calibración y
    producción, que es lo único que sostiene la garantía.
    """

    def __init__(self, detector, alpha: float = 0.01):
        self.detector = detector
        self.alpha = alpha
        self.calibration_scores_: np.ndarray | None = None

    def fit(self, X_train, X_calibration) -> "ConformalDetector":
        from src.unsupervised.models import anomaly_score

        self.detector.fit(X_train)
        self.calibration_scores_ = anomaly_score(self.detector, X_calibration)
        return self

    def p_values(self, X) -> np.ndarray:
        from src.unsupervised.models import anomaly_score

        if self.calibration_scores_ is None:
            raise ValueError("Hay que llamar a fit() antes de calcular p-valores.")
        return conformal_p_values(self.calibration_scores_, anomaly_score(self.detector, X))

    def predict(self, X) -> np.ndarray:
        """Devuelve True para las transacciones cuyo p-valor cae por debajo de alpha."""
        return self.p_values(X) <= self.alpha

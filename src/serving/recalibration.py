"""Módulo 9 — recalibración continua y un monitor de deriva que no necesita etiquetas.

El Módulo 8 dejó el hallazgo operativo más serio del repositorio: el paquete promete 1% de
alertas y entrega 1,57% el día 0 y 11,10% el día 16. La carga de trabajo se multiplica por
siete en diecisiete días sin que nadie toque el modelo.

Las piezas para arreglarlo ya estaban sueltas. Los Módulos 5 y 6 localizaron la causa —la
escala del score se corre entre el período de ajuste y el de operación— y el Módulo 6 mostró
que el p-valor conforme cumple su garantía mientras la calibración y el tráfico sean
intercambiables. El Módulo 7 mostró que refrescar una ventana cuesta una pasada lineal. Este
módulo junta las dos cosas: el conjunto de calibración deja de ser fijo y avanza con el
tráfico reciente, y un monitor sin etiquetas avisa cuándo la garantía dejó de valer.

Dos decisiones de diseño que no son obvias:

- **La ventana se mide en transacciones, no en días.** El volumen de PaySim cae 2.111x hacia
  fin de mes; una ventana de "los últimos siete días" terminaría con unos pocos cientos de
  scores, y con n puntos de calibración el p-valor más chico que se puede resolver es
  1/(n+1). Una ventana por cantidad mantiene la resolución constante aunque el volumen caiga.
- **Recalibrar con tráfico sin etiquetar mete fraude en la calibración.** Las etiquetas llegan
  con semanas de retraso, así que la ventana se alimenta con todo. El fraude engorda la cola
  del conjunto, sube el umbral y se deja de alertar justo lo que había que alertar. La
  alternativa tentadora es excluir de la ventana lo que ya se alertó, pero eso recorta en cada
  actualización la cola legítima que define el umbral, y puede cerrar un lazo que se
  retroalimenta: umbral más bajo, más alertas, más recorte. `run_recalibration.py` mide las
  dos variantes en vez de suponer cuál conviene.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kstest

from src.conformal.conformal import conformal_p_values, conformal_threshold


def _as_nonempty(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} no puede estar vacío.")
    return arr


def tail_ratio(p_values, alpha: float) -> float:
    """Fracción del tráfico con p-valor <= alpha, dividida por alpha. No usa etiquetas.

    Bajo intercambiabilidad vale aproximadamente 1: es la tasa de alertas relativa a la
    prometida, así que un 3 significa que el sistema alerta el triple de lo que dijo.

    Tiene un sesgo que conviene conocer antes de usarlo como disparador. Se calcula sobre todo
    el tráfico, fraude incluido, y el fraude cae casi entero debajo de alpha. Si la prevalencia
    sube, la razón sube aunque la calibración de lo legítimo siga perfecta — y en PaySim la
    prevalencia diaria sube justamente a fin de mes, cuando el volumen se derrumba y el conteo
    de fraude se mantiene. El monitor mezcla deriva con prevalencia, y sin etiquetas no hay
    forma de separarlas.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha debe estar en (0, 1); se recibió {alpha}")
    p = _as_nonempty(p_values, "p_values")
    return float((p <= alpha).mean() / alpha)


def uniformity_gap(p_values) -> float:
    """Distancia de Kolmogorov-Smirnov entre los p-valores y la uniforme en [0, 1].

    Bajo intercambiabilidad los p-valores del tráfico legítimo son uniformes: es el contenido
    estadístico de la garantía conforme (Módulo 6). Esta distancia resume la forma completa del
    histograma en un número, mientras que `tail_ratio` mira solo la cola que determina las
    alertas. Pueden discrepar: el Módulo 6 mostró detectores con histogramas nada uniformes y
    cola izquierda correcta.
    """
    p = _as_nonempty(p_values, "p_values")
    return float(kstest(p, "uniform").statistic)


class RollingCalibrator:
    """Conjunto de calibración conforme que avanza con el tráfico.

    Mantiene los `window_size` scores más recientes. Cada `update` agrega los del período que
    acaba de terminar y descarta los más viejos. El orden importa: primero se puntúa un
    período contra la calibración vigente y **después** se lo agrega a la ventana; hacerlo al
    revés calibraría cada día con sus propios datos y la tasa de alertas saldría perfecta por
    construcción, sin medir nada.
    """

    def __init__(self, initial_scores, window_size: int = 30_000, alpha: float = 0.01,
                 exclude_alerts: bool = False):
        if window_size < 1:
            raise ValueError(f"window_size debe ser positivo; se recibió {window_size}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha debe estar en (0, 1); se recibió {alpha}")

        self.window_size = int(window_size)
        self.alpha = alpha
        self.exclude_alerts = exclude_alerts
        self.scores_ = _as_nonempty(initial_scores, "initial_scores")[-self.window_size:]

    @property
    def n_calibration(self) -> int:
        return int(self.scores_.size)

    @property
    def threshold(self) -> float:
        """Score a partir del cual se alerta con la calibración vigente."""
        return conformal_threshold(self.scores_, self.alpha)

    def p_values(self, scores) -> np.ndarray:
        """P-valores conformes contra la ventana vigente."""
        return conformal_p_values(self.scores_, np.asarray(scores, dtype=float))

    def update(self, new_scores) -> "RollingCalibrator":
        """Incorpora los scores de un período ya puntuado y descarta los más viejos.

        Con `exclude_alerts` se descartan antes los scores que la calibración vigente habría
        alertado. Reduce la contaminación por fraude y a la vez recorta la cola legítima, que
        es la que define el umbral.
        """
        nuevos = np.asarray(new_scores, dtype=float).ravel()
        if self.exclude_alerts and nuevos.size:
            nuevos = nuevos[self.p_values(nuevos) > self.alpha]
        if nuevos.size:
            self.scores_ = np.concatenate([self.scores_, nuevos])[-self.window_size:]
        return self

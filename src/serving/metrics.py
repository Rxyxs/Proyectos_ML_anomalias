"""Telemetría Prometheus de la capa de serving (Módulo 8): cuántas veces se
llamó a cada método de `DetectorPackage`, con qué latencia y con qué resultado
-- y la distribución de los scores que efectivamente salieron, para poder
notar en producción cuándo esa distribución se corre (drift de scores).

Tres métricas, cada una con su propio propósito:

- `ANOMALY_PREDICT_REQUESTS_TOTAL`: cuenta invocaciones por detector y por
  resultado (`success`, `rejected_invalid_input`, `error`) -- el primer lugar
  donde mirar si empiezan a llegar rechazos.
- `ANOMALY_PREDICT_LATENCY_SECONDS`: histograma de latencia por detector, con
  buckets en el rango sub-milisegundo porque los detectores lineales (HBOS,
  ECOD, MAD-z, PCA) puntúan en centésimas de milisegundo -- con buckets
  pensados para llamadas de red (10ms+) toda esa señal caería en el primer
  bucket.
- `ANOMALY_SCORES_DISTRIBUTION`: histograma de los valores de score crudo
  emitidos. No es una medida de latencia ni de éxito/fracaso: es la señal de
  negocio para detectar cuándo la distribución de scores en producción se
  aleja de la vista en calibración.

El decorador `_instrument` envuelve los métodos de `DetectorPackage`
(src/serving/package.py) que efectivamente puntúan una transacción.
"""
from __future__ import annotations

import functools
import time

from prometheus_client import Counter, Histogram

ANOMALY_PREDICT_REQUESTS_TOTAL = Counter(
    "anomaly_predict_requests_total",
    "Invocaciones a la capa de scoring de DetectorPackage, por detector y resultado.",
    labelnames=("detector_name", "status"),
)

ANOMALY_PREDICT_LATENCY_SECONDS = Histogram(
    "anomaly_predict_latency_seconds",
    "Latencia de cada invocación a la capa de scoring, por detector.",
    labelnames=("detector_name",),
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
)

ANOMALY_SCORES_DISTRIBUTION = Histogram(
    "anomaly_scores_distribution",
    "Distribucion de los anomaly scores crudos emitidos, por detector -- para monitorear drift.",
    labelnames=("detector_name",),
)

# Estas dos son del batcher (src/serving/batcher.py::DynamicBatcher). Las
# invocaciones reales a score()/predict() que dispara un lote ya quedan
# instrumentadas por ANOMALY_PREDICT_* de arriba (una observación por lote,
# no por fila -- es justamente la señal de que el batching está reduciendo
# la cantidad de llamadas); estas dos métricas son sobre el COMPORTAMIENTO
# del batcher en sí, no sobre el detector.
ANOMALY_BATCH_SIZE_HISTOGRAM = Histogram(
    "anomaly_batch_size",
    "Cantidad de filas por lote efectivamente despachado, por detector.",
    labelnames=("detector_name",),
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

ANOMALY_BATCH_WAIT_TIME_SECONDS = Histogram(
    "anomaly_batch_wait_time_seconds",
    "Tiempo que cada solicitud individual esperó en cola antes de que se despachara su lote.",
    labelnames=("detector_name",),
    buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1),
)


def _detector_name(self) -> str:
    return str(self.metadata.get("detector", "desconocido"))


def _classify(exc: BaseException) -> str:
    from src.unsupervised.models import NonFiniteInputError

    return "rejected_invalid_input" if isinstance(exc, NonFiniteInputError) else "error"


def instrument(record_scores: bool = False):
    """Decorador para un método de `DetectorPackage`.

    Mide la latencia de la llamada, incrementa `ANOMALY_PREDICT_REQUESTS_TOTAL`
    con el resultado (éxito, rechazo por NaN/Inf, u otro error) y, si
    `record_scores=True` (para `score`/`score_from_scaled`, que devuelven un
    score por fila), observa cada score emitido en `ANOMALY_SCORES_DISTRIBUTION`.
    Siempre re-lanza la excepción original: la métrica se registra, no se
    esconde el fallo al llamador.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            detector_name = _detector_name(self)
            inicio = time.perf_counter()
            try:
                resultado = func(self, *args, **kwargs)
            except Exception as exc:
                elapsed = time.perf_counter() - inicio
                status = _classify(exc) if isinstance(exc, ValueError) else "error"
                ANOMALY_PREDICT_REQUESTS_TOTAL.labels(detector_name, status).inc()
                ANOMALY_PREDICT_LATENCY_SECONDS.labels(detector_name).observe(elapsed)
                raise
            elapsed = time.perf_counter() - inicio
            ANOMALY_PREDICT_REQUESTS_TOTAL.labels(detector_name, "success").inc()
            ANOMALY_PREDICT_LATENCY_SECONDS.labels(detector_name).observe(elapsed)
            if record_scores:
                histograma = ANOMALY_SCORES_DISTRIBUTION.labels(detector_name)
                for valor in resultado:
                    histograma.observe(float(valor))
            return resultado

        return wrapper

    return decorator

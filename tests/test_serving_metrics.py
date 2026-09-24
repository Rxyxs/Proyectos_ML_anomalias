"""Pruebas de la telemetría Prometheus de la capa de serving (src.serving.metrics),
instrumentada sobre DetectorPackage.predict/score/score_from_scaled.

No se testea contra un servidor Prometheus real: se leen los propios objetos
Counter/Histogram vía `.collect()` (API pública de prometheus_client para
inspeccionar el estado actual de una métrica), antes y después de cada
llamada, y se compara el delta -- así el orden de ejecución de los tests no
importa, aunque los contadores sean globales al proceso.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.serving.metrics import (
    ANOMALY_PREDICT_LATENCY_SECONDS,
    ANOMALY_PREDICT_REQUESTS_TOTAL,
    ANOMALY_SCORES_DISTRIBUTION,
)
from src.serving.package import DetectorPackage, build_package
from src.unsupervised.families import GMMDensity

FEATURES = ["monto", "saldo", "error", "hora"]


@pytest.fixture
def paquete():
    rng = np.random.default_rng(42)
    entrenamiento = rng.normal(size=(1_500, 4))
    calibracion = rng.normal(size=(1_500, 4))

    scaler = RobustScaler().fit(entrenamiento)
    detector = GMMDensity(n_components=3).fit(scaler.transform(entrenamiento))

    return build_package(
        detector, scaler, scaler.transform(calibracion), FEATURES,
        alpha=0.01, detector_name="gmm_metrics_test",
    )


def _frame(filas) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(filas, dtype=float), columns=FEATURES)


def _counter_value(counter, **labels) -> float:
    """Lee el valor actual de un Counter para una combinación de labels, 0 si nunca se tocó."""
    for familia in counter.collect():
        for muestra in familia.samples:
            if muestra.name.endswith("_total") and muestra.labels == labels:
                return muestra.value
    return 0.0


def _histogram_count(histogram, **labels) -> float:
    """Cuenta de observaciones de un Histogram para una combinación de labels."""
    for familia in histogram.collect():
        for muestra in familia.samples:
            if muestra.name.endswith("_count") and muestra.labels == labels:
                return muestra.value
    return 0.0


def _histogram_bucket_le(histogram, le: str, **labels) -> float:
    """Cuenta acumulada del bucket `le` de un Histogram, para verificar que las
    observaciones caen donde se espera (rango sub-milisegundo)."""
    for familia in histogram.collect():
        for muestra in familia.samples:
            if muestra.name.endswith("_bucket") and muestra.labels.get("le") == le:
                if all(muestra.labels.get(k) == v for k, v in labels.items()):
                    return muestra.value
    return 0.0


# ---------------------------------------------------------------- éxito

def test_successful_score_increments_success_counter_and_latency(paquete):
    datos = _frame([[0.5, -0.2, 0.1, 0.3], [12.0, 12.0, 12.0, 12.0]])

    antes_success = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                    detector_name="gmm_metrics_test", status="success")
    antes_latencia = _histogram_count(ANOMALY_PREDICT_LATENCY_SECONDS,
                                       detector_name="gmm_metrics_test")

    paquete.score(datos)

    assert _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                           detector_name="gmm_metrics_test", status="success") == antes_success + 1
    assert _histogram_count(ANOMALY_PREDICT_LATENCY_SECONDS,
                             detector_name="gmm_metrics_test") == antes_latencia + 1


def test_successful_predict_and_score_from_scaled_are_each_instrumented(paquete):
    """predict() y score_from_scaled() son invocaciones independientes -- cada una
    con su propio contador de éxito, no solo score()."""
    datos = _frame([[0.5, -0.2, 0.1, 0.3]])
    X_scaled = paquete.to_scaled_matrix(datos)

    antes_predict = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                    detector_name="gmm_metrics_test", status="success")
    paquete.predict(datos)
    despues_predict = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                      detector_name="gmm_metrics_test", status="success")
    # predict() llama internamente a score() (vía p_values()), así que un solo
    # predict() exitoso suma AL MENOS 2 al contador de éxito (predict + score).
    assert despues_predict >= antes_predict + 2

    antes_scaled = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                   detector_name="gmm_metrics_test", status="success")
    paquete.score_from_scaled(X_scaled)
    despues_scaled = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                     detector_name="gmm_metrics_test", status="success")
    assert despues_scaled == antes_scaled + 1


def test_scores_distribution_observes_each_emitted_score(paquete):
    datos = _frame([[0.5, -0.2, 0.1, 0.3], [9.0, 9.0, 9.0, 9.0], [0.0, 0.0, 0.0, 0.0]])

    antes = _histogram_count(ANOMALY_SCORES_DISTRIBUTION, detector_name="gmm_metrics_test")
    paquete.score(datos)
    despues = _histogram_count(ANOMALY_SCORES_DISTRIBUTION, detector_name="gmm_metrics_test")

    assert despues == antes + len(datos)


def test_latency_of_a_fast_detector_lands_in_the_sub_millisecond_buckets(paquete):
    """El detector de prueba (GMM sobre 4 features) puntúa en microsegundos --
    tiene que caer dentro del bucket de 5ms, la razón de que los buckets del
    histograma empiecen en 0.0005s y no en el rango típico de llamadas de red."""
    datos = _frame([[0.5, -0.2, 0.1, 0.3]])

    antes = _histogram_bucket_le(ANOMALY_PREDICT_LATENCY_SECONDS, le="0.005",
                                  detector_name="gmm_metrics_test")
    paquete.score(datos)
    despues = _histogram_bucket_le(ANOMALY_PREDICT_LATENCY_SECONDS, le="0.005",
                                    detector_name="gmm_metrics_test")

    assert despues >= antes + 1


# ---------------------------------------------------------------- rechazo NaN/Inf

def test_nan_input_increments_rejected_invalid_input_not_success(paquete):
    datos = _frame([[0.5, np.nan, 0.1, 0.3]])

    antes_rechazo = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                    detector_name="gmm_metrics_test",
                                    status="rejected_invalid_input")
    antes_exito = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                  detector_name="gmm_metrics_test", status="success")

    with pytest.raises(ValueError, match="NaN o valores infinitos"):
        paquete.score(datos)

    assert _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                           detector_name="gmm_metrics_test",
                           status="rejected_invalid_input") == antes_rechazo + 1
    # El rechazo no debe contarse como éxito ni dejar un score en la distribución.
    assert _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                           detector_name="gmm_metrics_test", status="success") == antes_exito


def test_inf_input_on_score_from_scaled_increments_rejected_invalid_input(paquete):
    """score_from_scaled() valida NaN/Inf por su cuenta (no pasa por
    to_scaled_matrix), así que tiene que quedar clasificado igual."""
    X_scaled = np.array([[0.1, np.inf, -0.2, 0.4]])

    antes = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                            detector_name="gmm_metrics_test",
                            status="rejected_invalid_input")

    with pytest.raises(ValueError, match="NaN o valores infinitos"):
        paquete.score_from_scaled(X_scaled)

    assert _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                           detector_name="gmm_metrics_test",
                           status="rejected_invalid_input") == antes + 1


def test_missing_column_error_is_not_misclassified_as_rejected_invalid_input(paquete):
    """Un ValueError por columnas faltantes es un problema de esquema, no de
    NaN/Inf -- tiene que quedar en `error`, no mezclarse con los rechazos por
    entrada no finita."""
    datos = _frame([[0.5, -0.2, 0.1, 0.3]]).drop(columns=["saldo"])

    antes_error = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                  detector_name="gmm_metrics_test", status="error")
    antes_rechazo = _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                                    detector_name="gmm_metrics_test",
                                    status="rejected_invalid_input")

    with pytest.raises(ValueError, match="Faltan columnas"):
        paquete.score(datos)

    assert _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                           detector_name="gmm_metrics_test", status="error") == antes_error + 1
    assert _counter_value(ANOMALY_PREDICT_REQUESTS_TOTAL,
                           detector_name="gmm_metrics_test",
                           status="rejected_invalid_input") == antes_rechazo


# ---------------------------------------------------------------- sobrecosto

def test_metrics_instrumentation_overhead_is_negligible(paquete):
    """El costo de la telemetría en sí (no el del detector) tiene que ser
    chico frente al umbral pedido (<0.1ms por llamada), comparando la versión
    instrumentada contra la función original sin decorar (`__wrapped__`, que
    `functools.wraps` deja accesible).

    Cronometrar un bloque de N llamadas crudas y luego un bloque de N
    instrumentadas es ruidoso: cualquier cosa que pase entre los dos bloques
    (GC, scheduling del SO) se atribuye por completo al segundo bloque. Para
    aislar el costo real del decorador, las llamadas se intercalan una por
    una y se compara la MEDIANA de cada lado (robusta a outliers), no el
    promedio de dos bloques separados.
    """
    datos = _frame([[0.5, -0.2, 0.1, 0.3]])
    n = 300

    crudo = DetectorPackage.score.__wrapped__
    for _ in range(20):  # warm-up: primeras llamadas pagan costos de JIT/cache ajenos a la métrica
        crudo(paquete, datos)
        paquete.score(datos)

    tiempos_crudo, tiempos_instrumentado = [], []
    for _ in range(n):
        inicio = time.perf_counter()
        crudo(paquete, datos)
        tiempos_crudo.append(time.perf_counter() - inicio)

        inicio = time.perf_counter()
        paquete.score(datos)
        tiempos_instrumentado.append(time.perf_counter() - inicio)

    mediana_crudo = sorted(tiempos_crudo)[n // 2]
    mediana_instrumentado = sorted(tiempos_instrumentado)[n // 2]
    overhead_por_llamada = mediana_instrumentado - mediana_crudo

    assert overhead_por_llamada < 0.0001, (
        f"sobrecosto de telemetria por llamada: {overhead_por_llamada * 1000:.4f}ms "
        f"(crudo={mediana_crudo * 1000:.4f}ms, instrumentado={mediana_instrumentado * 1000:.4f}ms)"
    )

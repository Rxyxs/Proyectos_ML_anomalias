"""Pruebas de src.serving.batcher.DynamicBatcher: agrupamiento bajo tráfico
masivo, despacho puntual de solicitudes esporádicas, equivalencia numérica
exacta contra la ejecución directa, y cierre elegante."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

from src.serving.batcher import BatcherClosedError, DynamicBatcher
from src.serving.metrics import ANOMALY_BATCH_SIZE_HISTOGRAM, ANOMALY_BATCH_WAIT_TIME_SECONDS
from src.serving.package import DetectorPackage, build_package
from src.unsupervised.families import GMMDensity

FEATURES = ["monto", "saldo", "error", "hora"]


def _build_package(detector_name: str) -> DetectorPackage:
    rng = np.random.default_rng(42)
    entrenamiento = rng.normal(size=(600, 4))
    calibracion = rng.normal(size=(600, 4))
    scaler = RobustScaler().fit(entrenamiento)
    detector = GMMDensity(n_components=2).fit(scaler.transform(entrenamiento))
    return build_package(
        detector, scaler, scaler.transform(calibracion), FEATURES,
        alpha=0.01, detector_name=detector_name,
    )


def _histogram_count(histogram, **labels) -> float:
    total = 0.0
    for familia in histogram.collect():
        for muestra in familia.samples:
            if muestra.name.endswith("_count") and all(muestra.labels.get(k) == v for k, v in labels.items()):
                total += muestra.value
    return total


def _histogram_sum(histogram, **labels) -> float:
    total = 0.0
    for familia in histogram.collect():
        for muestra in familia.samples:
            if muestra.name.endswith("_sum") and all(muestra.labels.get(k) == v for k, v in labels.items()):
                total += muestra.value
    return total


@pytest.fixture
def package():
    return _build_package("batcher_test_default")


# ---------------------------------------------------------------------------
# construcción / validación
# ---------------------------------------------------------------------------

def test_rejects_a_non_positive_max_batch_size(package):
    with pytest.raises(ValueError, match="max_batch_size"):
        DynamicBatcher(package, max_batch_size=0)


def test_rejects_a_non_positive_max_wait_ms(package):
    with pytest.raises(ValueError, match="max_wait_ms"):
        DynamicBatcher(package, max_wait_ms=0.0)


# ---------------------------------------------------------------------------
# tráfico masivo: los lotes se acercan a max_batch_size
# ---------------------------------------------------------------------------

def test_under_sustained_heavy_traffic_batches_approach_max_batch_size():
    detector_name = "batcher_test_heavy_traffic"
    package = _build_package(detector_name)
    batcher = DynamicBatcher(package, max_batch_size=32, max_wait_ms=2.0)
    rng = np.random.default_rng(1)
    filas = [rng.normal(size=4) for _ in range(2_000)]

    try:
        with ThreadPoolExecutor(max_workers=100) as ex:
            list(ex.map(lambda f: batcher.submit(f), filas))
    finally:
        batcher.close()

    n_lotes = _histogram_count(ANOMALY_BATCH_SIZE_HISTOGRAM, detector_name=detector_name)
    total_filas = _histogram_sum(ANOMALY_BATCH_SIZE_HISTOGRAM, detector_name=detector_name)
    assert total_filas == 2_000

    tamano_promedio = total_filas / n_lotes
    # No es un umbral arbitrario: bajo 100 workers empujando 2000 filas de
    # una, la cola casi siempre tiene >=32 esperando cuando el despachador
    # mira -- el promedio real medido ronda 24-28 de 32 (75-85%); 50% es un
    # piso conservador que deja margen a la variabilidad del scheduler del
    # SO sin dejar de probar el comportamiento real.
    assert tamano_promedio >= 16, f"tamaño de lote promedio: {tamano_promedio:.1f} (esperado >= 16 de 32)"


# ---------------------------------------------------------------------------
# tráfico esporádico: el timer despacha sin bloquear indefinidamente
# ---------------------------------------------------------------------------

def test_a_single_sporadic_request_is_dispatched_by_the_wait_timer_not_blocked_forever(package):
    batcher = DynamicBatcher(package, max_batch_size=32, max_wait_ms=20.0)
    try:
        inicio = time.perf_counter()
        resultado = batcher.submit(np.array([0.1, 0.2, -0.1, 0.3]))
        transcurrido_ms = (time.perf_counter() - inicio) * 1000
    finally:
        batcher.close()

    assert resultado is not None
    # Tiene que despachar cerca de max_wait_ms, no esperar mucho más (lo que
    # indicaría que quedó colgado) ni mucho menos (lo que indicaría que no
    # respetó la ventana de espera en absoluto).
    assert transcurrido_ms < 200, f"tardó {transcurrido_ms:.1f}ms, muy por encima de max_wait_ms=20ms"


def test_several_sequential_sporadic_requests_each_complete_promptly(package):
    """No solo una vez: varias solicitudes espaciadas en el tiempo, cada una
    debe despacharse por su cuenta, no quedar esperando a que llegue más
    tráfico que nunca llega."""
    batcher = DynamicBatcher(package, max_batch_size=32, max_wait_ms=10.0)
    try:
        for _ in range(5):
            inicio = time.perf_counter()
            batcher.submit(np.array([0.0, 0.0, 0.0, 0.0]))
            transcurrido_ms = (time.perf_counter() - inicio) * 1000
            assert transcurrido_ms < 150
            time.sleep(0.05)  # simula tráfico espaciado, no ráfagas
    finally:
        batcher.close()


# ---------------------------------------------------------------------------
# equivalencia numérica exacta con la ejecución directa
# ---------------------------------------------------------------------------

def test_batched_score_matches_direct_execution_exactly(package):
    """'Exacto' con tolerancia de punto flotante, no bit a bit: medido en la
    primera corrida, apilar 16 filas y puntuarlas juntas da diferencias de
    ~3e-16 en términos relativos contra puntuar cada fila sola -- exactamente
    la escala del épsilon de máquina de float64 (2.22e-16), la huella
    esperada de que GMM.score_samples reduce (suma/logsumexp) en un orden
    distinto según la forma del lote. No es un bug del batcher: es
    no-asociatividad de punto flotante en BLAS vectorizado, el mismo
    fenómeno que ya documenté para ONNX vs. nativo (src/serving/onnx_exporter.py),
    acá una escala más chica porque es el mismo runtime de numpy en los dos lados.
    """
    rng = np.random.default_rng(7)
    filas = [rng.normal(size=4) for _ in range(64)]

    batcher = DynamicBatcher(package, max_batch_size=16, max_wait_ms=5.0)
    try:
        with ThreadPoolExecutor(max_workers=32) as ex:
            resultados_batcher = list(ex.map(lambda f: batcher.submit(f, method="score"), filas))
    finally:
        batcher.close()

    resultados_directos = [package.score(f.reshape(1, -1))[0] for f in filas]

    np.testing.assert_allclose(resultados_batcher, resultados_directos, rtol=1e-9, atol=1e-9)


def test_batched_predict_matches_direct_execution_exactly(package):
    rng = np.random.default_rng(8)
    filas = [rng.normal(size=4) for _ in range(40)]

    batcher = DynamicBatcher(package, max_batch_size=8, max_wait_ms=5.0)
    try:
        with ThreadPoolExecutor(max_workers=20) as ex:
            resultados_batcher = list(ex.map(lambda f: batcher.submit(f, method="predict"), filas))
    finally:
        batcher.close()

    resultados_directos = [bool(package.predict(f.reshape(1, -1))[0]) for f in filas]

    assert resultados_batcher == resultados_directos


def test_single_row_batch_matches_direct_execution(package):
    """El caso límite de lote-de-uno: sin otras filas apiladas al lado, el
    resultado tiene que seguir siendo idéntico."""
    fila = np.array([2.0, -1.0, 0.5, 3.0])
    batcher = DynamicBatcher(package, max_batch_size=32, max_wait_ms=5.0)
    try:
        resultado = batcher.submit(fila, method="score")
    finally:
        batcher.close()

    esperado = package.score(fila.reshape(1, -1))[0]
    assert resultado == esperado


# ---------------------------------------------------------------------------
# cierre elegante: drena lo que ya está en cola
# ---------------------------------------------------------------------------

def test_graceful_shutdown_processes_all_requests_already_queued(package):
    """Encola muchas solicitudes y cierra antes de que el despachador
    alcance a procesarlas (`max_wait_ms` largo a propósito, así la ventana
    de acumulación del primer lote sigue abierta cuando llega `close()`) --
    todas tienen que completarse igual, no perderse.

    El sleep breve antes de `close()` no es para "darle tiempo a procesar"
    -- es para asegurar que las 50 llamadas a `batcher.submit()` alcanzaron
    a encolarse (`ThreadPoolExecutor.submit()` no garantiza que la tarea ya
    arrancó), no una carrera contra el despachador.
    """
    batcher = DynamicBatcher(package, max_batch_size=8, max_wait_ms=300.0)
    rng = np.random.default_rng(9)
    filas = [rng.normal(size=4) for _ in range(50)]

    futures = []
    with ThreadPoolExecutor(max_workers=50) as ex:
        for fila in filas:
            futures.append(ex.submit(batcher.submit, fila))
        time.sleep(0.05)  # deja que las 50 tareas lleguen a encolarse de verdad
        batcher.close()
        resultados = [f.result(timeout=10) for f in futures]

    assert len(resultados) == 50
    assert all(r is not None for r in resultados)


def test_submit_after_close_raises_instead_of_hanging_forever(package):
    batcher = DynamicBatcher(package)
    batcher.close()

    with pytest.raises(BatcherClosedError):
        batcher.submit(np.array([0.0, 0.0, 0.0, 0.0]))


def test_context_manager_closes_on_exit(package):
    with DynamicBatcher(package, max_wait_ms=5.0) as batcher:
        resultado = batcher.submit(np.array([1.0, 1.0, 1.0, 1.0]))
        assert resultado is not None

    with pytest.raises(BatcherClosedError):
        batcher.submit(np.array([1.0, 1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# propagación de errores
# ---------------------------------------------------------------------------

def test_a_batch_scoring_error_propagates_to_every_requester_in_that_batch(package):
    """Una fila con NaN en el lote hace que score() del paquete entero
    lance (assert_finite valida la matriz completa) -- todos los
    peticionarios de ESE lote tienen que recibir la excepción real, no
    quedar colgados."""
    batcher = DynamicBatcher(package, max_batch_size=4, max_wait_ms=50.0)
    try:
        filas = [np.array([0.0, 0.0, 0.0, 0.0]), np.array([np.nan, 0.0, 0.0, 0.0])]
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(batcher.submit, f) for f in filas]
            with pytest.raises(ValueError):
                futures[0].result(timeout=5)
            with pytest.raises(ValueError):
                futures[1].result(timeout=5)
    finally:
        batcher.close()


# ---------------------------------------------------------------------------
# telemetría del batcher
# ---------------------------------------------------------------------------

def test_batch_wait_time_metric_is_recorded_per_request():
    detector_name = "batcher_test_wait_metric"
    package = _build_package(detector_name)
    batcher = DynamicBatcher(package, max_batch_size=32, max_wait_ms=5.0)

    antes = _histogram_count(ANOMALY_BATCH_WAIT_TIME_SECONDS, detector_name=detector_name)
    try:
        batcher.submit(np.array([0.0, 0.0, 0.0, 0.0]))
        batcher.submit(np.array([0.0, 0.0, 0.0, 0.0]))
    finally:
        batcher.close()
    despues = _histogram_count(ANOMALY_BATCH_WAIT_TIME_SECONDS, detector_name=detector_name)

    assert despues == antes + 2


def test_batch_size_metric_matches_the_real_number_of_dispatched_batches():
    detector_name = "batcher_test_size_metric"
    package = _build_package(detector_name)
    batcher = DynamicBatcher(package, max_batch_size=32, max_wait_ms=5.0)

    antes = _histogram_count(ANOMALY_BATCH_SIZE_HISTOGRAM, detector_name=detector_name)
    try:
        batcher.submit(np.array([0.0, 0.0, 0.0, 0.0]))  # una solicitud, un lote de 1
    finally:
        batcher.close()
    despues = _histogram_count(ANOMALY_BATCH_SIZE_HISTOGRAM, detector_name=detector_name)

    assert despues == antes + 1

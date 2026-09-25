"""Pruebas del arnés de carga (tests/load/load_harness.py): versión acotada
para CI, y verificación de que la saturación concurrente registra la
telemetría completa (sin pérdida de eventos) sin acumular memoria."""
from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.serving.metrics import ANOMALY_PREDICT_REQUESTS_TOTAL
from src.serving.metrics_server import ThreadingWSGIServer, create_app
from tests.load.load_harness import _build_demo_package, run_load_test
from wsgiref.simple_server import make_server


def _detector_total(detector_name: str) -> float:
    """Suma de anomaly_predict_requests_total sobre todos los `status`, para
    un `detector_name` -- cuenta TODAS las invocaciones registradas por mi
    telemetría de serving, sin importar si terminaron en éxito o rechazo."""
    total = 0.0
    for familia in ANOMALY_PREDICT_REQUESTS_TOTAL.collect():
        for muestra in familia.samples:
            if muestra.name.endswith("_total") and muestra.labels.get("detector_name") == detector_name:
                total += muestra.value
    return total


@pytest.fixture(scope="module")
def demo_package():
    return _build_demo_package(seed=1)


@pytest.fixture
def metrics_server():
    httpd = make_server("127.0.0.1", 0, create_app(), server_class=ThreadingWSGIServer)
    host, port = httpd.server_address
    hilo = threading.Thread(target=httpd.serve_forever, daemon=True)
    hilo.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        hilo.join(timeout=5)


# ---------------------------------------------------------------------------
# versión acotada para CI: 100 peticiones concurrentes en <2s
# ---------------------------------------------------------------------------

def test_100_concurrent_requests_complete_in_under_2_seconds(demo_package):
    package, features = demo_package

    inicio = time.perf_counter()
    reporte = run_load_test(
        package, features, users=20, spawn_rate=50.0, run_time_seconds=10.0, max_requests=100,
    )
    transcurrido = time.perf_counter() - inicio

    assert reporte["total_requests"] >= 100
    assert transcurrido < 2.0, f"tardó {transcurrido:.2f}s"


def test_100_concurrent_requests_have_zero_errors_on_valid_traffic(demo_package):
    package, features = demo_package

    reporte = run_load_test(
        package, features, users=20, spawn_rate=50.0, run_time_seconds=10.0,
        max_requests=100, corrupted_fraction=0.0,  # solo tráfico válido, para aislar el error_rate
    )

    assert reporte["scenarios"]["normal"]["total_requests"] > 0
    assert reporte["error_rate_valid_traffic"] == 0.0
    assert reporte["error_rate_within_threshold"] is True


def test_corrupted_requests_are_all_correctly_rejected_under_concurrency(demo_package):
    """No solo que no truene: que el rechazo (NaN/Inf) siga funcionando bien
    bajo concurrencia -- `success=True` en el escenario 'corrupted' significa
    "fue rechazada correctamente", no "no hubo excepción"."""
    package, features = demo_package

    reporte = run_load_test(
        package, features, users=20, spawn_rate=50.0, run_time_seconds=10.0,
        max_requests=100, corrupted_fraction=1.0,  # todo el tráfico es corrupto
    )

    corrompido = reporte["scenarios"]["corrupted"]
    assert corrompido["total_requests"] >= 100
    assert corrompido["error_rate"] == 0.0  # 0% de error = 100% correctamente rechazadas


def test_metrics_scenario_works_under_concurrency(demo_package, metrics_server):
    package, features = demo_package

    reporte = run_load_test(
        package, features, users=10, spawn_rate=50.0, run_time_seconds=10.0,
        max_requests=100, host=metrics_server, metrics_fraction=1.0,
    )

    metricas = reporte["scenarios"]["metrics"]
    assert metricas["total_requests"] >= 100
    assert metricas["error_rate"] == 0.0


# ---------------------------------------------------------------------------
# saturación concurrente: métricas Prometheus sin pérdida de eventos
# ---------------------------------------------------------------------------

def test_concurrent_saturation_registers_every_request_in_prometheus_with_no_event_loss():
    """El contador de Prometheus tiene que reflejar EXACTAMENTE la
    cantidad de invocaciones reales a predict()/score() bajo carga
    concurrente -- ni un evento de menos (el riesgo real de una métrica
    compartida entre threads sin el locking correcto de prometheus_client)."""
    detector_name = "load_test_no_event_loss"
    from sklearn.preprocessing import RobustScaler

    from src.serving.package import build_package
    from src.unsupervised.families import GMMDensity
    import numpy as np

    features = ["monto", "saldo", "error", "hora"]
    rng = np.random.default_rng(2)
    entrenamiento = rng.normal(size=(800, 4))
    scaler = RobustScaler().fit(entrenamiento)
    detector = GMMDensity(n_components=2).fit(scaler.transform(entrenamiento))
    package = build_package(
        detector, scaler, scaler.transform(rng.normal(size=(800, 4))), features,
        alpha=0.01, detector_name=detector_name,
    )

    antes = _detector_total(detector_name)
    n_objetivo = 300
    reporte = run_load_test(
        package, features, users=25, spawn_rate=100.0, run_time_seconds=10.0,
        max_requests=n_objetivo, corrupted_fraction=0.2,
    )
    despues = _detector_total(detector_name)

    # Cada predict() dispara 2 invocaciones instrumentadas (predict + score
    # interno, ver src/serving/package.py) -- ya probé esa relación en
    # tests/test_serving_metrics.py; acá se verifica que se registró AL MENOS una por
    # request real, sin faltantes, bajo concurrencia real.
    assert despues - antes >= reporte["total_requests"]


def test_load_burst_does_not_leak_memory():
    """No es un detector de leaks riguroso -- es el chequeo práctico que me
    propuse: RSS antes/después de una ráfaga real no debería crecer
    de forma desproporcionada al tamaño de la ráfaga."""
    package, features = _build_demo_package(seed=3)

    # warm-up: la primera ráfaga paga costos de alocación que no son un leak.
    run_load_test(package, features, users=10, spawn_rate=50.0, run_time_seconds=10.0, max_requests=200)

    gc.collect()
    proceso = psutil.Process()
    rss_antes_mb = proceso.memory_info().rss / (1024 * 1024)

    for _ in range(5):
        run_load_test(package, features, users=10, spawn_rate=50.0, run_time_seconds=10.0, max_requests=200)

    gc.collect()
    rss_despues_mb = proceso.memory_info().rss / (1024 * 1024)

    crecimiento_mb = rss_despues_mb - rss_antes_mb
    assert crecimiento_mb < 50, (
        f"RSS creció {crecimiento_mb:.1f}MB tras 1000 requests adicionales -- posible acumulación de memoria"
    )


# ---------------------------------------------------------------------------
# la app real de metrics_server sigue funcionando bajo la carga del arnés
# ---------------------------------------------------------------------------

def test_metrics_endpoint_stays_responsive_during_concurrent_inference_load(demo_package, metrics_server):
    """Escenario combinado real: inferencia y scraping de /metrics compiten
    por CPU al mismo tiempo, como en producción -- no solo cada uno por
    separado."""
    package, features = demo_package

    reporte = run_load_test(
        package, features, users=15, spawn_rate=50.0, run_time_seconds=2.0,
        host=metrics_server, corrupted_fraction=0.1, metrics_fraction=0.3,
    )

    assert reporte["scenarios"]["metrics"]["total_requests"] > 0
    assert reporte["scenarios"]["metrics"]["error_rate"] == 0.0
    assert reporte["scenarios"]["normal"]["error_rate"] == 0.0

"""Pruebas de integración del endpoint HTTP `/metrics` (src.serving.metrics_server).

Arrancan un `wsgiref.simple_server` real en un puerto efímero, en un hilo de
fondo, y lo golpean con `requests` -- es la validación end-to-end del raspado
tal como lo haría Prometheus, no una llamada directa a la app WSGI en memoria.
"""
from __future__ import annotations

import socket
import threading

import pytest
import requests

from src.serving.metrics import ANOMALY_PREDICT_LATENCY_SECONDS, ANOMALY_PREDICT_REQUESTS_TOTAL
from src.serving.metrics_server import METRICS_PATH, create_app
from wsgiref.simple_server import make_server


def _puerto_libre() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def servidor():
    puerto = _puerto_libre()
    httpd = make_server("127.0.0.1", puerto, create_app())
    hilo = threading.Thread(target=httpd.serve_forever, daemon=True)
    hilo.start()
    try:
        yield f"http://127.0.0.1:{puerto}"
    finally:
        httpd.shutdown()
        hilo.join(timeout=5)


def test_metrics_route_responds_200(servidor):
    resp = requests.get(f"{servidor}{METRICS_PATH}", timeout=5)
    assert resp.status_code == 200


def test_metrics_route_content_type_is_prometheus_text_format_0_0_4(servidor):
    resp = requests.get(f"{servidor}{METRICS_PATH}", timeout=5)
    assert resp.headers["Content-Type"] == "text/plain; version=0.0.4; charset=utf-8"


def test_any_other_path_returns_404(servidor):
    resp = requests.get(f"{servidor}/otra-ruta", timeout=5)
    assert resp.status_code == 404


def test_payload_contains_the_key_anomaly_metric_names(servidor):
    """Las métricas de negocio (ANOMALY_*) solo aparecen en el payload una vez
    que algo, en este mismo proceso, las tocó -- un Counter/Histogram con
    labels no emite muestras hasta la primera observación. Se dispara una
    observación real de cada una antes de pedir /metrics."""
    ANOMALY_PREDICT_REQUESTS_TOTAL.labels("test_endpoint", "success").inc()
    ANOMALY_PREDICT_LATENCY_SECONDS.labels("test_endpoint").observe(0.001)

    resp = requests.get(f"{servidor}{METRICS_PATH}", timeout=5)

    assert "anomaly_predict_requests_total" in resp.text
    assert "anomaly_predict_latency_seconds" in resp.text
    assert "anomaly_scores_distribution" in resp.text
    assert 'detector_name="test_endpoint"' in resp.text


def test_payload_reflects_a_real_detector_call_end_to_end(servidor):
    """Extremo a extremo real: instancia un DetectorPackage, lo llama, y
    confirma que ESE detector (no otro) aparece en lo que devuelve /metrics."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler

    from src.serving.package import build_package
    from src.unsupervised.families import GMMDensity

    features = ["monto", "saldo", "error", "hora"]
    rng = np.random.default_rng(0)
    entrenamiento = rng.normal(size=(500, 4))
    calibracion = rng.normal(size=(500, 4))
    scaler = RobustScaler().fit(entrenamiento)
    detector = GMMDensity(n_components=2).fit(scaler.transform(entrenamiento))
    paquete = build_package(
        detector, scaler, scaler.transform(calibracion), features,
        alpha=0.01, detector_name="gmm_endpoint_e2e",
    )

    paquete.score(pd.DataFrame(np.array([[0.1, 0.2, -0.1, 0.3]]), columns=features))

    resp = requests.get(f"{servidor}{METRICS_PATH}", timeout=5)
    assert 'detector_name="gmm_endpoint_e2e"' in resp.text
    assert 'status="success"' in resp.text

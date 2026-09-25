"""Aislamiento de la zona de rate limiting dedicada a `/metrics` y
consistencia del endpoint de métricas después de saturar el proxy.

Usa la misma fixture `proxy_stack` (tests/conftest.py) que
tests/test_nginx_load_e2e.py -- pytest la construye una sola vez por sesión
aunque los dos módulos la pidan.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from tests.conftest import E2E_DOCKER_HABILITADO

pytestmark = pytest.mark.skipif(
    not E2E_DOCKER_HABILITADO,
    reason="Docker no disponible, o en CI corriendo fuera del job dedicado (ver tests/conftest.py)",
)


def _disparar_rafaga(url: str, n_requests: int, n_workers: int = 40) -> list[requests.Response]:
    def _uno(_):
        try:
            return requests.get(url, timeout=5)
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        resultados = list(pool.map(_uno, range(n_requests)))
    return [r for r in resultados if r is not None]


def test_metrics_tiene_su_propio_limite_mas_estricto_que_el_general(proxy_stack):
    """60 requests concurrentes contra /metrics (zona metrics_limit: 10r/s,
    burst=5) tienen que rechazar una fraccion mucho mayor que la misma
    cantidad contra / (zona api_limit: 100r/s, burst=20) -- confirma que las
    dos zonas de tests/test_nginx_proxy.py estan realmente aisladas en
    tiempo de ejecucion, no solo en el texto de la config."""
    respuestas_metrics = _disparar_rafaga(f"{proxy_stack.proxy_url}/metrics", n_requests=60)
    respuestas_generales = _disparar_rafaga(f"{proxy_stack.proxy_url}/", n_requests=60)

    assert len(respuestas_metrics) >= 50 and len(respuestas_generales) >= 50

    tasa_rechazo_metrics = sum(1 for r in respuestas_metrics if r.status_code in (503, 429)) / len(respuestas_metrics)
    tasa_rechazo_general = sum(1 for r in respuestas_generales if r.status_code in (503, 429)) / len(respuestas_generales)

    assert tasa_rechazo_metrics > 0, "con 60 requests contra una zona de 10r/s+burst5 esperaba ver rechazos"
    assert tasa_rechazo_metrics > tasa_rechazo_general, (
        f"metrics_limit (10r/s) deberia rechazar una fraccion mayor que api_limit (100r/s) con la misma "
        f"carga: metrics={tasa_rechazo_metrics:.0%} vs general={tasa_rechazo_general:.0%}"
    )


def test_metrics_sigue_sirviendo_formato_prometheus_valido_despues_de_saturar(proxy_stack):
    """Satura /metrics a proposito y despues confirma, con requests
    espaciados para pasar el rate limit, que el endpoint sigue respondiendo
    texto Prometheus valido y estable -- que el proxy rechace exceso de
    trafico no puede dejar al backend en un estado roto o a medio responder."""
    _disparar_rafaga(f"{proxy_stack.proxy_url}/metrics", n_requests=60)

    time.sleep(1.0)  # deja que la zona metrics_limit se vacie antes de medir en limpio

    for _ in range(5):
        resp = requests.get(f"{proxy_stack.proxy_url}/metrics", timeout=5)
        assert resp.status_code == 200
        cuerpo = resp.text
        assert "# HELP" in cuerpo
        assert "# TYPE" in cuerpo
        # metrica que prometheus_client expone siempre (colector de gc, no depende del
        # SO) -- process_start_time_seconds no sirve aca: solo existe en Linux (via /proc)
        assert "python_gc_objects_collected_total" in cuerpo
        time.sleep(0.12)

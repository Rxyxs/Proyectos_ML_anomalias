"""Pruebas e2e del proxy Nginx contra contenedores reales (fixture
`proxy_stack` en tests/conftest.py): rate limiting bajo ráfaga concurrente,
cabeceras de seguridad en respuestas exitosas Y rechazadas, y el overhead de
latencia que agrega el proxy frente a golpear el backend directo.

Todo lo de acá se salta automáticamente si no hay Docker (mismo mecanismo
que tests/test_container_security.py y tests/test_nginx_proxy.py) -- las
pruebas estructurales estrictas de esos dos módulos siguen corriendo siempre,
sin excepción; esto es la capa adicional que sí necesita la pila completa.
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from tests.conftest import E2E_DOCKER_HABILITADO

pytestmark = pytest.mark.skipif(
    not E2E_DOCKER_HABILITADO,
    reason="Docker no disponible, o en CI corriendo fuera del job dedicado (ver tests/conftest.py)",
)

CABECERAS_DE_SEGURIDAD_REQUERIDAS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


def _disparar_rafaga(url: str, n_requests: int, n_workers: int = 60) -> list[requests.Response]:
    def _uno(_):
        try:
            return requests.get(url, timeout=5)
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        resultados = list(pool.map(_uno, range(n_requests)))
    return [r for r in resultados if r is not None]


# -------------------------------------------- rate limiting general (api_limit)

def test_rafaga_por_encima_de_100rps_burst_20_dispara_503(proxy_stack):
    """300 requests casi simultaneas contra `/` (zona api_limit: 100r/s,
    burst=20 nodelay) tienen que exceder por mucho la capacidad -- nginx
    responde 503 para las que rechaza."""
    respuestas = _disparar_rafaga(f"{proxy_stack.proxy_url}/", n_requests=300)
    assert len(respuestas) >= 250, "muy pocas respuestas llegaron -- la rafaga no se disparo bien"

    codigos = [r.status_code for r in respuestas]
    rechazadas = [c for c in codigos if c in (503, 429)]
    aceptadas = [c for c in codigos if c not in (503, 429)]

    assert rechazadas, f"esperaba al menos una respuesta 503/429, los codigos fueron: {set(codigos)}"
    assert aceptadas, "todas las respuestas fueron rechazadas -- el burst=20 nodelay no dejo pasar nada"


def test_cabeceras_de_seguridad_presentes_en_200_y_en_503(proxy_stack):
    """Repite la rafaga y confirma que las 4 cabeceras estan tanto en una
    respuesta que SI paso (200/404, forwarded al backend) como en una que
    nginx rechazo (503/429) -- por eso las declare con `always` en nginx.conf."""
    respuestas = _disparar_rafaga(f"{proxy_stack.proxy_url}/", n_requests=300)

    aceptada = next((r for r in respuestas if r.status_code not in (503, 429)), None)
    rechazada = next((r for r in respuestas if r.status_code in (503, 429)), None)
    assert aceptada is not None, "necesito al menos una respuesta aceptada para comparar cabeceras"
    assert rechazada is not None, "necesito al menos una respuesta rechazada para comparar cabeceras"

    for resp, etiqueta in ((aceptada, "aceptada"), (rechazada, "rechazada")):
        for cabecera, valor in CABECERAS_DE_SEGURIDAD_REQUERIDAS.items():
            assert resp.headers.get(cabecera) == valor, (
                f"falta o difiere '{cabecera}' en la respuesta {etiqueta} "
                f"(status={resp.status_code}): {resp.headers.get(cabecera)!r}"
            )


# ------------------------------------------------------- overhead del proxy

def test_overhead_de_latencia_del_proxy_es_bajo(proxy_stack):
    """Compara la mediana de latencia de /metrics golpeado directo al backend
    vs a traves del proxy. Uso `/metrics` real (no `/`, que 404ea) para medir
    el mismo trabajo de verdad en ambos lados, con una sesion HTTP persistente
    (mismo socket TCP entre requests) para que la comparacion sea sobre el
    procesamiento, no sobre el costo de abrir conexion cada vez.

    El umbral: nginx en si mismo agrega una sobrecarga de procesamiento
    sub-milisegundo (es su propio diseño, bien documentado) -- pero esta
    medicion es de punta a punta a traves de la red bridge/NAT de Docker, que
    agrega su propio salto ademas del de nginx, y en un runner de CI
    compartido hay jitter que ninguno de los dos controla. Un techo de 5ms
    para la MEDIANA es una SLA defendible para un proxy reverso en este
    escenario -- no la sobrecarga aislada de nginx, que es menor. El valor
    real medido se imprime siempre, se cumpla o no la hipotesis de partida.
    """
    n = 20
    espera_entre_requests_s = 0.12  # /metrics: zona metrics_limit = 10r/s -- me quedo bien por debajo

    sesion = requests.Session()
    directas_ms = []
    proxiadas_ms = []

    for _ in range(n):
        inicio = time.perf_counter()
        resp = sesion.get(f"{proxy_stack.direct_url}/metrics", timeout=5)
        directas_ms.append((time.perf_counter() - inicio) * 1000)
        assert resp.status_code == 200
        time.sleep(espera_entre_requests_s)

    for _ in range(n):
        inicio = time.perf_counter()
        resp = sesion.get(f"{proxy_stack.proxy_url}/metrics", timeout=5)
        proxiadas_ms.append((time.perf_counter() - inicio) * 1000)
        assert resp.status_code == 200
        time.sleep(espera_entre_requests_s)

    mediana_directa = statistics.median(directas_ms)
    mediana_proxiada = statistics.median(proxiadas_ms)
    overhead_ms = mediana_proxiada - mediana_directa

    print(
        f"\n[overhead nginx] directo: {mediana_directa:.3f}ms (mediana, n={n}) | "
        f"proxiado: {mediana_proxiada:.3f}ms (mediana, n={n}) | overhead: {overhead_ms:.3f}ms"
    )

    assert overhead_ms < 5.0, (
        f"overhead de la mediana fue {overhead_ms:.3f}ms (directo={mediana_directa:.3f}ms, "
        f"proxiado={mediana_proxiada:.3f}ms) -- por encima del techo de 5ms"
    )

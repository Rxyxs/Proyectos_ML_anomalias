"""Fixture compartida: levanta serving + proxy Nginx como contenedores reales
en una red Docker propia, para las pruebas e2e de tests/test_nginx_load_e2e.py
y tests/test_proxy_performance.py.

No reuso docker-compose.prod.yml tal cual: en produccion `serving` no publica
su puerto al host a propósito (solo el proxy es la entrada externa), pero la
prueba de overhead necesita golpear el backend DIRECTO para comparar contra
la ruta con proxy -- así que esta fixture arma su propia topología efímera
(mismas imágenes, mismo nginx.conf), con el puerto de `serving` publicado
solo para esta medición, y la tira abajo al terminar.

session-scoped: los dos módulos que necesitan este stack comparten una sola
build/arranque en vez de repetirlo -- el build de la imagen de serving
(pandas/torch/xgboost) tarda un par de minutos, no tiene sentido pagarlo dos
veces en la misma corrida.

Bandera de ejecución para CI: todo runner de GitHub Actions trae Docker ya
activo, en los tres jobs (`pytest` x2 por la matriz, y `docker`) -- sin esta
bandera, el build pesado de la imagen (y las dos suites e2e completas) se
repetiría 3 veces por cada push, solo por casualidad de que Docker esta
disponible en todos lados. `RUN_DOCKER_E2E=1` lo habilita explicitamente en
UN SOLO job de CI (ver .github/workflows/ci.yml, job `docker`); localmente,
sin esa variable, corre igual si hay Docker -- la bandera es para no pagar
el build 3 veces en CI, no para ocultar la prueba en desarrollo.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
NGINX_CONF = REPO_ROOT / "docker" / "nginx.conf"


def _docker_disponible() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def _e2e_docker_habilitado() -> bool:
    if not _docker_disponible():
        return False
    en_ci = os.environ.get("CI", "").lower() == "true"
    return (not en_ci) or os.environ.get("RUN_DOCKER_E2E") == "1"


DOCKER_DISPONIBLE = _docker_disponible()
E2E_DOCKER_HABILITADO = _e2e_docker_habilitado()


@dataclass(frozen=True)
class ProxyStack:
    direct_url: str  # http://127.0.0.1:<puerto> -- serving, sin pasar por nginx
    proxy_url: str    # http://127.0.0.1:<puerto> -- serving, a traves de nginx


def _puerto_publicado(nombre_contenedor: str, puerto_interno: int) -> int:
    resultado = subprocess.run(
        ["docker", "port", nombre_contenedor, str(puerto_interno)],
        capture_output=True, text=True, timeout=10, check=True,
    )
    # formato tipico: "0.0.0.0:54321\n" (puede listar varias lineas, IPv4 e IPv6)
    primera_linea = resultado.stdout.strip().splitlines()[0]
    return int(primera_linea.rsplit(":", 1)[-1])


def _esperar_url(url: str, timeout_s: float = 60.0) -> None:
    limite = time.monotonic() + timeout_s
    ultimo_error = None
    while time.monotonic() < limite:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code in (200, 404):
                return
        except requests.RequestException as exc:
            ultimo_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"{url} no respondio en {timeout_s}s (ultimo error: {ultimo_error})")


@pytest.fixture(scope="session")
def proxy_stack():
    if not E2E_DOCKER_HABILITADO:
        razon = (
            "Docker no está disponible en este entorno" if not DOCKER_DISPONIBLE
            else "en CI, las pruebas e2e con Docker solo corren en el job dedicado (RUN_DOCKER_E2E=1)"
        )
        pytest.skip(razon)

    sufijo = uuid.uuid4().hex[:8]
    tag_imagen = f"anomaly-detector-serving:e2e-{sufijo}"
    red = f"anomaly-e2e-net-{sufijo}"
    nombre_serving = f"anomaly-e2e-serving-{sufijo}"
    nombre_proxy = f"anomaly-e2e-proxy-{sufijo}"

    subprocess.run(["docker", "build", "-t", tag_imagen, str(REPO_ROOT)], check=True, timeout=900)
    subprocess.run(["docker", "network", "create", red], check=True, capture_output=True, timeout=30)

    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", nombre_serving,
                "--network", red,
                "--network-alias", "serving",
                "-p", "127.0.0.1::8001",
                tag_imagen,
            ],
            check=True, capture_output=True, timeout=30,
        )
        puerto_serving = _puerto_publicado(nombre_serving, 8001)
        direct_url = f"http://127.0.0.1:{puerto_serving}"
        _esperar_url(f"{direct_url}/metrics")

        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", nombre_proxy,
                "--network", red,
                "-p", "127.0.0.1::80",
                "-v", f"{NGINX_CONF}:/etc/nginx/nginx.conf:ro",
                "nginx:alpine-slim",
            ],
            check=True, capture_output=True, timeout=30,
        )
        puerto_proxy = _puerto_publicado(nombre_proxy, 80)
        proxy_url = f"http://127.0.0.1:{puerto_proxy}"
        _esperar_url(f"{proxy_url}/metrics")

        yield ProxyStack(direct_url=direct_url, proxy_url=proxy_url)
    finally:
        subprocess.run(["docker", "rm", "-f", nombre_serving, nombre_proxy], capture_output=True)
        subprocess.run(["docker", "network", "rm", red], capture_output=True)
        subprocess.run(["docker", "image", "rm", "-f", tag_imagen], capture_output=True)

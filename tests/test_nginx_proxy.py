"""Validación del proxy inverso Nginx (docker/nginx.conf) y de su entrada en
docker-compose.prod.yml.

Dos niveles, igual que tests/test_container_security.py:

1. Estáticas (siempre corren): parsean nginx.conf y el compose como
   texto/YAML y comprueban rate limiting, cabeceras de seguridad, proxying y
   aislamiento de red.
2. Real (se salta si no hay Docker): corre `nginx -t` con la imagen oficial
   y el nginx.conf real montado -- valida sintaxis de verdad, no solo que
   las directivas esperadas aparezcan como substring.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
NGINX_CONF = REPO_ROOT / "docker" / "nginx.conf"
COMPOSE_PROD = REPO_ROOT / "docker-compose.prod.yml"


def _docker_disponible() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


DOCKER_DISPONIBLE = _docker_disponible()


# --------------------------------------------------------- nginx.conf estático

@pytest.fixture(scope="module")
def nginx_conf_texto() -> str:
    return NGINX_CONF.read_text(encoding="utf-8")


def test_define_zona_de_rate_limiting_general_100_req_por_segundo(nginx_conf_texto):
    assert "limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;" in nginx_conf_texto


def test_define_zona_de_rate_limiting_dedicada_a_metrics_10_req_por_segundo(nginx_conf_texto):
    assert "limit_req_zone $binary_remote_addr zone=metrics_limit:10m rate=10r/s;" in nginx_conf_texto


def test_la_ruta_general_aplica_el_limite_con_burst_20_nodelay(nginx_conf_texto):
    assert "limit_req zone=api_limit burst=20 nodelay;" in nginx_conf_texto


def test_metrics_tiene_su_propio_limite_dedicado(nginx_conf_texto):
    """No reutiliza `api_limit`: /metrics necesita su propia zona (10r/s),
    no el limite general de 100r/s -- si comparten zona, este test falla."""
    bloque_metrics = nginx_conf_texto.split("location /metrics", 1)[1].split("location /", 1)[0]
    assert "limit_req zone=metrics_limit" in bloque_metrics
    assert "zone=api_limit" not in bloque_metrics


@pytest.mark.parametrize("cabecera,valor", [
    ("X-Frame-Options", '"DENY"'),
    ("X-Content-Type-Options", '"nosniff"'),
    ("X-XSS-Protection", '"1; mode=block"'),
    ("Referrer-Policy", '"strict-origin-when-cross-origin"'),
])
def test_cabeceras_de_seguridad_requeridas(nginx_conf_texto, cabecera, valor):
    assert f"add_header {cabecera} {valor} always;" in nginx_conf_texto


def test_las_dos_rutas_reenvian_al_servicio_de_serving(nginx_conf_texto):
    assert nginx_conf_texto.count("proxy_pass http://serving:8001;") == 2


def test_escucha_en_el_puerto_80(nginx_conf_texto):
    assert "listen 80;" in nginx_conf_texto


# ------------------------------------------------ docker-compose.prod.yml

@pytest.fixture(scope="module")
def compose_prod() -> dict:
    return yaml.safe_load(COMPOSE_PROD.read_text(encoding="utf-8"))


def test_compose_prod_define_el_servicio_proxy_con_la_imagen_oficial(compose_prod):
    assert compose_prod["services"]["proxy"]["image"] == "nginx:alpine-slim"


def test_compose_prod_expone_solo_el_puerto_80_del_proxy(compose_prod):
    assert compose_prod["services"]["proxy"]["ports"] == ["80:80"]


def test_compose_prod_serving_ya_no_expone_su_puerto_al_host(compose_prod):
    """El proxy es la unica entrada externa: `serving` no debe tener una
    clave `ports` que lo haga alcanzable saltandose el rate limiting y las
    cabeceras del proxy."""
    assert "ports" not in compose_prod["services"]["serving"]


def test_compose_prod_proxy_y_serving_comparten_la_red_interna(compose_prod):
    assert compose_prod["services"]["proxy"]["networks"] == ["monitoring_net"]
    assert compose_prod["services"]["serving"]["networks"] == ["monitoring_net"]


def test_compose_prod_limita_recursos_del_proxy(compose_prod):
    limites = compose_prod["services"]["proxy"]["deploy"]["resources"]["limits"]
    assert limites["cpus"] == "0.5"
    assert limites["memory"] == "128M"


def test_compose_prod_proxy_es_read_only_con_tmpfs_en_tmp_y_cache(compose_prod):
    proxy = compose_prod["services"]["proxy"]
    assert proxy["read_only"] is True
    assert set(proxy["tmpfs"]) == {"/tmp", "/var/cache/nginx"}


def test_compose_prod_proxy_monta_el_nginx_conf_real(compose_prod):
    assert "./docker/nginx.conf:/etc/nginx/nginx.conf:ro" in compose_prod["services"]["proxy"]["volumes"]


# --------------------------------------------------- validación en vivo

@pytest.mark.skipif(not DOCKER_DISPONIBLE, reason="Docker no está disponible en este entorno")
def test_nginx_conf_es_sintacticamente_valido_segun_nginx_t():
    """`nginx -t` con la imagen real: valida sintaxis contra el binario de
    verdad, no una lista de substrings esperados -- una directiva mal escrita
    pasaría todas las pruebas estáticas de arriba y solo esta la atraparía."""
    resultado = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{NGINX_CONF}:/etc/nginx/nginx.conf:ro",
            "nginx:alpine-slim",
            "nginx", "-t",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert resultado.returncode == 0, (
        f"nginx -t fallo:\nstdout={resultado.stdout}\nstderr={resultado.stderr}"
    )

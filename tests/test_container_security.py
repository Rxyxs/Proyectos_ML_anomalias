"""Validación del Dockerfile multi-stage y de docker-compose.prod.yml.

Dos niveles de prueba:

1. Estáticas (siempre corren, no necesitan Docker instalado): parsean el
   Dockerfile y el compose de producción como texto/YAML y comprueban la
   estructura -- dos etapas, usuario no-root con UID fijo, HEALTHCHECK sin
   curl/wget, límites de recursos, `no-new-privileges`.
2. Reales (se saltan si no hay daemon de Docker disponible): construyen la
   imagen, levantan un contenedor efímero y verifican en vivo que el proceso
   corre con UID != 0 y que el HEALTHCHECK reporta "healthy". En esta
   máquina no hay Docker instalado, así que estas pruebas quedan skipped acá;
   corren de verdad en cualquier entorno que sí lo tenga (por ejemplo, el
   job `docker` de CI).
"""
from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE_PROD = REPO_ROOT / "docker-compose.prod.yml"


def _docker_disponible() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


DOCKER_DISPONIBLE = _docker_disponible()


# --------------------------------------------------------- Dockerfile estático

@pytest.fixture(scope="module")
def dockerfile_texto() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_es_multi_stage(dockerfile_texto):
    etapas = [l for l in dockerfile_texto.splitlines() if l.strip().upper().startswith("FROM")]
    assert len(etapas) >= 2, "esperaba al menos dos instrucciones FROM (builder + runtime)"
    assert any("AS builder" in l for l in etapas)
    assert any("AS runtime" in l for l in etapas)


def test_dockerfile_crea_usuario_no_root_con_uid_10001(dockerfile_texto):
    assert "useradd --uid 10001" in dockerfile_texto
    assert "USER appuser" in dockerfile_texto


def test_dockerfile_usa_python_bytecode_y_buffer_flags(dockerfile_texto):
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile_texto
    assert "PYTHONUNBUFFERED=1" in dockerfile_texto


def test_dockerfile_healthcheck_no_depende_de_curl_ni_wget(dockerfile_texto):
    """Mira solo la instrucción HEALTHCHECK en sí (no los comentarios
    alrededor, que sí pueden mencionar curl/wget al explicar por qué no se
    usan)."""
    lineas_healthcheck = [
        l for l in dockerfile_texto.splitlines()
        if l.strip().upper().startswith("HEALTHCHECK") or l.strip().upper().startswith("CMD PYTHON")
    ]
    assert lineas_healthcheck, "no encontre la instruccion HEALTHCHECK"
    instruccion = " ".join(lineas_healthcheck).lower()
    assert "curl" not in instruccion
    assert "wget" not in instruccion
    assert "urllib.request" in dockerfile_texto


def test_dockerfile_expone_puerto_8001(dockerfile_texto):
    assert "EXPOSE 8001" in dockerfile_texto


def test_dockerfile_user_va_despues_del_healthcheck_no_antes(dockerfile_texto):
    """El HEALTHCHECK se declara antes de bajar privilegios con USER -- no
    depende de en qué orden Docker las ejecuta (las dos son metadata de
    imagen, no pasos secuenciales), pero mantenerlas en este orden deja claro
    que el HEALTHCHECK aplica al proceso final, ya sin privilegios de root."""
    idx_healthcheck = dockerfile_texto.index("HEALTHCHECK")
    idx_user = dockerfile_texto.index("USER appuser")
    assert idx_healthcheck < idx_user


# ------------------------------------------------ docker-compose.prod.yml

@pytest.fixture(scope="module")
def compose_prod() -> dict:
    return yaml.safe_load(COMPOSE_PROD.read_text(encoding="utf-8"))


def test_compose_prod_define_serving_y_prometheus(compose_prod):
    servicios = compose_prod["services"]
    assert "serving" in servicios
    assert "prometheus" in servicios


def test_compose_prod_limita_recursos_del_serving(compose_prod):
    limites = compose_prod["services"]["serving"]["deploy"]["resources"]["limits"]
    assert limites["cpus"] == "2.0"
    assert limites["memory"] == "1G"


def test_compose_prod_tiene_no_new_privileges_en_ambos_servicios(compose_prod):
    for nombre in ("serving", "prometheus"):
        assert "no-new-privileges:true" in compose_prod["services"][nombre]["security_opt"]


def test_compose_prod_usa_red_aislada_monitoring_net(compose_prod):
    assert "monitoring_net" in compose_prod["networks"]
    for nombre in ("serving", "prometheus"):
        assert compose_prod["services"][nombre]["networks"] == ["monitoring_net"]


def test_compose_prod_tiene_volumen_de_metricas_persistente(compose_prod):
    assert "metrics-data" in compose_prod["volumes"]
    volumenes_prometheus = compose_prod["services"]["prometheus"]["volumes"]
    assert any(v.startswith("metrics-data:") for v in volumenes_prometheus)


# --------------------------------------------------- validación en vivo

@pytest.mark.skipif(not DOCKER_DISPONIBLE, reason="Docker no está disponible en este entorno")
class TestContenedorEnVivo:
    """Construye la imagen una sola vez para todo el módulo (es una imagen
    pesada: pandas/numpy/scikit-learn/torch/xgboost) y reutiliza el tag entre
    los distintos tests en vivo."""

    tag = f"anomaly-detector-serving:test-{uuid.uuid4().hex[:8]}"

    @pytest.fixture(scope="class", autouse=True)
    def imagen_construida(self):
        subprocess.run(
            ["docker", "build", "-t", self.tag, str(REPO_ROOT)],
            check=True, timeout=900,
        )
        yield
        subprocess.run(["docker", "image", "rm", "-f", self.tag], capture_output=True)

    @pytest.fixture
    def contenedor(self):
        nombre = f"anomaly-test-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "run", "-d", "--name", nombre, self.tag],
            check=True, capture_output=True, timeout=30,
        )
        try:
            yield nombre
        finally:
            subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)

    def test_el_proceso_corre_con_uid_no_root(self, contenedor):
        resultado = subprocess.run(
            ["docker", "exec", contenedor, "id", "-u"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        uid = int(resultado.stdout.strip())
        assert uid == 10001
        assert uid != 0

    def test_healthcheck_reporta_healthy(self, contenedor):
        limite = time.monotonic() + 60
        estado = None
        while time.monotonic() < limite:
            resultado = subprocess.run(
                ["docker", "inspect", "--format={{.State.Health.Status}}", contenedor],
                capture_output=True, text=True, timeout=10, check=True,
            )
            estado = resultado.stdout.strip()
            if estado == "healthy":
                break
            time.sleep(3)
        assert estado == "healthy", f"HEALTHCHECK nunca reportó healthy (último estado: {estado!r})"

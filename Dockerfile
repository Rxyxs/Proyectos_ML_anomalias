# ---- Etapa 1: build de dependencias ----
# Aisla la instalacion de paquetes (y su cache de pip) del runtime final:
# lo unico que cruza a la etapa siguiente es el arbol /install ya resuelto.
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Etapa 2: runtime ----
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/usr/local/lib/python3.10/site-packages

# UID fijo (10001) en vez de dejar que useradd asigne el siguiente libre:
# asi el mismo numero es verificable desde fuera del contenedor (docker
# inspect, docker exec ... id -u) sin depender del orden de creacion.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser src ./src

EXPOSE 8001

# Sirve /metrics con la telemetria del proceso. Los contadores/histogramas
# ANOMALY_* (src/serving/metrics.py) solo tienen datos una vez que algo en
# este mismo proceso llama a DetectorPackage.predict/score/score_from_scaled
# -- este contenedor demuestra el mecanismo de scraping end-to-end, no
# sustituye un servicio de inferencia real sirviendo trafico.
#
# HEALTHCHECK con urllib de la stdlib en vez de curl/wget: ninguno de los
# dos viene en python:3.10-slim, y agregarlos solo para el healthcheck
# significaria sumar paquetes del sistema a una imagen que ya no los
# necesita para nada mas.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/metrics', timeout=2)" || exit 1

USER appuser

CMD ["python", "-m", "src.serving.metrics_server"]

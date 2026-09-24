FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8001

# Sirve /metrics con la telemetria del proceso. Los contadores/histogramas
# ANOMALY_* (src/serving/metrics.py) solo tienen datos una vez que algo en
# este mismo proceso llama a DetectorPackage.predict/score/score_from_scaled
# -- este contenedor demuestra el mecanismo de scraping end-to-end, no
# sustituye un servicio de inferencia real sirviendo trafico.
CMD ["python", "-m", "src.serving.metrics_server"]

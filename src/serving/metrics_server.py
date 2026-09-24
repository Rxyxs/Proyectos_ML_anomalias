"""Módulo 8 (continuación) — servidor HTTP standalone que expone `/metrics`
para que Prometheus raspee la telemetría de `src/serving/metrics.py`.

`prometheus_client.make_wsgi_app()` ya devuelve exactamente lo que un
scraper de Prometheus espera -- sin `Accept` header (el caso real de un
scrape), responde `Content-Type: text/plain; version=0.0.4; charset=utf-8`,
verificado contra la librería instalada -- así que no hace falta FastAPI ni
ningún framework web para esto: `wsgiref.simple_server` (stdlib) alcanza, y
no agrega una dependencia nueva a un repo que hasta ahora nunca sirvió HTTP.

La app de prometheus_client por sí sola responde en CUALQUIER ruta, no solo
`/metrics` -- `create_app()` la envuelve para que sea addressable
específicamente en `/metrics` y devuelva 404 en cualquier otra, que es lo
que pide el Día 2, no "todo es /metrics".

Ejecución: python -m src.serving.metrics_server
"""
from __future__ import annotations

from wsgiref.simple_server import WSGIServer, make_server

from prometheus_client import make_wsgi_app

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
METRICS_PATH = "/metrics"

_prometheus_app = make_wsgi_app()


def create_app():
    """App WSGI que sirve la telemetría de Prometheus solo en `METRICS_PATH`."""

    def app(environ, start_response):
        if environ.get("PATH_INFO") == METRICS_PATH:
            return _prometheus_app(environ, start_response)
        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"not found"]

    return app


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> WSGIServer:
    """Arranca el servidor de métricas y bloquea sirviendo peticiones."""
    servidor = make_server(host, port, create_app())
    print(f"Sirviendo metricas Prometheus en http://{host}:{port}{METRICS_PATH}")
    servidor.serve_forever()
    return servidor


if __name__ == "__main__":
    serve()

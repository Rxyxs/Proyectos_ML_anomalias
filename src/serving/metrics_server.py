"""Módulo 8 (continuación) — servidor HTTP standalone que expone `/metrics`
para que Prometheus raspee la telemetría de `src/serving/metrics.py`.

`prometheus_client.make_wsgi_app()` ya devuelve exactamente lo que un
scraper de Prometheus espera -- sin `Accept` header (el caso real de un
scrape), responde `Content-Type: text/plain; version=0.0.4; charset=utf-8`,
verificado contra la librería instalada -- así que no hace falta FastAPI ni
ningún framework web para esto: `wsgiref.simple_server` (stdlib) alcanza, y
no agrega una dependencia nueva a un repo que hasta ahora nunca sirvió HTTP.

La app de prometheus_client por sí sola responde en CUALQUIER ruta, no solo
`/metrics` -- envolví `create_app()` para que sea addressable
específicamente en `/metrics` y devuelva 404 en cualquier otra, no "todo es
/metrics".

Servidor con threads y backlog ampliado: probando carga concurrente sobre
`/metrics` con 15 workers scrapeando en paralelo encontré p95=610ms,
p99>1s -- contra scrapes individuales de unos pocos ms. Mi primera hipótesis
("el servidor atiende un request a la vez") resultó ser INCOMPLETA: agregar
`ThreadingMixIn` por sí solo casi no cambió los números (p95=528ms) en una
comparación A/B directa. Perfilando por nivel de concurrencia (1, 3, 5, 15
workers) el salto aparece justo entre 5 y 15 -- exactamente el valor por
defecto de `socketserver.BaseServer.request_queue_size` (5): con más de 5
conexiones entrantes simultáneas, el backlog TCP del `listen()` subyacente
se llena, y el SO retiene el resto hasta que se libera un lugar, agregando
cientos de milisegundos de espera antes de que el proceso Python vea
siquiera la conexión. Confirmé subiendo `request_queue_size` a 128: p95 bajó
de ~600ms a **71ms** (~8x), medido en el mismo escenario. `ThreadingMixIn`
sigue siendo necesario además del backlog más grande -- sin él, las
conexiones se aceptan más rápido pero se siguen procesando de a una.
`/metrics` es de solo lectura y sin estado propio (la app de
`prometheus_client` ya es segura para llamarse desde varios threads a la
vez), así que no hay ninguna sección crítica nueva que proteger.

Ejecución: python -m src.serving.metrics_server
"""
from __future__ import annotations

from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from prometheus_client import make_wsgi_app

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
METRICS_PATH = "/metrics"
REQUEST_QUEUE_SIZE = 128

_prometheus_app = make_wsgi_app()


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """`WSGIServer` con un thread por conexión y un backlog TCP más grande
    que el default de 5 -- ver la nota en el docstring del módulo: el
    backlog, no la falta de threads, era la causa real de la latencia bajo
    carga concurrente."""

    daemon_threads = True
    request_queue_size = REQUEST_QUEUE_SIZE


def create_app():
    """App WSGI que sirve la telemetría de Prometheus solo en `METRICS_PATH`."""

    def app(environ, start_response):
        if environ.get("PATH_INFO") == METRICS_PATH:
            return _prometheus_app(environ, start_response)
        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"not found"]

    return app


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> WSGIServer:
    """Arranca el servidor de métricas (con threads) y bloquea sirviendo peticiones."""
    servidor = make_server(host, port, create_app(), server_class=ThreadingWSGIServer)
    print(f"Sirviendo metricas Prometheus en http://{host}:{port}{METRICS_PATH}")
    servidor.serve_forever()
    return servidor


if __name__ == "__main__":
    serve()

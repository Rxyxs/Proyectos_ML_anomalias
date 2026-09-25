"""Micro-batching dinámico: agrupa solicitudes individuales de scoring en
una sola matriz NumPy y las despacha juntas a `DetectorPackage`, para
amortizar el costo fijo por llamada (instrumentación, overhead de
scikit-learn/ONNX Runtime) entre varias filas en vez de pagarlo una por una.

`queue.Queue` + `threading`, no `asyncio.Queue`: el resto de la capa de
serving de este repo (`DetectorPackage`, `ONNXDetectorWrapper`) es
síncrona, y el arnés de carga del Día 10 ya usa threads, no un loop de
eventos -- meter asyncio acá agregaría un segundo modelo de concurrencia sin
necesidad real.

Cada `submit()` encola su fila y se BLOQUEA en un `concurrent.futures.Future`
hasta que un hilo despachador en el fondo arma un lote (por tamaño o por
tiempo, lo que pase primero) y lo corre en una sola llamada vectorizada. Es
el mismo patrón que usaría un servidor de inferencia real detrás de un
endpoint HTTP: cada request-handler thread llama a `submit()` y espera, sin
saber que su fila terminó apilada junto a las de otros threads.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.serving.metrics import ANOMALY_BATCH_SIZE_HISTOGRAM, ANOMALY_BATCH_WAIT_TIME_SECONDS

DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_MAX_WAIT_MS = 2.0
# Cuánto espera cada `queue.Queue.get()` de sondeo cuando el lote todavía
# está vacío -- acota qué tan rápido el hilo despachador nota `close()`
# cuando no hay tráfico, sin ocupar CPU en un loop ajustado.
_IDLE_POLL_SECONDS = 0.1


@dataclass
class _PendingRequest:
    row: np.ndarray  # 1D: una sola fila
    future: "Future[Any]"
    enqueued_at: float
    method: str  # "score" o "predict"


class BatcherClosedError(RuntimeError):
    """`submit()` se llamó después de `close()` -- ningún hilo despachador
    va a drenar esta solicitud, así que se rechaza de inmediato en vez de
    bloquear para siempre."""


class DynamicBatcher:
    """Agrupa llamadas a `detector_package.score()`/`.predict()`.

    Lógica de acumulación: el hilo despachador arma un lote esperando hasta
    `max_wait_ms` desde que llega la PRIMERA solicitud del lote, o hasta
    juntar `max_batch_size` solicitudes -- lo que ocurra primero. Bajo
    tráfico sostenido, eso empuja los lotes cerca de `max_batch_size`; con
    una sola solicitud esporádica, el lote de 1 se despacha apenas expira
    `max_wait_ms`, nunca se queda esperando indefinidamente.
    """

    def __init__(self, detector_package, max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                 max_wait_ms: float = DEFAULT_MAX_WAIT_MS):
        if max_batch_size < 1:
            raise ValueError("max_batch_size debe ser al menos 1.")
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms debe ser positivo.")

        self.detector_package = detector_package
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._detector_name = str(getattr(detector_package, "metadata", {}).get("detector", "desconocido"))

        self._cola: "queue.Queue[_PendingRequest]" = queue.Queue()
        self._detener = threading.Event()
        self._cerrado = threading.Event()
        self._hilo = threading.Thread(target=self._dispatch_loop, daemon=True, name="DynamicBatcher")
        self._hilo.start()

    # ------------------------------------------------------------------ API pública

    def submit(self, row, method: str = "score", timeout: Optional[float] = 30.0):
        """Encola `row` (1D, una fila) y bloquea hasta tener el resultado
        individual correspondiente a esa fila (no el lote entero).

        `method` es `"score"` o `"predict"` -- el nombre real del método de
        `DetectorPackage` a invocar sobre el lote.
        """
        if self._cerrado.is_set():
            raise BatcherClosedError("DynamicBatcher está cerrado; no acepta nuevas solicitudes.")

        fila = np.asarray(row, dtype=float).reshape(-1)
        future: "Future[Any]" = Future()
        self._cola.put(_PendingRequest(row=fila, future=future, enqueued_at=time.perf_counter(), method=method))
        return future.result(timeout=timeout)

    def close(self, timeout: Optional[float] = 10.0) -> None:
        """Cierre elegante: deja de aceptar solicitudes nuevas (`submit()`
        posteriores lanzan `BatcherClosedError`), pero el hilo despachador
        sigue drenando y procesando lo que YA está en cola hasta vaciarla
        antes de terminar.
        """
        self._cerrado.set()
        self._detener.set()
        self._hilo.join(timeout=timeout)

    def __enter__(self) -> "DynamicBatcher":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ hilo despachador

    def _dispatch_loop(self) -> None:
        while True:
            lote = self._collect_batch()
            if lote:
                self._process_batch(lote)
            elif self._detener.is_set():
                return

    def _collect_batch(self) -> list[_PendingRequest]:
        """Junta hasta `max_batch_size` solicitudes, esperando como máximo
        `max_wait_ms` desde que llega la primera. Devuelve `[]` si no llegó
        ninguna en el sondeo de reposo (`_IDLE_POLL_SECONDS`) -- normal
        cuando no hay tráfico, no es un error.
        """
        try:
            primero = self._cola.get(timeout=_IDLE_POLL_SECONDS)
        except queue.Empty:
            return []

        lote = [primero]
        deadline = time.perf_counter() + self.max_wait_ms / 1000.0
        while len(lote) < self.max_batch_size:
            restante = deadline - time.perf_counter()
            if restante <= 0:
                break
            try:
                lote.append(self._cola.get(timeout=restante))
            except queue.Empty:
                break
        return lote

    def _process_batch(self, lote: list[_PendingRequest]) -> None:
        ahora = time.perf_counter()
        for item in lote:
            ANOMALY_BATCH_WAIT_TIME_SECONDS.labels(self._detector_name).observe(ahora - item.enqueued_at)
        ANOMALY_BATCH_SIZE_HISTOGRAM.labels(self._detector_name).observe(len(lote))

        # Se agrupa por método real a invocar (score/predict): normalmente
        # un batcher sirve un solo método, pero mezclar no debería romper
        # nada ni devolver el tipo de resultado equivocado a nadie.
        por_metodo: dict[str, list[_PendingRequest]] = {}
        for item in lote:
            por_metodo.setdefault(item.method, []).append(item)

        for metodo, items in por_metodo.items():
            matriz = np.vstack([it.row for it in items])
            try:
                fn = getattr(self.detector_package, metodo)
                resultados = fn(matriz)
            except Exception as exc:  # noqa: BLE001 - se propaga tal cual a cada peticionario
                for it in items:
                    it.future.set_exception(exc)
                continue
            for it, resultado in zip(items, resultados):
                it.future.set_result(resultado)

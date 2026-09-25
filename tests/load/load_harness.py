"""Arnés de carga concurrente: satura `DetectorPackage.predict()` (tráfico
normal y corrupto) y, en paralelo, el endpoint `/metrics`
(`src/serving/metrics_server.py`) para confirmar que el raspado de
Prometheus no degrada la latencia de inferencia.

Sin Locust a propósito: Locust está construido alrededor de un cliente HTTP
contra un servidor real, pero este repo no expone un endpoint HTTP de
inferencia -- solo `/metrics`. Los escenarios "normal" y "corrupto" son
llamadas EN PROCESO a `DetectorPackage`, así que Locust no encajaría para
esos dos sin inventar una API HTTP que no existe. Se usa
`concurrent.futures`/`threading` puro (sin dependencia nueva) para los tres
escenarios; el de `/metrics` sí golpea el servidor real por HTTP.

Tres escenarios, elegidos por muestreo ponderado en cada request de cada
worker (no en bloques separados, para que compitan por CPU de verdad, como
en producción):

- **normal**: `predict()` sobre una fila sintética sin corromper.
- **corrupted**: `predict()` sobre una fila con un NaN/Inf inyectado a
  propósito -- el resultado CORRECTO es que `DetectorPackage` la rechace
  (`ValueError`/`NonFiniteInputError`), así que un rechazo ahí cuenta como
  éxito del escenario, no como error del sistema.
- **metrics**: `GET {host}/metrics` -- solo si se pasa `--host`.

Ejecución:

    python -m tests.load.load_harness --users 20 --run-time 5 --output-json outputs/load/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.serving.package import DetectorPackage

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "load" / "load_test_report.json"
VALID_ERROR_RATE_THRESHOLD = 0.001  # 0.1%, mi umbral aceptable de error sobre trafico VALIDO


@dataclass
class RequestResult:
    scenario: str
    latency_seconds: float
    success: bool
    error_type: Optional[str] = None


def _percentile(valores_ms: list[float], pct: float) -> Optional[float]:
    if not valores_ms:
        return None
    return float(np.percentile(np.asarray(valores_ms, dtype=float), pct))


def _synthetic_row(features: list[str], rng: np.random.Generator, corrupted: bool) -> pd.DataFrame:
    fila = rng.normal(size=len(features))
    if corrupted:
        idx = rng.integers(0, len(features))
        fila[idx] = rng.choice([np.nan, np.inf, -np.inf])
    return pd.DataFrame([fila], columns=features)


def _run_inference_request(package: DetectorPackage, features: list[str],
                            rng: np.random.Generator, corrupted: bool) -> RequestResult:
    """`success` significa "el sistema hizo lo correcto", no "no hubo
    excepción": para `corrupted=True` lo correcto es que `predict()` RECHACE
    la fila (ValueError) -- si no la rechaza, eso es el fallo real (una
    entrada no finita se coló), no un éxito."""
    scenario = "corrupted" if corrupted else "normal"
    X = _synthetic_row(features, rng, corrupted)
    inicio = time.perf_counter()
    try:
        package.predict(X)
    except ValueError as exc:
        elapsed = time.perf_counter() - inicio
        return RequestResult(scenario, elapsed, success=corrupted, error_type=type(exc).__name__)
    except Exception as exc:  # noqa: BLE001
        return RequestResult(scenario, time.perf_counter() - inicio, success=False, error_type=type(exc).__name__)
    else:
        elapsed = time.perf_counter() - inicio
        if corrupted:
            # No hubo excepción sobre una fila con NaN/Inf: el rechazo que
            # se esperaba no ocurrió.
            return RequestResult(scenario, elapsed, success=False, error_type="not_rejected")
        return RequestResult(scenario, elapsed, success=True, error_type=None)


def _run_metrics_scrape(host: str, session: requests.Session) -> RequestResult:
    inicio = time.perf_counter()
    try:
        resp = session.get(f"{host}/metrics", timeout=5)
        resp.raise_for_status()
        return RequestResult("metrics", time.perf_counter() - inicio, success=True)
    except Exception as exc:  # noqa: BLE001
        return RequestResult("metrics", time.perf_counter() - inicio, success=False, error_type=type(exc).__name__)


def run_load_test(
    package: DetectorPackage,
    features: list[str],
    users: int = 20,
    spawn_rate: float = 10.0,
    run_time_seconds: float = 5.0,
    host: Optional[str] = None,
    corrupted_fraction: float = 0.1,
    metrics_fraction: float = 0.2,
    seed: int = 0,
    max_requests: Optional[int] = None,
) -> dict:
    """Lanza `users` hilos, cada uno disparando requests en loop (mezcla
    ponderada de los tres escenarios) durante `run_time_seconds`, con un
    ramp-up de `spawn_rate` hilos/segundo -- los mismos dos parámetros que
    usa Locust, con el mismo significado.

    `max_requests`, si se da, corta la corrida apenas se junta ese total
    (lo que pase primero entre eso y `run_time_seconds`) -- para una versión
    acotada y rápida en CI, sin depender de que el reloj corra el tiempo
    completo para juntar una cantidad fija de requests.
    """
    resultados: list[RequestResult] = []
    lock = threading.Lock()
    detener = threading.Event()
    metrics_habilitado = host is not None

    def worker(worker_id: int) -> None:
        rng = np.random.default_rng(seed + worker_id + 1)
        session = requests.Session() if metrics_habilitado else None
        while not detener.is_set():
            r = rng.random()
            if metrics_habilitado and r < metrics_fraction:
                resultado = _run_metrics_scrape(host, session)  # type: ignore[arg-type]
            elif r < (metrics_fraction if metrics_habilitado else 0.0) + corrupted_fraction:
                resultado = _run_inference_request(package, features, rng, corrupted=True)
            else:
                resultado = _run_inference_request(package, features, rng, corrupted=False)
            with lock:
                if detener.is_set():
                    break
                resultados.append(resultado)
                if max_requests is not None and len(resultados) >= max_requests:
                    detener.set()

    hilos = []
    intervalo_spawn = (1.0 / spawn_rate) if spawn_rate > 0 else 0.0
    inicio_prueba = time.perf_counter()
    for i in range(users):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        hilos.append(t)
        if intervalo_spawn:
            time.sleep(intervalo_spawn)

    while not detener.is_set():
        if time.perf_counter() - inicio_prueba >= run_time_seconds:
            detener.set()
            break
        time.sleep(0.005)
    for t in hilos:
        t.join(timeout=5)

    duracion_real = time.perf_counter() - inicio_prueba
    return _summarize(resultados, duracion_real)


def _summarize(resultados: list[RequestResult], duracion_segundos: float) -> dict:
    por_escenario: dict[str, dict] = {}
    for escenario in ("normal", "corrupted", "metrics"):
        subset = [r for r in resultados if r.scenario == escenario]
        if not subset:
            continue
        latencias_ms = [r.latency_seconds * 1000 for r in subset]
        exitosos = sum(r.success for r in subset)
        por_escenario[escenario] = {
            "total_requests": len(subset),
            "successful": exitosos,
            "error_rate": 1 - (exitosos / len(subset)),
            "p50_ms": _percentile(latencias_ms, 50),
            "p95_ms": _percentile(latencias_ms, 95),
            "p99_ms": _percentile(latencias_ms, 99),
        }

    total = len(resultados)
    normales = por_escenario.get("normal")
    tasa_error_valida = normales["error_rate"] if normales else 0.0

    return {
        "total_requests": total,
        "duration_seconds": duracion_segundos,
        "rps": (total / duracion_segundos) if duracion_segundos > 0 else 0.0,
        "error_rate_valid_traffic": tasa_error_valida,
        "error_rate_within_threshold": tasa_error_valida < VALID_ERROR_RATE_THRESHOLD,
        "scenarios": por_escenario,
    }


def _build_demo_package(seed: int = 0) -> tuple[DetectorPackage, list[str]]:
    """Paquete sintético autocontenido, para que la CLI corra sin el dataset
    PaySim real (493MB, no versionado) -- mismo patrón que usan los
    fixtures de `tests/test_serving.py`/`test_serving_metrics.py`."""
    from sklearn.preprocessing import RobustScaler

    from src.serving.package import build_package
    from src.unsupervised.families import GMMDensity

    features = ["monto", "saldo", "error", "hora"]
    rng = np.random.default_rng(seed)
    entrenamiento = rng.normal(size=(1_500, 4))
    calibracion = rng.normal(size=(1_500, 4))
    scaler = RobustScaler().fit(entrenamiento)
    detector = GMMDensity(n_components=3).fit(scaler.transform(entrenamiento))
    package = build_package(
        detector, scaler, scaler.transform(calibracion), features,
        alpha=0.01, detector_name="load_test_demo",
    )
    return package, features


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.load.load_harness",
        description="Prueba de carga concurrente sobre DetectorPackage y /metrics.",
    )
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--spawn-rate", type=float, default=10.0)
    parser.add_argument("--run-time", type=float, default=5.0, help="Duración en segundos.")
    parser.add_argument("--host", type=str, default=None,
                         help="URL base para el escenario /metrics (ej. http://127.0.0.1:8001). "
                              "Si se omite, ese escenario se salta.")
    parser.add_argument("--corrupted-fraction", type=float, default=0.1)
    parser.add_argument("--headless", action="store_true",
                         help="Aceptado por compatibilidad con la CLI de Locust -- este arnés siempre corre headless.")
    parser.add_argument(
        "--output-json", type=Path, nargs="?", const=DEFAULT_OUTPUT_PATH, default=None,
        help=f"Guarda el reporte en JSON (default si se pasa sin valor: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    package, features = _build_demo_package()

    reporte = run_load_test(
        package, features,
        users=args.users, spawn_rate=args.spawn_rate, run_time_seconds=args.run_time,
        host=args.host, corrupted_fraction=args.corrupted_fraction,
    )

    print(f"Total requests: {reporte['total_requests']}  RPS: {reporte['rps']:.1f}  "
          f"duración: {reporte['duration_seconds']:.2f}s")
    for nombre, datos in reporte["scenarios"].items():
        print(
            f"  {nombre:10s} n={datos['total_requests']:<6} error_rate={datos['error_rate']:.4%}  "
            f"p50={datos['p50_ms']:.2f}ms p95={datos['p95_ms']:.2f}ms p99={datos['p99_ms']:.2f}ms"
        )
    print(f"Tasa de error en tráfico válido: {reporte['error_rate_valid_traffic']:.4%} "
          f"({'OK' if reporte['error_rate_within_threshold'] else 'SUPERA EL UMBRAL DE 0.1%'})")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Reporte -> {args.output_json}")

    return 0 if reporte["error_rate_within_threshold"] else 1


if __name__ == "__main__":
    sys.exit(main())

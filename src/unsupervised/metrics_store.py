"""Persistencia de métricas comparativas de los modelos no supervisados en DuckDB.

Guarda una fila por (modelo, corrida) en una tabla local `data/processed/metrics.duckdb`,
para poder comparar resultados entre ejecuciones sin depender de los prints de consola.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "metrics.duckdb"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS unsupervised_metrics (
    run_ts TIMESTAMP,
    model_name VARCHAR,
    pr_auc DOUBLE,
    precision_at_50 DOUBLE,
    recall_at_50 DOUBLE,
    precision_at_100 DOUBLE,
    recall_at_100 DOUBLE,
    precision_at_200 DOUBLE,
    recall_at_200 DOUBLE
)
"""


def save_metrics(results: dict, db_path: Path = DB_PATH) -> None:
    """Inserta una fila por modelo con sus métricas (`pr_auc`, `precision_recall_at_k`).

    `results` sigue el formato producido por `train_unsupervised.evaluate`:
    {model_name: {"pr_auc": float, "precision_recall_at_k": {k: (precision, recall)}}}
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(CREATE_TABLE_SQL)
        run_ts = dt.datetime.now()
        for name, res in results.items():
            at_k = res["precision_recall_at_k"]
            p50, r50 = at_k.get(50, (None, None))
            p100, r100 = at_k.get(100, (None, None))
            p200, r200 = at_k.get(200, (None, None))
            con.execute(
                "INSERT INTO unsupervised_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_ts, name, res["pr_auc"], p50, r50, p100, r100, p200, r200],
            )
    finally:
        con.close()


def load_latest_metrics(db_path: Path = DB_PATH):
    """Devuelve la última corrida guardada como DataFrame (una fila por modelo)."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(CREATE_TABLE_SQL)
        return con.execute(
            """
            SELECT * FROM unsupervised_metrics
            WHERE run_ts = (SELECT max(run_ts) FROM unsupervised_metrics)
            ORDER BY pr_auc DESC
            """
        ).df()
    finally:
        con.close()


BENCHMARK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_metrics (
    run_ts TIMESTAMP,
    model_name VARCHAR,
    family VARCHAR,
    pr_auc DOUBLE,
    roc_auc DOUBLE,
    precision_at_50 DOUBLE,
    recall_at_50 DOUBLE,
    precision_at_100 DOUBLE,
    recall_at_100 DOUBLE,
    precision_at_200 DOUBLE,
    recall_at_200 DOUBLE,
    fit_seconds DOUBLE,
    score_seconds DOUBLE
)
"""


def save_benchmark_metrics(results: dict, model_labels: dict | None = None, db_path: Path = DB_PATH) -> None:
    """Persiste el benchmark del módulo 3 en su propia tabla.

    Va aparte de `unsupervised_metrics` porque agrega familia, ROC-AUC y tiempos de ajuste
    y scoring: mezclarlo en la tabla del módulo 2 obligaría a migrar los datos históricos.

    `results` sigue el formato producido por `benchmark.evaluate_all`.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(BENCHMARK_TABLE_SQL)
        run_ts = dt.datetime.now()
        labels = model_labels or {}
        for name, res in results.items():
            at_k = res["precision_recall_at_k"]
            p50, r50 = at_k.get(50, (None, None))
            p100, r100 = at_k.get(100, (None, None))
            p200, r200 = at_k.get(200, (None, None))
            con.execute(
                "INSERT INTO benchmark_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_ts,
                    labels.get(name, name),
                    res.get("family"),
                    res["pr_auc"],
                    res.get("roc_auc"),
                    p50, r50, p100, r100, p200, r200,
                    res.get("fit_seconds"),
                    res.get("score_seconds"),
                ],
            )
    finally:
        con.close()


def load_latest_benchmark(db_path: Path = DB_PATH):
    """Devuelve la última corrida del benchmark como DataFrame, ordenada por PR-AUC."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(BENCHMARK_TABLE_SQL)
        return con.execute(
            """
            SELECT * FROM benchmark_metrics
            WHERE run_ts = (SELECT max(run_ts) FROM benchmark_metrics)
            ORDER BY pr_auc DESC
            """
        ).df()
    finally:
        con.close()

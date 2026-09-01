"""Pruebas unitarias para src.unsupervised.metrics_store (persistencia DuckDB)."""
from pathlib import Path

from src.unsupervised.metrics_store import load_latest_metrics, save_metrics


def _sample_results() -> dict:
    return {
        "isolation_forest": {
            "pr_auc": 0.42,
            "precision_recall_at_k": {50: (0.8, 0.1), 100: (0.7, 0.2), 200: (0.6, 0.3)},
        },
        "mad_baseline": {
            "pr_auc": 0.20,
            "precision_recall_at_k": {50: (0.4, 0.05), 100: (0.35, 0.1), 200: (0.3, 0.15)},
        },
    }


def test_save_and_load_metrics_roundtrip(tmp_path: Path):
    db_path = tmp_path / "metrics.duckdb"
    save_metrics(_sample_results(), db_path=db_path)

    df = load_latest_metrics(db_path=db_path)

    assert len(df) == 2
    assert set(df["model_name"]) == {"isolation_forest", "mad_baseline"}
    # Ordenado descendente por pr_auc: isolation_forest (0.42) debe ir primero.
    assert df.iloc[0]["model_name"] == "isolation_forest"


def test_save_metrics_appends_across_runs(tmp_path: Path):
    db_path = tmp_path / "metrics.duckdb"
    save_metrics(_sample_results(), db_path=db_path)
    save_metrics(_sample_results(), db_path=db_path)

    import duckdb

    con = duckdb.connect(str(db_path))
    total_rows = con.execute("SELECT count(*) FROM unsupervised_metrics").fetchone()[0]
    con.close()

    assert total_rows == 4

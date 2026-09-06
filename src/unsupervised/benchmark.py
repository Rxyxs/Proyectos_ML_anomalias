"""Módulo 3 — benchmark comparativo de todas las familias de detección de anomalías.

Entrena, en el mismo split y con el mismo escalado, los detectores del módulo 2
(Isolation Forest, LOF, MAD-z, autoencoder) junto con las siete familias complementarias
de `families.py`, y encima de todos ellos los ensembles de `ensemble.py`.

La pregunta que responde no es "cuál es el mejor modelo" sino dos más útiles para decidir
qué desplegar:

1. **¿Qué familias son realmente complementarias?** Si dos detectores ordenan las
   transacciones casi igual (correlación de Spearman alta entre sus scores), tener los dos
   no aporta nada. La matriz de correlación lo muestra explícitamente.
2. **¿Cuánto cuesta cada punto de PR-AUC?** Se cronometra ajuste y scoring por separado:
   en producción el scoring corre por transacción y el ajuste una vez al día, así que un
   detector lento de entrenar pero rápido de puntuar es perfectamente viable — y al revés
   no.

Ejecución:

    python -m src.unsupervised.benchmark
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.unsupervised.autoencoder import reconstruction_error, train_autoencoder
from src.unsupervised.ensemble import build_ensembles
from src.unsupervised.families import build_detectors
from src.unsupervised.loader import get_unsupervised_data
from src.unsupervised.metrics_store import save_benchmark_metrics
from src.unsupervised.models import anomaly_score, build_isolation_forest, build_lof, build_mad_baseline
from src.unsupervised.style import GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, style_axes
from src.unsupervised.train_unsupervised import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
RANKING_FIGURE_PATH = FIGURES_DIR / "benchmark_ranking.png"
BENCHMARK_PR_FIGURE_PATH = FIGURES_DIR / "benchmark_pr_curve.png"
CORRELATION_FIGURE_PATH = FIGURES_DIR / "benchmark_correlation.png"

# Familia de cada detector: agrupa el ranking por enfoque, no por nombre de librería.
MODEL_FAMILY = {
    "isolation_forest": "Aislamiento",
    "lof": "Densidad local",
    "knn_distance": "Distancia global",
    "mad_baseline": "Estadístico por feature",
    "hbos": "Estadístico por feature",
    "ecod": "Estadístico por feature",
    "robust_mahalanobis": "Covarianza robusta",
    "gmm_density": "Densidad paramétrica",
    "ocsvm_nystroem": "Frontera con kernel",
    "pca_reconstruction": "Reconstrucción",
    "autoencoder": "Reconstrucción",
    "ensemble_rank_avg": "Ensemble",
    "ensemble_z_avg": "Ensemble",
    "ensemble_z_max": "Ensemble",
}

FAMILY_COLORS = {
    "Aislamiento": "#2a78d6",
    "Densidad local": "#eb6834",
    "Distancia global": "#00897b",
    "Estadístico por feature": "#8a8a8a",
    "Covarianza robusta": "#6d4c41",
    "Densidad paramétrica": "#9c27b0",
    "Frontera con kernel": "#f9a825",
    "Reconstrucción": "#4caf50",
    "Ensemble": "#c2185b",
}

MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "mad_baseline": "MAD-z (baseline)",
    "autoencoder": "Autoencoder (ReLU)",
    "pca_reconstruction": "PCA (reconstrucción)",
    "gmm_density": "Gaussian Mixture",
    "robust_mahalanobis": "Mahalanobis robusto (MCD)",
    "knn_distance": "kNN (k-ésima distancia)",
    "ocsvm_nystroem": "One-Class SVM (Nyström)",
    "hbos": "HBOS",
    "ecod": "ECOD",
    "ensemble_rank_avg": "Ensemble — promedio de rangos",
    "ensemble_z_avg": "Ensemble — promedio z",
    "ensemble_z_max": "Ensemble — máximo z",
}


def run_detectors(X_train: np.ndarray, X_test: np.ndarray) -> dict[str, dict]:
    """Ajusta cada detector con datos normales y puntúa el set de prueba, cronometrando ambas fases."""
    detectors = {
        "isolation_forest": build_isolation_forest(),
        "lof": build_lof(),
        "mad_baseline": build_mad_baseline(),
        **build_detectors(),
    }

    outputs: dict[str, dict] = {}
    for name, detector in detectors.items():
        print(f"  [{name}] ajustando...", flush=True)
        start = time.perf_counter()
        detector.fit(X_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scores = anomaly_score(detector, X_test)
        score_seconds = time.perf_counter() - start

        outputs[name] = {"scores": scores, "fit_seconds": fit_seconds, "score_seconds": score_seconds}
        print(f"  [{name}] fit {fit_seconds:.1f}s | score {score_seconds:.1f}s", flush=True)

    # El autoencoder no comparte la API score_samples, se ejecuta aparte con la misma
    # instrumentación para que sus tiempos sean comparables con los demás.
    print("  [autoencoder] ajustando...", flush=True)
    start = time.perf_counter()
    model = train_autoencoder(X_train.astype("float32"), activation="relu")
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores = reconstruction_error(model, X_test.astype("float32"))
    score_seconds = time.perf_counter() - start

    outputs["autoencoder"] = {"scores": scores, "fit_seconds": fit_seconds, "score_seconds": score_seconds}
    print(f"  [autoencoder] fit {fit_seconds:.1f}s | score {score_seconds:.1f}s", flush=True)
    return outputs


def evaluate_all(outputs: dict[str, dict], y_test: pd.Series) -> dict[str, dict]:
    """Agrega PR-AUC, ROC-AUC y precision/recall@k a cada entrada de `outputs`."""
    results = {}
    for name, output in outputs.items():
        metrics = evaluate(y_test, output["scores"])
        metrics["roc_auc"] = roc_auc_score(y_test, output["scores"])
        results[name] = {**output, **metrics, "family": MODEL_FAMILY.get(name, "Otro")}
    return results


def plot_ranking(results: dict[str, dict], output_path: Path = RANKING_FIGURE_PATH):
    """Ranking horizontal de PR-AUC, coloreado por familia de detector."""
    ordered = sorted(results.items(), key=lambda item: item[1]["pr_auc"])
    labels = [MODEL_LABELS.get(name, name) for name, _ in ordered]
    values = [res["pr_auc"] for _, res in ordered]
    colors = [FAMILY_COLORS.get(res["family"], INK_MUTED) for _, res in ordered]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(ordered) + 2.2))
    ax.barh(labels, values, color=colors)
    for y, value in enumerate(values):
        ax.text(value + max(values) * 0.01, y, f"{value:.3f}", va="center", color=INK_SECONDARY, fontsize=9)

    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("PR-AUC (más alto = mejor)", color=INK_SECONDARY)
    ax.set_title("Detección de anomalías — PR-AUC por familia de detector", color=INK_PRIMARY, fontsize=13)
    style_axes(ax)
    ax.grid(axis="y", visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in FAMILY_COLORS.values()]
    ax.legend(handles, list(FAMILY_COLORS), frameon=False, fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_pr_curves(results: dict[str, dict], y_test: pd.Series, top_n: int = 6,
                   output_path: Path = BENCHMARK_PR_FIGURE_PATH):
    """Curvas Precision-Recall de los `top_n` detectores con mejor PR-AUC."""
    best = sorted(results.items(), key=lambda item: item[1]["pr_auc"], reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for name, res in best:
        precision, recall, _ = precision_recall_curve(y_test, res["scores"])
        ax.plot(
            recall, precision,
            color=FAMILY_COLORS.get(res["family"], INK_SECONDARY),
            linewidth=2,
            label=f"{MODEL_LABELS.get(name, name)} (PR-AUC={res['pr_auc']:.3f})",
        )

    baseline = y_test.mean()
    ax.axhline(baseline, linestyle="--", linewidth=1, color=INK_MUTED, label=f"Azar ({baseline:.4f})")
    ax.set_xlabel("Recall", color=INK_SECONDARY)
    ax.set_ylabel("Precision", color=INK_SECONDARY)
    ax.set_title(f"Curva Precision-Recall — {top_n} mejores detectores", color=INK_PRIMARY, fontsize=13)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def score_correlation(results: dict[str, dict]) -> pd.DataFrame:
    """Correlación de Spearman entre los rankings de anomalía de cada detector base."""
    base = {name: res["scores"] for name, res in results.items() if res["family"] != "Ensemble"}
    matrix = np.column_stack(list(base.values()))
    rho, _ = spearmanr(matrix)
    labels = [MODEL_LABELS.get(name, name) for name in base]
    return pd.DataFrame(np.atleast_2d(rho), index=labels, columns=labels)


def plot_correlation(corr: pd.DataFrame, output_path: Path = CORRELATION_FIGURE_PATH):
    """Mapa de calor de la correlación entre detectores: verde = complementarios, rojo = redundantes."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", color=INK_SECONDARY, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, color=INK_SECONDARY, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else INK_PRIMARY, fontsize=7)

    ax.set_title("Correlación de Spearman entre anomaly scores", color=INK_PRIMARY, fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8).outline.set_edgecolor(GRIDLINE)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def summary_table(results: dict[str, dict]) -> pd.DataFrame:
    """Tabla ordenada por PR-AUC con métricas y costo computacional por detector."""
    rows = []
    for name, res in results.items():
        p100, r100 = res["precision_recall_at_k"][100]
        rows.append({
            "modelo": MODEL_LABELS.get(name, name),
            "familia": res["family"],
            "pr_auc": res["pr_auc"],
            "roc_auc": res["roc_auc"],
            "precision@100": p100,
            "recall@100": r100,
            "fit_s": res["fit_seconds"],
            "score_s": res["score_seconds"],
        })
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Cargando datos (entrenamiento 100% normal, prueba mixta)...")
    X_train, X_test, y_test = get_unsupervised_data()
    print(f"Train (solo normales): {X_train.shape} | Test (mixto): {X_test.shape}, fraude={int(y_test.sum())}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nEntrenando detectores...")
    outputs = run_detectors(X_train_scaled, X_test_scaled)

    # Los ensembles se construyen sobre los scores ya calculados: su costo de ajuste es
    # cero y el de scoring es solo la agregación de columnas.
    base_scores = {name: output["scores"] for name, output in outputs.items()}
    start = time.perf_counter()
    ensembles = build_ensembles(base_scores)
    ensemble_seconds = time.perf_counter() - start
    for name, scores in ensembles.items():
        outputs[name] = {
            "scores": scores,
            "fit_seconds": 0.0,
            "score_seconds": ensemble_seconds / len(ensembles),
        }

    results = evaluate_all(outputs, y_test)

    table = summary_table(results)
    print("\n=== Benchmark de detección de anomalías (ordenado por PR-AUC) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    corr = score_correlation(results)
    print("\nPares de detectores más redundantes (Spearman > 0.9):")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    redundant = upper[upper > 0.9].sort_values(ascending=False)
    print(redundant.to_string() if not redundant.empty else "  ninguno — todas las familias aportan orden distinto")

    plot_ranking(results)
    plot_pr_curves(results, y_test)
    plot_correlation(corr)
    plt.close("all")
    print(f"\nRanking:      {RANKING_FIGURE_PATH}")
    print(f"Curvas PR:    {BENCHMARK_PR_FIGURE_PATH}")
    print(f"Correlación:  {CORRELATION_FIGURE_PATH}")

    save_benchmark_metrics(results, model_labels=MODEL_LABELS)
    print("\nMétricas del benchmark guardadas en data/processed/metrics.duckdb (tabla benchmark_metrics)")

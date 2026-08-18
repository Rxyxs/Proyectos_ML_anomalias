"""Entrenamiento y evaluación de modelos no supervisados de detección de anomalías.

Los modelos se ajustan únicamente con transacciones normales y se evalúan sobre un
conjunto de prueba mixto, simulando la detección de fraude "zero-day": patrones nuevos
que un modelo supervisado (entrenado solo con fraude ya etiquetado) no podría reconocer.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import RobustScaler

from src.unsupervised.loader import get_unsupervised_data
from src.unsupervised.models import anomaly_score, build_isolation_forest, build_lof

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "isolation_forest.joblib"
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
SCORES_FIGURE_PATH = FIGURES_DIR / "unsupervised_scores.png"
PR_CURVE_FIGURE_PATH = FIGURES_DIR / "unsupervised_pr_curve.png"

K_VALUES = [50, 100, 200]

MODEL_LABELS = {"isolation_forest": "Isolation Forest", "lof": "Local Outlier Factor"}
MODEL_COLORS = {"isolation_forest": "#2a78d6", "lof": "#eb6834"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def precision_recall_at_k(y_true: pd.Series, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Precision@k y recall@k: de las k transacciones con mayor anomaly score, cuántas son fraude real."""
    y_true_arr = np.asarray(y_true)
    k = min(k, len(y_true_arr))
    top_k_idx = np.argsort(scores)[::-1][:k]
    hits = y_true_arr[top_k_idx].sum()
    precision = hits / k
    recall = hits / y_true_arr.sum()
    return precision, recall


def evaluate(y_test: pd.Series, scores: np.ndarray) -> dict:
    """Métricas de evaluación: PR-AUC global y precision/recall en distintos puntos de corte k."""
    pr_auc = average_precision_score(y_test, scores)
    at_k = {k: precision_recall_at_k(y_test, scores, k) for k in K_VALUES}
    return {"pr_auc": pr_auc, "precision_recall_at_k": at_k}


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_score_distributions(results: dict, y_test: pd.Series, output_path: Path = SCORES_FIGURE_PATH):
    """Histograma comparativo del anomaly score para normales vs. fraude, uno por modelo."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        scores = res["scores"]
        # Recorta el rango visible a percentiles (no los datos) para que una cola extrema
        # no comprima la zona donde realmente se separan las dos distribuciones.
        x_min, x_max = np.percentile(scores, [0.5, 99.5])
        use_log = x_min > 0

        if use_log:
            bins = np.geomspace(x_min, x_max, 60)
            ax.set_xscale("log")
        else:
            bins = np.linspace(x_min, x_max, 60)

        ax.hist(scores[y_test.values == 0], bins=bins, alpha=0.6, density=True, label="No fraude", color="#2a78d6")
        ax.hist(scores[y_test.values == 1], bins=bins, alpha=0.6, density=True, label="Fraude", color="#eb6834")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Anomaly score (más alto = más anómalo)" + (" — escala log" if use_log else ""), color=INK_SECONDARY)
        ax.set_ylabel("Densidad", color=INK_SECONDARY)
        ax.set_title(MODEL_LABELS.get(name, name), color=INK_PRIMARY, fontsize=12)
        _style_axes(ax)
        ax.legend(frameon=False)

    fig.suptitle("Distribución del Anomaly Score — normal vs. fraude", color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_pr_curve(results: dict, y_test: pd.Series, output_path: Path = PR_CURVE_FIGURE_PATH):
    """Curva Precision-Recall comparativa entre los modelos no supervisados."""
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_test.mean()

    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res["scores"])
        ax.plot(
            recall, precision,
            color=MODEL_COLORS.get(name, INK_SECONDARY),
            linewidth=2,
            label=f"{MODEL_LABELS.get(name, name)} (PR-AUC={res['pr_auc']:.3f})",
        )

    ax.axhline(baseline, linestyle="--", linewidth=1, color=INK_MUTED, label=f"Baseline (azar={baseline:.4f})")
    ax.set_xlabel("Recall", color=INK_SECONDARY)
    ax.set_ylabel("Precision", color=INK_SECONDARY)
    ax.set_title("Curva Precision-Recall — modelos no supervisados", color=INK_PRIMARY, fontsize=13)
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("Cargando datos (entrenamiento 100% normal, prueba mixta)...")
    X_train, X_test, y_test = get_unsupervised_data()
    print(f"Train (solo normales): {X_train.shape} | Test (mixto): {X_test.shape}, fraude={int(y_test.sum())}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "isolation_forest": build_isolation_forest(),
        "lof": build_lof(),
    }

    results = {}
    for name, model in models.items():
        print(f"\nEntrenando {MODEL_LABELS[name]}...")
        model.fit(X_train_scaled)
        scores = anomaly_score(model, X_test_scaled)
        metrics = evaluate(y_test, scores)
        results[name] = {"model": model, "scores": scores, **metrics}

        print(f"--- {MODEL_LABELS[name]} ---")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        for k, (precision, recall) in metrics["precision_recall_at_k"].items():
            print(f"  Precision@{k}: {precision:.4f} | Recall@{k}: {recall:.4f}")

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    print(f"\nMejor modelo por PR-AUC: {MODEL_LABELS[best_name]} ({results[best_name]['pr_auc']:.4f})")

    plot_score_distributions(results, y_test)
    plot_pr_curve(results, y_test)
    plt.close("all")
    print(f"\nGráfica de distribución de anomaly scores: {SCORES_FIGURE_PATH}")
    print(f"Gráfica de curva Precision-Recall: {PR_CURVE_FIGURE_PATH}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": results["isolation_forest"]["model"], "scaler": scaler}, MODEL_OUTPUT_PATH)
    print(f"\nIsolation Forest (+ scaler) guardado en {MODEL_OUTPUT_PATH}")

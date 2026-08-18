"""Gráficas comparativas para evaluar los modelos candidatos de detección de fraude."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

FIGURES_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "figures"

MODEL_LABELS = {
    "logistic_regression": "Regresión Logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "logistic_regression": "#2a78d6",
    "random_forest": "#eb6834",
    "xgboost": "#1baf7a",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc_curves(results: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: Path | None = None):
    """Dibuja las curvas ROC de todos los modelos entrenados en una sola figura."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, metrics in results.items():
        y_proba = metrics["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(
            fpr, tpr,
            color=MODEL_COLORS.get(name, INK_SECONDARY),
            linewidth=2,
            label=f"{MODEL_LABELS.get(name, name)} (AUC={metrics['roc_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=INK_MUTED, label="Azar (AUC=0.5)")
    ax.set_xlabel("Tasa de falsos positivos", color=INK_SECONDARY)
    ax.set_ylabel("Tasa de verdaderos positivos", color=INK_SECONDARY)
    ax.set_title("Curvas ROC — comparación de modelos", color=INK_PRIMARY, fontsize=13)
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_pr_curves(results: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: Path | None = None):
    """Dibuja las curvas Precision-Recall (más informativas que ROC bajo desbalance extremo)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_test.mean()

    for name, metrics in results.items():
        y_proba = metrics["model"].predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(
            recall, precision,
            color=MODEL_COLORS.get(name, INK_SECONDARY),
            linewidth=2,
            label=f"{MODEL_LABELS.get(name, name)} (PR-AUC={metrics['pr_auc']:.3f})",
        )

    ax.axhline(baseline, linestyle="--", linewidth=1, color=INK_MUTED, label=f"Baseline (azar={baseline:.4f})")
    ax.set_xlabel("Recall", color=INK_SECONDARY)
    ax.set_ylabel("Precision", color=INK_SECONDARY)
    ax.set_title("Curvas Precision-Recall — comparación de modelos", color=INK_PRIMARY, fontsize=13)
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_confusion_matrices(results: dict, output_path: Path | None = None):
    """Dibuja la matriz de confusión de cada modelo, normalizada por fila para lectura visual."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, results.items()):
        cm = metrics["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        ax.imshow(cm_norm, cmap=BLUE_SEQUENTIAL, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm_norm[i, j] > 0.6 else INK_PRIMARY
                ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})", ha="center", va="center", color=text_color, fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No fraude", "Fraude"], color=INK_SECONDARY)
        ax.set_yticklabels(["No fraude", "Fraude"], color=INK_SECONDARY)
        ax.set_xlabel("Predicho", color=INK_SECONDARY)
        ax.set_ylabel("Real", color=INK_SECONDARY)
        ax.set_title(MODEL_LABELS.get(name, name), color=INK_PRIMARY, fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Matrices de confusión (normalizadas por fila)", color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def generate_comparison_report(
    results: dict, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path = FIGURES_DIR
) -> dict[str, Path]:
    """Genera y guarda las tres figuras comparativas (ROC, PR, matrices de confusión)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "roc_curves": output_dir / "roc_curves.png",
        "pr_curves": output_dir / "pr_curves.png",
        "confusion_matrices": output_dir / "confusion_matrices.png",
    }

    plot_roc_curves(results, X_test, y_test, paths["roc_curves"])
    plot_pr_curves(results, X_test, y_test, paths["pr_curves"])
    plot_confusion_matrices(results, paths["confusion_matrices"])
    plt.close("all")

    return paths

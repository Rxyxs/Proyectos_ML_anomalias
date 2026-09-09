"""Módulo 7 — deriva y etiquetas: los dos problemas que los módulos anteriores dejaron abiertos.

Los Módulos 5 y 6 concluyeron, por dos caminos independientes, que ningún umbral fijo
aprendido del pasado sobrevive al paso del tiempo. Ninguno propuso qué hacer al respecto.
Y los seis módulos previos operan en uno de dos extremos —todas las etiquetas o ninguna—
cuando el caso real es tener unas pocas.

Este módulo responde las dos cosas con tres experimentos:

1. **Half-Space Trees con ventana deslizante**: ¿un detector que se actualiza solo aguanta
   la deriva mejor que los quince que se ajustan una vez?
2. **Apilado semi-supervisado**: si aparecen 50, 100 o 1.000 etiquetas, ¿cuánto aportan los
   scores de los quince detectores como features de un clasificador?
3. **Aprendizaje activo**: la capacidad de revisión produce etiquetas. ¿En qué conviene
   gastarla para detectar mejor mañana, y no solo hoy?

Ejecución:

    python -m src.adaptive.run_adaptive
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.adaptive.active import compare_strategies
from src.adaptive.hs_trees import HalfSpaceTrees
from src.adaptive.stacking import label_budget_curve, stack_scores
from src.operations.temporal import get_temporal_data, period_index
from src.unsupervised.benchmark import MODEL_LABELS, run_detectors
from src.unsupervised.models import anomaly_score
from src.unsupervised.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
STREAMING_FIGURE_PATH = FIGURES_DIR / "adaptive_streaming.png"
BUDGET_FIGURE_PATH = FIGURES_DIR / "adaptive_label_budget.png"
ACTIVE_FIGURE_PATH = FIGURES_DIR / "adaptive_active_learning.png"

# Detector no supervisado con el que se ordena la cola de revisión antes de tener etiquetas.
# Se elige Gaussian Mixture por ser el mejor calibrado del Módulo 5, no el de mejor PR-AUC:
# la cola se trabaja de arriba hacia abajo y ahí importa la precisión en la cabeza.
RANKING_DETECTOR = "gmm_density"

BUDGETS = (50, 100, 200, 500, 1_000, 5_000)


def streaming_experiment(X_test_scaled: np.ndarray, y_test, steps, cutoff_step: int,
                         X_train_scaled: np.ndarray) -> pd.DataFrame:
    """Half-Space Trees con ventana fija contra ventana que se refresca cada día.

    Ambas variantes comparten árboles, semilla y ventana inicial: la única diferencia es que
    una actualiza su perfil de masa con el tráfico del día anterior y la otra no.
    """
    periods = period_index(steps, cutoff_step)
    y_arr = np.asarray(y_test)

    estatico = HalfSpaceTrees(random_state=42).fit(X_train_scaled)
    adaptativo = HalfSpaceTrees(random_state=42).fit(X_train_scaled)

    rows = []
    for period in np.unique(periods):
        selected = periods == period
        n_period = int(selected.sum())
        positives = int(y_arr[selected].sum())
        if positives < 5 or n_period < 1_000:
            continue

        X_period = X_test_scaled[selected]
        y_period = y_arr[selected]

        rows.append({
            "dia": int(period),
            "n": n_period,
            "roc_auc_estatico": roc_auc_score(y_period, anomaly_score(estatico, X_period)),
            "roc_auc_adaptativo": roc_auc_score(y_period, anomaly_score(adaptativo, X_period)),
        })

        # El detector adaptativo se refresca con el tráfico del día que acaba de pasar, que es
        # información disponible sin ninguna etiqueta.
        adaptativo.update_window(X_period)

    return pd.DataFrame(rows)


def plot_streaming(tabla: pd.DataFrame, output_path: Path = STREAMING_FIGURE_PATH):
    """ROC-AUC diario del detector con ventana fija contra el que la refresca."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(tabla["dia"], tabla["roc_auc_estatico"], linewidth=2, marker="o", markersize=4,
            label="Ventana fija (ajustada una vez)")
    ax.plot(tabla["dia"], tabla["roc_auc_adaptativo"], linewidth=2, marker="o", markersize=4,
            label="Ventana refrescada cada día")

    ax.set_xlabel("Días después del corte temporal", color=INK_SECONDARY)
    ax.set_ylabel("ROC-AUC del día", color=INK_SECONDARY)
    ax.set_title("Half-Space Trees: adaptarse cuesta una pasada lineal",
                 color=INK_PRIMARY, fontsize=13)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_label_budget(tabla: pd.DataFrame, output_path: Path = BUDGET_FIGURE_PATH):
    """PR-AUC contra presupuesto de etiquetado, por estrategia y conjunto de features."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, estrategia, titulo in [
        (axes[0], "top", "Etiquetando la cola de alertas"),
        (axes[1], "random", "Etiquetando al azar"),
    ]:
        sub = tabla[tabla["estrategia"] == estrategia]
        for features, grupo in sub.groupby("features"):
            ax.plot(grupo["presupuesto"], grupo["pr_auc"], linewidth=2, marker="o",
                    label=features)
        ax.set_xscale("log")
        ax.set_xlabel("Transacciones etiquetadas", color=INK_SECONDARY)
        ax.set_title(titulo, color=INK_PRIMARY, fontsize=12)
        style_axes(ax)

    axes[0].set_ylabel("PR-AUC en el período tardío", color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Los scores de los 15 detectores como features de un clasificador",
                 color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_active_learning(tabla: pd.DataFrame, output_path: Path = ACTIVE_FIGURE_PATH):
    """Curvas de aprendizaje por estrategia de selección, y fraude encontrado en el camino."""
    fig, (ax_pr, ax_fraude) = plt.subplots(1, 2, figsize=(13, 5))

    for estrategia, grupo in tabla.groupby("estrategia"):
        ax_pr.plot(grupo["etiquetas"], grupo["pr_auc"], linewidth=2, marker="o", label=estrategia)
        ax_fraude.plot(grupo["etiquetas"], grupo["fraudes_encontrados"], linewidth=2,
                       marker="o", label=estrategia)

    ax_pr.set_xlabel("Transacciones revisadas", color=INK_SECONDARY)
    ax_pr.set_ylabel("PR-AUC en el período tardío", color=INK_SECONDARY)
    ax_pr.set_title("Cuánto se aprende", color=INK_PRIMARY, fontsize=12)
    style_axes(ax_pr)
    ax_pr.legend(frameon=False, fontsize=9)

    ax_fraude.set_xlabel("Transacciones revisadas", color=INK_SECONDARY)
    ax_fraude.set_ylabel("Fraudes encontrados al revisar", color=INK_SECONDARY)
    ax_fraude.set_title("Cuánto se atrapa en el camino", color=INK_PRIMARY, fontsize=12)
    style_axes(ax_fraude)
    ax_fraude.legend(frameon=False, fontsize=9)

    fig.suptitle("Explotar la cola de alertas contra explorar donde el modelo duda",
                 color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("Cargando el split temporal...")
    data = get_temporal_data()
    y_test, steps = data["y_test"], data["steps_test"]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(data["X_train"])
    X_test_scaled = scaler.transform(data["X_test"])
    X_early_scaled = scaler.transform(data["X_early"])

    print(f"Train (normales): {X_train_scaled.shape} | Etiquetado temprano: {X_early_scaled.shape}, "
          f"fraude={int(data['y_early'].sum())} ({data['y_early'].mean():.4%})")
    print(f"Test (tardío): {X_test_scaled.shape}, fraude={int(y_test.sum())} ({y_test.mean():.4%})")

    # ------------------------------------------------------------ 1. streaming
    print("\n=== 1. Half-Space Trees: ventana fija contra ventana refrescada ===")
    streaming = streaming_experiment(X_test_scaled, y_test, steps, data["cutoff_step"],
                                     X_train_scaled)
    print(streaming.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nPromedio  estático: {streaming['roc_auc_estatico'].mean():.4f} | "
          f"adaptativo: {streaming['roc_auc_adaptativo'].mean():.4f}")

    # ------------------------------------------------------------ 2. apilado
    print("\nEntrenando los 15 detectores para usarlos como features...")
    results = run_detectors(X_train_scaled, X_test_scaled)
    scores_test = {name: out["scores"] for name, out in results.items()}
    scores_early = {name: out["score_fn"](X_early_scaled) for name, out in results.items()}

    X_early_stacked = stack_scores(X_early_scaled, scores_early)
    X_test_stacked = stack_scores(X_test_scaled, scores_test)
    print(f"Features: {X_early_scaled.shape[1]} originales -> {X_early_stacked.shape[1]} apiladas")

    print("\n=== 2. Cuántas etiquetas hacen falta, y cómo gastarlas ===")
    budget = label_budget_curve(
        X_early_scaled, X_early_stacked, data["y_early"],
        X_test_scaled, X_test_stacked, y_test,
        ranking_scores=scores_early[RANKING_DETECTOR], budgets=BUDGETS,
    )
    print(budget.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    referencia = average_precision_score(y_test, scores_test[RANKING_DETECTOR])
    print(f"\nReferencia sin etiquetas ({MODEL_LABELS.get(RANKING_DETECTOR)}): PR-AUC={referencia:.4f}")

    # ------------------------------------------------------------ 3. activo
    print("\n=== 3. Aprendizaje activo: en qué gastar la capacidad de revisión ===")
    activo = compare_strategies(
        X_early_stacked, data["y_early"], X_test_stacked, y_test,
        unsupervised_scores=scores_early[RANKING_DETECTOR],
        n_rounds=8, batch_size=50,
    )
    resumen = activo.pivot(index="etiquetas", columns="estrategia", values="pr_auc")
    print(resumen.to_string(float_format=lambda v: f"{v:.4f}"))

    encontrados = activo.pivot(index="etiquetas", columns="estrategia", values="fraudes_encontrados")
    print("\nFraudes encontrados al revisar:")
    print(encontrados.to_string())

    plot_streaming(streaming)
    plot_label_budget(budget)
    plot_active_learning(activo)
    plt.close("all")
    print(f"\nStreaming:  {STREAMING_FIGURE_PATH}")
    print(f"Etiquetas:  {BUDGET_FIGURE_PATH}")
    print(f"Activo:     {ACTIVE_FIGURE_PATH}")

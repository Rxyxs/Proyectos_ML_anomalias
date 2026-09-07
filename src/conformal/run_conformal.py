"""Módulo 6 — ¿la garantía conforme sobrevive a la deriva temporal?

El Módulo 5 mostró que el umbral por cuantil se rompe: Deep SVDD promete 0,1% de falsas
alarmas y entrega 35%. La detección conforme promete una garantía en muestra finita sobre
esa misma cantidad. La pregunta de este módulo es si esa promesa se cumple sobre PaySim, y
la respuesta depende enteramente de un supuesto que se puede aislar experimentalmente.

Se comparan dos escenarios con **el mismo detector, el mismo ajuste y la misma calibración**,
cambiando solo de dónde salen las transacciones evaluadas:

- **Intercambiable**: calibración y evaluación provienen del mismo período temprano. El
  supuesto que sostiene la garantía se cumple por construcción, así que la tasa observada
  debería quedar en torno a α.
- **Con deriva**: la calibración sigue siendo del período temprano y la evaluación pasa al
  período posterior. Es el escenario real de producción — y el que rompe el supuesto.

La diferencia entre ambas columnas es el aporte del módulo: aísla cuánto del desajuste de
umbrales del Módulo 5 viene de la deriva y cuánto del método de calibración.

Ejecución:

    python -m src.conformal.run_conformal
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.conformal.conformal import conformal_p_values, coverage_report
from src.operations.temporal import get_temporal_data
from src.unsupervised.benchmark import MODEL_LABELS, run_detectors
from src.unsupervised.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
COVERAGE_FIGURE_PATH = FIGURES_DIR / "conformal_coverage.png"
PVALUE_FIGURE_PATH = FIGURES_DIR / "conformal_pvalues.png"

ALPHAS = (0.001, 0.005, 0.01, 0.05)

# Detectores elegidos por lo que representan, no por su puesto en el ranking: el mejor
# calibrado del Módulo 5 (GMM), el peor calibrado (Deep SVDD), uno intermedio (Mahalanobis)
# y uno de los nuevos del Módulo 6 (LODA).
DETECTORES = ["gmm_density", "deep_svdd", "robust_mahalanobis", "loda"]


def split_calibration(scores: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Parte los scores de calibración en dos mitades intercambiables entre sí.

    Una mitad calibra y la otra hace de conjunto de evaluación "sin deriva": provienen del
    mismo período y del mismo muestreo, así que la intercambiabilidad se cumple por
    construcción y la garantía debería observarse.
    """
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(scores)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def compare_scenarios(calib_scores: np.ndarray, test_scores: np.ndarray, y_test) -> pd.DataFrame:
    """Cobertura conforme en el escenario intercambiable y en el escenario con deriva."""
    calibration, held_out = split_calibration(calib_scores)

    # Escenario intercambiable: las "legítimas" son la otra mitad de la calibración.
    sin_deriva = coverage_report(
        calibration, held_out, np.zeros(len(held_out), dtype=int), alphas=ALPHAS
    )
    con_deriva = coverage_report(calibration, test_scores, y_test, alphas=ALPHAS)

    return pd.DataFrame({
        "alpha": con_deriva["alpha"],
        "fpr_intercambiable": sin_deriva["fpr_observada"],
        "razon_intercambiable": sin_deriva["razon_fpr_alpha"],
        "fpr_con_deriva": con_deriva["fpr_observada"],
        "razon_con_deriva": con_deriva["razon_fpr_alpha"],
        "alertas": con_deriva["alertas"],
        "recall": con_deriva["recall"],
        "precision": con_deriva["precision"],
    })


def plot_coverage(comparaciones: dict[str, pd.DataFrame], output_path: Path = COVERAGE_FIGURE_PATH):
    """Tasa de falsas alarmas observada contra la garantizada, en los dos escenarios."""
    fig, (ax_ok, ax_drift) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, columna, titulo in [
        (ax_ok, "fpr_intercambiable", "Calibración y evaluación del mismo período"),
        (ax_drift, "fpr_con_deriva", "Evaluación en el período posterior"),
    ]:
        for name, tabla in comparaciones.items():
            ax.plot(tabla["alpha"], tabla[columna], linewidth=2, marker="o",
                    label=MODEL_LABELS.get(name, name))

        lims = [5e-4, 1.0]
        ax.plot(lims, lims, linestyle="--", linewidth=1, color=INK_MUTED, label="Garantía (α)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lims)
        ax.set_xlabel("α garantizado", color=INK_SECONDARY)
        ax.set_title(titulo, color=INK_PRIMARY, fontsize=12)
        style_axes(ax)

    ax_ok.set_ylabel("Tasa de falsas alarmas observada", color=INK_SECONDARY)
    ax_ok.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("La garantía conforme vale mientras haya intercambiabilidad",
                 color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_pvalue_distribution(calib_scores: dict, results: dict, y_test,
                             output_path: Path = PVALUE_FIGURE_PATH):
    """Distribución de p-valores de las transacciones legítimas del período posterior.

    Bajo intercambiabilidad los p-valores de lo legítimo son uniformes en [0, 1]; ese es el
    contenido estadístico de la garantía. Cuánto se aparta el histograma de la uniforme es
    una medida directa de cuánta deriva hay, sin necesidad de etiquetas.
    """
    y_arr = np.asarray(y_test)
    fig, axes = plt.subplots(1, len(calib_scores), figsize=(4.2 * len(calib_scores), 4))
    if len(calib_scores) == 1:
        axes = [axes]

    for ax, (name, calib) in zip(axes, calib_scores.items()):
        p_values = conformal_p_values(calib, results[name]["scores"])
        ax.hist(p_values[y_arr == 0], bins=40, range=(0, 1), density=True,
                color="#2a78d6", alpha=0.75)
        ax.axhline(1.0, linestyle="--", linewidth=1.5, color=INK_MUTED)
        ax.set_xlabel("p-valor conforme", color=INK_SECONDARY)
        ax.set_title(MODEL_LABELS.get(name, name), color=INK_PRIMARY, fontsize=11)
        style_axes(ax)

    axes[0].set_ylabel("Densidad (uniforme = 1)", color=INK_SECONDARY)
    fig.suptitle("Transacciones legítimas: bajo intercambiabilidad estos p-valores serían uniformes",
                 color=INK_PRIMARY, fontsize=12)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("Cargando el split temporal del Módulo 5...")
    data = get_temporal_data()
    X_train, X_test, y_test = data["X_train"], data["X_test"], data["y_test"]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_calib_scaled = scaler.transform(data["X_calib"])

    print(f"Train: {X_train.shape} | Calib: {data['X_calib'].shape} | "
          f"Test: {X_test.shape}, fraude={int(y_test.sum())} ({y_test.mean():.4%})")

    print("\nEntrenando los 15 detectores...")
    results = run_detectors(X_train_scaled, X_test_scaled)
    calib_scores = {name: out["score_fn"](X_calib_scaled) for name, out in results.items()}

    comparaciones = {}
    for name in DETECTORES:
        comparaciones[name] = compare_scenarios(calib_scores[name], results[name]["scores"], y_test)
        print(f"\n=== {MODEL_LABELS.get(name, name)} ===")
        print(comparaciones[name].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== Resumen: razón entre la tasa observada y la garantizada (alpha = 0.01) ===")
    resumen = pd.DataFrame([
        {
            "detector": MODEL_LABELS.get(name, name),
            "intercambiable": tabla.loc[tabla.alpha == 0.01, "razon_intercambiable"].iloc[0],
            "con_deriva": tabla.loc[tabla.alpha == 0.01, "razon_con_deriva"].iloc[0],
        }
        for name, tabla in comparaciones.items()
    ])
    print(resumen.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n(1.0 = la garantía se cumple exactamente; el ruido muestral la mueve un ~10-20%)")

    plot_coverage(comparaciones)
    plot_pvalue_distribution({n: calib_scores[n] for n in DETECTORES}, results, y_test)
    plt.close("all")
    print(f"\nCobertura:   {COVERAGE_FIGURE_PATH}")
    print(f"P-valores:   {PVALUE_FIGURE_PATH}")

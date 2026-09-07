"""Módulo 5 — del ranking a la operación: umbral, dinero y envejecimiento.

Responde las tres preguntas que quedan abiertas después del benchmark del Módulo 3, todas
sobre el mismo split temporal y con prevalencia real:

1. **¿Cuánto del PR-AUC del Módulo 3 era optimismo del split aleatorio?** Se reentrena todo
   cortando por tiempo, que es como funciona un sistema real.
2. **¿El que mejor rankea es el que más plata salva?** El ranking por PR-AUC y el ranking
   por ahorro neto no tienen por qué coincidir cuando el 10% de los fraudes concentra la
   mitad del monto.
3. **¿El umbral elegido el día 1 sigue sirviendo el día 30?** Se compara la tasa de falsos
   positivos prometida por el cuantil de entrenamiento contra la observada después.

Ejecución:

    python -m src.operations.run_operations
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.operations.costs import (
    amount_concentration,
    break_even_review_cost,
    net_savings,
    savings_curve,
    value_weighted_recall,
)
from src.operations.temporal import evaluate_by_period, get_temporal_data, volume_drift
from src.operations.thresholds import (
    capacity_threshold,
    cost_optimal_threshold,
    empirical_alarm_rate,
    quantile_threshold,
    threshold_report,
)
from src.unsupervised.benchmark import MODEL_FAMILY, MODEL_LABELS, run_detectors
from src.unsupervised.metrics_store import save_benchmark_metrics
from src.unsupervised.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
SAVINGS_FIGURE_PATH = FIGURES_DIR / "operations_savings.png"
DRIFT_FIGURE_PATH = FIGURES_DIR / "operations_drift.png"
CALIBRATION_FIGURE_PATH = FIGURES_DIR / "operations_calibration.png"

# Costo de revisar una alerta, en las mismas unidades que `amount`. Es un supuesto de
# negocio, no un dato del dataset, y su magnitud relativa al costo de equilibrio decide si
# el problema es de modelado o de capacidad: por debajo del equilibrio (~1.000 en este
# período) conviene revisar absolutamente todo y el umbral deja de importar. Se fija por
# encima para que el óptimo sea interior y la curva de ahorro diga algo.
REVIEW_COST = 5_000.0

# Se asume que una alerta a tiempo bloquea la transacción y evita la pérdida completa.
RECOVERY_RATE = 1.0

# Capacidad de revisión del equipo, en alertas por día.
ALERTS_PER_DAY = 100

TOP_N_PLOT = 4


def evaluate_operationally(results: dict, y_test, amounts, calib_scores: dict,
                           n_days: int) -> pd.DataFrame:
    """Métricas por conteo y por monto, lado a lado, para cada detector."""
    y_arr = np.asarray(y_test)

    rows = []
    for name, output in results.items():
        scores = output["scores"]
        threshold = capacity_threshold(scores, ALERTS_PER_DAY, n_days)
        flagged = scores >= threshold

        optimum = cost_optimal_threshold(y_arr, scores, amounts, REVIEW_COST, RECOVERY_RATE)
        alpha_prometido = 0.01
        umbral_cuantil = quantile_threshold(calib_scores[name], alpha_prometido)

        rows.append({
            "modelo": MODEL_LABELS.get(name, name),
            "clave": name,
            "pr_auc": average_precision_score(y_arr, scores),
            # ROC-AUC no depende de la prevalencia, así que es lo único directamente
            # comparable contra los números del Módulo 3, medidos al 14,1% de fraude.
            "roc_auc": roc_auc_score(y_arr, scores),
            "recall_capacidad": float(((y_arr == 1) & flagged).sum() / max(1, y_arr.sum())),
            "recall_monto_capacidad": value_weighted_recall(y_arr, flagged, amounts),
            "ahorro_capacidad": net_savings(y_arr, flagged, amounts, REVIEW_COST, RECOVERY_RATE),
            "ahorro_optimo": optimum["net_savings"],
            "alertas_optimo": optimum["n_alerts"],
            "alpha_observado": empirical_alarm_rate(scores[y_arr == 0], umbral_cuantil),
        })
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def plot_savings(results: dict, y_test, amounts, top_names: list[str],
                 output_path: Path = SAVINGS_FIGURE_PATH):
    """Curva de ahorro neto contra cantidad de alertas revisadas, con el óptimo marcado."""
    fig, (ax_money, ax_recall) = plt.subplots(1, 2, figsize=(13, 5))

    for name in top_names:
        curve = savings_curve(y_test, results[name]["scores"], amounts, REVIEW_COST, RECOVERY_RATE)
        label = MODEL_LABELS.get(name, name)
        ax_money.plot(curve["revisadas"], curve["ahorro_neto"] / 1e6, linewidth=2, label=label)
        ax_recall.plot(curve["revisadas"], curve["recall_en_monto"], linewidth=2, label=label)

        best = curve.loc[curve["ahorro_neto"].idxmax()]
        ax_money.scatter([best["revisadas"]], [best["ahorro_neto"] / 1e6], s=30, zorder=5)

    ax_money.axhline(0, linestyle="--", linewidth=1, color=INK_MUTED)
    ax_money.set_xscale("log")
    ax_money.set_xlabel("Alertas revisadas (escala log)", color=INK_SECONDARY)
    ax_money.set_ylabel("Ahorro neto (millones)", color=INK_SECONDARY)
    ax_money.set_title("Ahorro neto por presupuesto de revisión", color=INK_PRIMARY, fontsize=12)
    style_axes(ax_money)
    ax_money.legend(frameon=False, fontsize=9)

    ax_recall.set_xscale("log")
    ax_recall.set_xlabel("Alertas revisadas (escala log)", color=INK_SECONDARY)
    ax_recall.set_ylabel("Fracción del monto defraudado recuperada", color=INK_SECONDARY)
    ax_recall.set_title("Recall en pesos, no en conteo", color=INK_PRIMARY, fontsize=12)
    style_axes(ax_recall)
    ax_recall.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Costo de revisión = {REVIEW_COST:,.0f} por alerta", color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_drift(degradation: dict, volumes: pd.DataFrame, output_path: Path = DRIFT_FIGURE_PATH):
    """PR-AUC por día del período de prueba, contra el volumen de transacciones de ese día."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name, table in degradation.items():
        ax.plot(table["period"], table["metric"], linewidth=2, marker="o", markersize=3,
                label=MODEL_LABELS.get(name, name))

    ax.set_xlabel("Días después del corte temporal", color=INK_SECONDARY)
    ax.set_ylabel("ROC-AUC del día", color=INK_SECONDARY)
    ax.set_title("¿El detector envejece? ROC-AUC día a día sobre datos futuros",
                 color=INK_PRIMARY, fontsize=13)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax_volume = ax.twinx()
    ax_volume.bar(volumes["period"], volumes["n"], alpha=0.15, color=INK_MUTED, zorder=0)
    ax_volume.set_ylabel("Transacciones por día (barras)", color=INK_MUTED)
    ax_volume.tick_params(colors=INK_MUTED)
    ax_volume.spines["top"].set_visible(False)

    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_calibration(reports: dict, reports_train: dict | None = None,
                     output_path: Path = CALIBRATION_FIGURE_PATH):
    """FPR prometida por el cuantil contra la observada, calibrando de dos formas distintas.

    Línea llena: el umbral sale de normales retenidas que el detector no usó para ajustarse.
    Línea punteada: sale del propio conjunto de entrenamiento.

    Que ambas líneas se superpongan es el resultado, no un descuido: descarta el sobreajuste
    como causa del desajuste y lo atribuye al desplazamiento temporal de la escala del score.
    Un detector cuyo ranking se mantiene estable en el tiempo puede tener, aun así, una
    escala que se corre — y entonces ningún umbral fijo aprendido del pasado sirve.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for name, report in reports.items():
        linea, = ax.plot(report["alpha"], report["alpha_observado"], linewidth=2, marker="o",
                         label=MODEL_LABELS.get(name, name))
        if reports_train and name in reports_train:
            ax.plot(reports_train[name]["alpha"], reports_train[name]["alpha_observado"],
                    linewidth=1.5, linestyle=":", marker="x", markersize=5,
                    color=linea.get_color(), alpha=0.8)

    # Los límites salen de los datos: fijarlos a mano recortaría justamente el caso
    # interesante, que es el umbral mal calibrado disparándose fuera del rango esperado.
    todos = list(reports.values()) + list((reports_train or {}).values())
    observados = np.concatenate([r["alpha_observado"].to_numpy() for r in todos])
    prometidos = np.concatenate([r["alpha"].to_numpy() for r in todos])
    finitos = np.concatenate([observados[observados > 0], prometidos])
    lims = [float(finitos.min() * 0.5), float(finitos.max() * 2)]

    ax.plot(lims, lims, linestyle="--", linewidth=1, color=INK_MUTED, label="Calibración perfecta")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("FPR prometida por el cuantil (α)", color=INK_SECONDARY)
    ax.set_ylabel("FPR observada en el período posterior", color=INK_SECONDARY)
    ax.set_title("Umbral calibrado con validación retenida (línea) vs. con entrenamiento (puntos)",
                 color=INK_PRIMARY, fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("Split temporal: entrenamiento con el pasado, evaluación con el futuro...")
    data = get_temporal_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_test, amounts, steps = data["y_test"], data["amounts_test"], data["steps_test"]

    print(f"Corte en step={data['cutoff_step']} | Train (normales): {X_train.shape} | "
          f"Test: {X_test.shape}, fraude={int(y_test.sum())} ({y_test.mean():.4%})")

    equilibrio = break_even_review_cost(y_test, amounts, RECOVERY_RATE)
    print(f"Costo de revisión de equilibrio: {equilibrio:,.0f} por alerta "
          f"(el usado es {REVIEW_COST:,.0f}, {REVIEW_COST / equilibrio:.1f}x)")

    concentracion = amount_concentration(y_test, amounts)
    print(f"Monto: mediana del fraude={concentracion['median_fraud_amount']:,.0f} | "
          f"el 10% más caro concentra {concentracion['share_of_amount']:.1%} del monto defraudado")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_calib_scaled = scaler.transform(data["X_calib"])

    print("\nEntrenando los 13 detectores sobre el split temporal...")
    results = run_detectors(X_train_scaled, X_test_scaled)

    # Los scores de entrenamiento son la base de los umbrales sin etiquetas. Se calculan
    # con los detectores ya ajustados (score_fn), no reajustando todo una segunda vez.
    print("\nPuntuando el conjunto de entrenamiento (base de los umbrales sin etiquetas)...")
    train_results = {name: output["score_fn"](X_train_scaled) for name, output in results.items()}
    calib_results = {name: output["score_fn"](X_calib_scaled) for name, output in results.items()}

    n_days = int(np.ceil((steps.max() - data["cutoff_step"]) / 24))
    print(f"El período de prueba cubre {n_days} días posteriores al corte.")

    table = evaluate_operationally(results, y_test, amounts, calib_results, n_days)
    print("\n=== Operación sobre el split temporal (prevalencia real) ===")
    print(table.drop(columns=["clave"]).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    por_ranking = table.iloc[0]
    por_dinero = table.loc[table["ahorro_optimo"].idxmax()]
    print(f"\nMejor por PR-AUC:       {por_ranking['modelo']} ({por_ranking['pr_auc']:.4f})")
    print(f"Mejor por ahorro neto:  {por_dinero['modelo']} ({por_dinero['ahorro_optimo']:,.0f})")
    if por_ranking["clave"] != por_dinero["clave"]:
        print("  -> el que mejor rankea NO es el que más dinero salva")

    top_names = table["clave"].head(TOP_N_PLOT).tolist()

    print("\n=== Calibración del umbral: cuantil sobre entrenamiento vs. validación retenida ===")
    reports, reports_train = {}, {}
    for name in top_names:
        reports_train[name] = threshold_report(train_results[name], results[name]["scores"], y_test)
        reports[name] = threshold_report(calib_results[name], results[name]["scores"], y_test)
        print(f"\n{MODEL_LABELS.get(name, name)}")
        comparacion = pd.DataFrame({
            "alpha": reports[name]["alpha"],
            "obs_desde_train": reports_train[name]["alpha_observado"],
            "obs_desde_calib": reports[name]["alpha_observado"],
            "alertas": reports[name]["alertas"],
            "recall": reports[name]["recall"],
            "precision": reports[name]["precision"],
        })
        print(comparacion.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== Degradación temporal (ROC-AUC por día; PR-AUC no compara entre días) ===")
    degradation = {}
    for name in top_names:
        # ROC-AUC y no PR-AUC: la prevalencia diaria varía y el PR-AUC la sigue, así que
        # una curva de PR-AUC por día mide el cambio de prevalencia, no la degradación.
        degradation[name] = evaluate_by_period(
            y_test, results[name]["scores"], steps, data["cutoff_step"], roc_auc_score
        )
    resumen = pd.DataFrame({
        MODEL_LABELS.get(n, n): [d["metric"].iloc[0], d["metric"].iloc[-1], d["metric"].mean()]
        for n, d in degradation.items() if len(d) >= 2
    }, index=["primer día", "último día", "promedio"])
    print(resumen.to_string(float_format=lambda v: f"{v:.4f}"))

    volumes = volume_drift(steps, data["cutoff_step"])
    print(f"\nVolumen diario en prueba: primer día={volumes['n'].iloc[0]}, "
          f"último día={volumes['n'].iloc[-1]}, razón={volumes['n'].iloc[0] / max(1, volumes['n'].iloc[-1]):.1f}x")

    plot_savings(results, y_test, amounts, top_names)
    plot_drift(degradation, volumes)
    plot_calibration(reports, reports_train)
    plt.close("all")
    print(f"\nAhorro:       {SAVINGS_FIGURE_PATH}")
    print(f"Drift:        {DRIFT_FIGURE_PATH}")
    print(f"Calibración:  {CALIBRATION_FIGURE_PATH}")

    persistible = {
        row["clave"]: {
            "pr_auc": row["pr_auc"],
            "roc_auc": row["roc_auc"],
            "precision_recall_at_k": {},
            "family": MODEL_FAMILY.get(row["clave"], "Otro"),
            "fit_seconds": results[row["clave"]]["fit_seconds"],
            "score_seconds": results[row["clave"]]["score_seconds"],
        }
        for _, row in table.iterrows()
    }
    save_benchmark_metrics(persistible, model_labels=MODEL_LABELS)
    print("\nMétricas del split temporal guardadas en benchmark_metrics")

"""Módulo 4 — entrenamiento y evaluación de los tres modelos profundos.

Los tres nacen de una limitación concreta encontrada en el benchmark del Módulo 3:

| Modelo | Limitación que ataca | Unidad evaluada |
|---|---|---|
| VAE | El mejor detector fue un modelo de densidad (GMM); el autoencoder solo reconstruye | Transacción |
| Deep SVDD | El autoencoder apenas superó su ablación lineal — quizá el problema es el objetivo | Transacción |
| Autoencoder GRU | Los trece detectores tratan cada fila como independiente | **Cuenta destino** |

Los dos primeros comparten split, escalado y métricas con el Módulo 3, así que sus números
entran en la misma tabla. El tercero cambia la unidad de análisis y **se reporta aparte**:
comparar un PR-AUC por cuenta con uno por transacción sería comparar dos problemas.

Ejecución:

    python -m src.deep.train_deep
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.deep.one_class import build_deep_detectors
from src.deep.sequences import (
    FEATURE_NAMES,
    aggregate_step_errors,
    fraud_window_coverage,
    get_sequence_data,
    sequence_step_errors,
    train_sequence_autoencoder,
)
from src.data.loader import load_raw_data
from src.unsupervised.autoencoder import reconstruction_error, train_autoencoder
from src.unsupervised.loader import get_unsupervised_data
from src.unsupervised.metrics_store import save_benchmark_metrics, save_sequence_metrics
from src.unsupervised.models import anomaly_score
from src.unsupervised.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, style_axes
from src.unsupervised.train_unsupervised import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
DEEP_PR_FIGURE_PATH = FIGURES_DIR / "deep_pr_curve.png"
SEQUENCE_FIGURE_PATH = FIGURES_DIR / "sequence_detector.png"

MODEL_LABELS = {
    "vae": "VAE (ELBO)",
    "deep_svdd": "Deep SVDD",
    "autoencoder": "Autoencoder (ReLU, Módulo 2)",
    "sequence_gru": "Autoencoder GRU por cuenta",
}
MODEL_COLORS = {
    "vae": "#9c27b0",
    "deep_svdd": "#00897b",
    "autoencoder": "#4caf50",
    "sequence_gru": "#c2185b",
}
MODEL_FAMILY = {
    "vae": "Densidad profunda",
    "deep_svdd": "Una clase profunda",
    "autoencoder": "Reconstrucción",
    "sequence_gru": "Secuencial por cuenta",
}


def run_transaction_level() -> tuple[dict, pd.Series]:
    """Entrena VAE y Deep SVDD en el mismo split del Módulo 3, con el autoencoder de referencia."""
    print("Cargando datos a nivel de transacción (entrenamiento 100% normal)...")
    X_train, X_test, y_test = get_unsupervised_data()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}, fraude={int(y_test.sum())}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name, detector in build_deep_detectors().items():
        print(f"\nEntrenando {MODEL_LABELS[name]}...", flush=True)
        start = time.perf_counter()
        detector.fit(X_train_scaled)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scores = anomaly_score(detector, X_test_scaled)
        score_seconds = time.perf_counter() - start

        metrics = evaluate(y_test, scores)
        results[name] = {
            "scores": scores,
            "roc_auc": roc_auc_score(y_test, scores),
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "family": MODEL_FAMILY[name],
            **metrics,
        }
        print(f"  PR-AUC={metrics['pr_auc']:.4f} | fit {fit_seconds:.1f}s | score {score_seconds:.1f}s")

    # Autoencoder del Módulo 2 como referencia en la misma corrida: sin él, la comparación
    # dependería de números medidos en otra ejecución.
    print(f"\nEntrenando {MODEL_LABELS['autoencoder']} (referencia)...", flush=True)
    start = time.perf_counter()
    model = train_autoencoder(X_train_scaled.astype("float32"), activation="relu")
    fit_seconds = time.perf_counter() - start
    scores = reconstruction_error(model, X_test_scaled.astype("float32"))
    metrics = evaluate(y_test, scores)
    results["autoencoder"] = {
        "scores": scores,
        "roc_auc": roc_auc_score(y_test, scores),
        "fit_seconds": fit_seconds,
        "score_seconds": 0.0,
        "family": MODEL_FAMILY["autoencoder"],
        **metrics,
    }
    print(f"  PR-AUC={metrics['pr_auc']:.4f}")

    return results, y_test


def run_account_level() -> dict:
    """Entrena el autoencoder GRU sobre las historias de las cuentas destino."""
    print("\nConstruyendo secuencias por cuenta destino...", flush=True)
    (seq_train, mask_train), (seq_test, mask_test), y_test = get_sequence_data()
    print(f"Train (cuentas limpias): {seq_train.shape} | Test: {seq_test.shape}, "
          f"cuentas con fraude={int(y_test.sum())} ({y_test.mean():.2%})")
    print(f"Features por paso: {FEATURE_NAMES}")

    # Antes de interpretar el resultado: si el truncado a las 20 transacciones más recientes
    # dejara la mayoría del fraude fuera de la ventana, un PR-AUC bajo no diría nada sobre el
    # modelo. Se descarta esa explicación con un número, no con una suposición.
    coverage = fraud_window_coverage(load_raw_data())
    print(f"Cobertura del truncado: {coverage['inside_window']}/{coverage['fraud_transactions']} "
          f"de fraude dentro de la ventana ({coverage['coverage']:.1%}), "
          f"posición mediana desde el final: {coverage['median_position_from_end']:.0f}")

    print(f"\nEntrenando {MODEL_LABELS['sequence_gru']}...", flush=True)
    start = time.perf_counter()
    model = train_sequence_autoencoder(seq_train, mask_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    step_errors = sequence_step_errors(model, seq_test, mask_test)
    scores = aggregate_step_errors(step_errors, mask_test, how="mean")
    score_seconds = time.perf_counter() - start

    y_series = pd.Series(y_test)
    metrics = evaluate(y_series, scores)
    result = {
        "scores": scores,
        "roc_auc": roc_auc_score(y_series, scores),
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "family": MODEL_FAMILY["sequence_gru"],
        **metrics,
    }
    print(f"  PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={result['roc_auc']:.4f} "
          f"| fit {fit_seconds:.1f}s | score {score_seconds:.1f}s")

    # La otra explicación posible de un resultado pobre es la agregación: una única
    # transacción anómala se diluye al promediarla sobre veinte pasos. Se comparan las
    # alternativas sobre el mismo modelo entrenado, así que la única diferencia es el score.
    print()
    print("  Agregación del error por paso (mismo modelo entrenado):")
    for how in ["mean", "max", "top3", "last"]:
        alternativa = aggregate_step_errors(step_errors, mask_test, how=how)
        print(f"    {how:5s} PR-AUC={average_precision_score(y_series, alternativa):.4f} "
              f"| ROC-AUC={roc_auc_score(y_series, alternativa):.4f}")

    return {"result": result, "y_test": y_series}


def summary_table(results: dict, labels: dict = MODEL_LABELS) -> pd.DataFrame:
    """Tabla ordenada por PR-AUC con métricas y costo por modelo."""
    rows = []
    for name, res in results.items():
        p100, r100 = res["precision_recall_at_k"][100]
        rows.append({
            "modelo": labels.get(name, name),
            "familia": res["family"],
            "pr_auc": res["pr_auc"],
            "roc_auc": res["roc_auc"],
            "precision@100": p100,
            "recall@100": r100,
            "fit_s": res["fit_seconds"],
            "score_s": res["score_seconds"],
        })
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def plot_deep_pr_curves(results: dict, y_test, output_path: Path = DEEP_PR_FIGURE_PATH):
    """Curvas Precision-Recall de los modelos profundos a nivel de transacción."""
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for name, res in sorted(results.items(), key=lambda item: -item[1]["pr_auc"]):
        precision, recall, _ = precision_recall_curve(y_test, res["scores"])
        ax.plot(recall, precision, color=MODEL_COLORS.get(name, INK_SECONDARY), linewidth=2,
                label=f"{MODEL_LABELS.get(name, name)} (PR-AUC={res['pr_auc']:.3f})")

    baseline = float(np.mean(y_test))
    ax.axhline(baseline, linestyle="--", linewidth=1, color=INK_MUTED, label=f"Azar ({baseline:.4f})")
    ax.set_xlabel("Recall", color=INK_SECONDARY)
    ax.set_ylabel("Precision", color=INK_SECONDARY)
    ax.set_title("Modelos profundos de una clase — nivel transacción", color=INK_PRIMARY, fontsize=13)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_sequence_detector(result: dict, y_test, output_path: Path = SEQUENCE_FIGURE_PATH):
    """Distribución del score por cuenta y su curva Precision-Recall, lado a lado."""
    scores = result["scores"]
    fig, (ax_dist, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    x_min, x_max = np.percentile(scores, [0.5, 99.5])
    bins = np.geomspace(max(x_min, 1e-6), x_max, 60) if x_min > 0 else np.linspace(x_min, x_max, 60)
    if x_min > 0:
        ax_dist.set_xscale("log")

    y_values = np.asarray(y_test)
    ax_dist.hist(scores[y_values == 0], bins=bins, alpha=0.6, density=True,
                 label="Cuenta limpia", color="#2a78d6")
    ax_dist.hist(scores[y_values == 1], bins=bins, alpha=0.6, density=True,
                 label="Cuenta con fraude", color="#eb6834")
    ax_dist.set_xlim(x_min, x_max)
    ax_dist.set_xlabel("Error de reconstrucción de la historia", color=INK_SECONDARY)
    ax_dist.set_ylabel("Densidad", color=INK_SECONDARY)
    ax_dist.set_title("Score por cuenta destino", color=INK_PRIMARY, fontsize=12)
    style_axes(ax_dist)
    ax_dist.legend(frameon=False)

    precision, recall, _ = precision_recall_curve(y_test, scores)
    ax_pr.plot(recall, precision, color=MODEL_COLORS["sequence_gru"], linewidth=2,
               label=f"Autoencoder GRU (PR-AUC={result['pr_auc']:.3f})")
    baseline = float(np.mean(y_values))
    ax_pr.axhline(baseline, linestyle="--", linewidth=1, color=INK_MUTED, label=f"Azar ({baseline:.4f})")
    ax_pr.set_xlabel("Recall", color=INK_SECONDARY)
    ax_pr.set_ylabel("Precision", color=INK_SECONDARY)
    ax_pr.set_title("Precision-Recall por cuenta", color=INK_PRIMARY, fontsize=12)
    style_axes(ax_pr)
    ax_pr.legend(frameon=False, loc="upper right")

    fig.suptitle("Detector secuencial — la unidad evaluada es la cuenta, no la transacción",
                 color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


if __name__ == "__main__":
    transaction_results, y_transaction = run_transaction_level()

    print("\n=== Nivel transacción (comparable con el Módulo 3) ===")
    print(summary_table(transaction_results).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    account = run_account_level()

    print("\n=== Nivel cuenta (NO comparable con la tabla anterior) ===")
    print(summary_table({"sequence_gru": account["result"]}).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    plot_deep_pr_curves(transaction_results, y_transaction)
    plot_sequence_detector(account["result"], account["y_test"])
    plt.close("all")
    print(f"\nCurvas PR (transacción): {DEEP_PR_FIGURE_PATH}")
    print(f"Detector secuencial:     {SEQUENCE_FIGURE_PATH}")

    save_benchmark_metrics(transaction_results, model_labels=MODEL_LABELS)
    save_sequence_metrics({"sequence_gru": account["result"]}, model_labels=MODEL_LABELS)
    print("\nMétricas guardadas en data/processed/metrics.duckdb")
    print("  nivel transacción -> benchmark_metrics | nivel cuenta -> sequence_metrics")

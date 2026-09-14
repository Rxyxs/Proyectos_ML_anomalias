"""Módulo 9 — ¿recalibrar la ventana conforme devuelve la tasa de alertas a lo prometido?

Tres estrategias sobre el mismo detector, el mismo escalador y la misma calibración inicial
que el paquete del Módulo 8. Lo único que cambia es qué pasa con la calibración al terminar
cada día:

- `estatico`: nada. Es el paquete del Módulo 8, y su tasa de alertas va de 1,57% a 11,10%.
- `ventana`: se agregan todos los scores del día a la ventana, sin etiquetas.
- `ventana_oraculo`: se agregan solo las legítimas, con etiquetas al instante. No es
  desplegable; es la ablación que aísla el costo de contaminar la ventana con fraude.
- `ventana_sin_alertas`: se agregan solo los que no se alertaron, para no meter fraude.

Los scores del período de prueba se calculan una sola vez: el detector no cambia, solo la
calibración contra la que se comparan. Eso vuelve la simulación barata y garantiza que las
diferencias entre estrategias no vengan del modelo.

Además se evalúa si el monitor sin etiquetas (`tail_ratio`) sigue a la tasa real de falsos
positivos, que sí necesita etiquetas y en producción llegaría semanas tarde.

Ejecución:

    python -m src.serving.run_recalibration
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

from src.operations.temporal import get_temporal_data, period_index
from src.serving.recalibration import RollingCalibrator, tail_ratio, uniformity_gap
from src.unsupervised.families import GMMDensity
from src.unsupervised.models import anomaly_score
from src.unsupervised.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECALIBRATION_FIGURE_PATH = PROJECT_ROOT / "data" / "processed" / "figures" / "recalibration.png"

ALPHA = 0.01
WINDOW_SIZE = 30_000

# Mismo filtro de volumen que el análisis de deriva del Módulo 5: sobre días con pocas
# transacciones la tasa diaria es ruido. Esos días igual se puntúan y alimentan la ventana;
# solo se excluyen de los promedios.
MIN_ROWS_PER_DAY = 1_000

STRATEGIES = {
    "estatico": None,
    "ventana": {"exclude_alerts": False},
    # Ablación, no estrategia desplegable: alimenta la ventana solo con legítimas usando las
    # etiquetas al instante. Separa el efecto buscado —absorber la deriva— del costo de meter
    # fraude en la calibración, que en producción no se puede evitar porque las etiquetas
    # llegan semanas tarde.
    "ventana_oraculo": {"exclude_alerts": False, "oracle": True},
    "ventana_sin_alertas": {"exclude_alerts": True},
}


def simulate(calibration_scores, test_scores, y_test, periods, alpha: float = ALPHA,
             window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Recorre los días en orden y registra qué alerta cada estrategia.

    Cada día se puntúa contra la calibración vigente y recién después se actualiza la ventana.
    Las etiquetas se usan solo para evaluar (FPR, fraude capturado), nunca para decidir.
    """
    scores = np.asarray(test_scores, dtype=float)
    y = np.asarray(y_test).astype(int)
    periods = np.asarray(periods)

    filas = []
    for estrategia, config in STRATEGIES.items():
        calibrador = RollingCalibrator(
            calibration_scores, window_size=window_size, alpha=alpha,
            exclude_alerts=bool(config and config.get("exclude_alerts")),
        )
        for dia in np.unique(periods):
            seleccion = periods == dia
            s, yd = scores[seleccion], y[seleccion]

            p = calibrador.p_values(s)
            alerta = p <= alpha
            legitimas, fraudes = yd == 0, yd == 1

            filas.append({
                "estrategia": estrategia,
                "dia": int(dia),
                "n": int(seleccion.sum()),
                "umbral": calibrador.threshold,
                "alertas": int(alerta.sum()),
                "tasa_alerta": float(alerta.mean()),
                "fpr": float(alerta[legitimas].mean()) if legitimas.any() else np.nan,
                "fraudes": int(fraudes.sum()),
                "capturados": int((alerta & fraudes).sum()),
                "razon_cola": tail_ratio(p, alpha),
                "brecha_uniforme": uniformity_gap(p),
            })

            if config is not None:
                calibrador.update(s[legitimas] if config.get("oracle") else s)

    return pd.DataFrame(filas)


def summarize(tabla: pd.DataFrame, alpha: float = ALPHA,
              min_rows: int = MIN_ROWS_PER_DAY) -> pd.DataFrame:
    """Una fila por estrategia: cuánto se aparta de lo prometido y cuánto fraude atrapa.

    Las tasas se promedian sobre días con volumen suficiente; el fraude capturado y las
    alertas totales cuentan todos los días, porque un fraude en un día chico sigue siendo plata.
    """
    filas = []
    for estrategia, grupo in tabla.groupby("estrategia", sort=False):
        evaluables = grupo[grupo["n"] >= min_rows]
        filas.append({
            "estrategia": estrategia,
            "tasa_media": evaluables["tasa_alerta"].mean(),
            "tasa_max": evaluables["tasa_alerta"].max(),
            "desvio_abs_medio": (evaluables["tasa_alerta"] - alpha).abs().mean(),
            "fpr_media": evaluables["fpr"].mean(),
            "alertas": int(grupo["alertas"].sum()),
            "fraude_capturado": grupo["capturados"].sum() / max(1, int(grupo["fraudes"].sum())),
        })
    return pd.DataFrame(filas)


def monitor_agreement(tabla: pd.DataFrame, estrategia: str = "estatico", alpha: float = ALPHA,
                      min_rows: int = MIN_ROWS_PER_DAY) -> dict:
    """¿El monitor sin etiquetas ordena los días igual que la FPR real?

    Se compara `razon_cola` —calculable el mismo día, sin etiquetas— contra la FPR dividida por
    alpha, que es la misma cantidad restringida a lo legítimo y por lo tanto necesita etiquetas.
    """
    dias = tabla[(tabla["estrategia"] == estrategia) & (tabla["n"] >= min_rows)]
    rho, _ = spearmanr(dias["razon_cola"], dias["fpr"] / alpha)
    return {
        "dias": int(len(dias)),
        "spearman": float(rho),
        "razon_cola_media": float(dias["razon_cola"].mean()),
        "razon_fpr_media": float((dias["fpr"] / alpha).mean()),
    }


def plot_recalibration(tabla: pd.DataFrame, alpha: float = ALPHA, min_rows: int = MIN_ROWS_PER_DAY,
                       output_path: Path = RECALIBRATION_FIGURE_PATH):
    """Tasa de alertas diaria por estrategia, y el monitor sin etiquetas contra la FPR real."""
    evaluables = tabla[tabla["n"] >= min_rows]
    fig, (ax_tasa, ax_monitor) = plt.subplots(1, 2, figsize=(14, 5.5))

    for estrategia, grupo in evaluables.groupby("estrategia", sort=False):
        ax_tasa.plot(grupo["dia"], grupo["tasa_alerta"] * 100, linewidth=2, marker="o",
                     markersize=4, label=estrategia)
    ax_tasa.axhline(alpha * 100, linestyle="--", linewidth=1.2, color=INK_MUTED,
                    label=f"prometido ({alpha:.0%})")
    ax_tasa.set_xlabel("Días después del corte temporal", color=INK_SECONDARY)
    ax_tasa.set_ylabel("Transacciones alertadas (%)", color=INK_SECONDARY)
    ax_tasa.set_title("Tasa de alertas diaria según qué se hace con la calibración",
                      color=INK_PRIMARY, fontsize=12)
    style_axes(ax_tasa)
    ax_tasa.legend(frameon=False, fontsize=9)

    estatico = evaluables[evaluables["estrategia"] == "estatico"]
    ax_monitor.plot(estatico["dia"], estatico["razon_cola"], linewidth=2, marker="o",
                    markersize=4, label="monitor sin etiquetas (razón de cola)")
    ax_monitor.plot(estatico["dia"], estatico["fpr"] / alpha, linewidth=2, marker="s",
                    markersize=4, label="FPR real / α (necesita etiquetas)")
    ax_monitor.axhline(1.0, linestyle="--", linewidth=1.2, color=INK_MUTED)
    ax_monitor.set_xlabel("Días después del corte temporal", color=INK_SECONDARY)
    ax_monitor.set_ylabel("Múltiplo de lo prometido", color=INK_SECONDARY)
    ax_monitor.set_title("¿El monitor sin etiquetas sigue a la FPR real? (paquete estático)",
                         color=INK_PRIMARY, fontsize=12)
    style_axes(ax_monitor)
    ax_monitor.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("Cargando el split temporal del Módulo 5...")
    data = get_temporal_data()
    X_train, X_calib, X_test = data["X_train"], data["X_calib"], data["X_test"]
    y_test, steps = data["y_test"], data["steps_test"]

    # Mismo detector y mismo escalado que el paquete del Módulo 8.
    scaler = RobustScaler().fit(X_train)
    detector = GMMDensity().fit(scaler.transform(X_train))

    calibracion = anomaly_score(detector, scaler.transform(X_calib))
    scores_prueba = anomaly_score(detector, scaler.transform(X_test))
    periodos = period_index(steps, data["cutoff_step"])

    print(f"Calibración inicial: {calibracion.size:,} scores | ventana: {WINDOW_SIZE:,} "
          f"| prueba: {scores_prueba.size:,} transacciones en {len(np.unique(periodos))} días")

    tabla = simulate(calibracion, scores_prueba, y_test, periodos)

    evaluables = tabla[tabla["n"] >= MIN_ROWS_PER_DAY]
    print("\n=== Tasa de alertas diaria (%) ===")
    print((evaluables.pivot(index="dia", columns="estrategia", values="tasa_alerta") * 100)
          [list(STRATEGIES)].to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== Umbral vigente por día ===")
    print(evaluables.pivot(index="dia", columns="estrategia", values="umbral")
          [list(STRATEGIES)].to_string(float_format=lambda v: f"{v:.2f}"))

    print(f"\n=== Resumen (prometido: {ALPHA:.0%}) ===")
    print(summarize(tabla).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    acuerdo = monitor_agreement(tabla)
    print("\n=== Monitor sin etiquetas contra FPR real (paquete estático) ===")
    print(f"días evaluados: {acuerdo['dias']} | Spearman: {acuerdo['spearman']:.3f}")
    print(f"razón de cola media: {acuerdo['razon_cola_media']:.2f} | "
          f"FPR/alpha media: {acuerdo['razon_fpr_media']:.2f}")

    plot_recalibration(tabla)
    plt.close("all")
    print(f"\nFigura: {RECALIBRATION_FIGURE_PATH}")

"""Panorama del dataset: las cuatro cosas que hay que ver antes de modelar nada.

Varias decisiones de todo el repositorio se justifican en propiedades de PaySim que conviene
mirar antes que cualquier métrica. Este módulo las calcula y las grafica juntas:

1. **El desbalance.** El fraude es el 0,13% del dataset. Es la razón por la que la accuracy
   no se usa en ningún módulo y por la que PR-AUC domina sobre ROC-AUC.
2. **El fraude vive en dos tipos de transacción.** `TRANSFER` y `CASH_OUT` concentran el
   100% de los casos; en `PAYMENT`, `DEBIT` y `CASH_IN` no hay uno solo. Es una regla que
   un modelo supervisado aprende en el primer corte, y explica por qué los supervisados
   parecen tan buenos en este dataset.
3. **El monto del fraude es mucho mayor.** Su mediana es ~6x la de una transacción legítima,
   y el 10% más caro concentra la mitad del monto defraudado. De ahí salen las métricas en
   pesos del Módulo 5.
4. **El volumen se derrumba a fin de mes.** Los primeros diez días tienen 3,19M transacciones
   y los últimos diez apenas 298k. Un split aleatorio esconde ese cambio por completo; el
   split temporal del Módulo 5 lo expone, y el filtro de volumen del análisis de deriva
   existe por él.

Ejecución:

    python -m src.data.overview
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_raw_data
from src.unsupervised.style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, style_axes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERVIEW_FIGURE_PATH = PROJECT_ROOT / "data" / "processed" / "figures" / "dataset_overview.png"

FRAUD_COLOR = "#eb6834"
LEGIT_COLOR = "#2a78d6"
HOURS_PER_DAY = 24


def class_balance(df: pd.DataFrame) -> dict:
    """Cuántas transacciones hay de cada clase y qué fracción representa el fraude."""
    n_fraud = int(df["isFraud"].sum())
    return {
        "n_total": int(len(df)),
        "n_fraud": n_fraud,
        "prevalence": n_fraud / len(df) if len(df) else 0.0,
    }


def fraud_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de fraude por tipo de transacción, ordenada de mayor a menor."""
    tabla = df.groupby("type")["isFraud"].agg(n="size", fraude="sum")
    tabla["tasa"] = tabla["fraude"] / tabla["n"]
    return tabla.sort_values("tasa", ascending=False).reset_index()


def amount_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos del monto por clase, en la escala original."""
    filas = []
    for etiqueta, nombre in ((0, "legítima"), (1, "fraude")):
        montos = df.loc[df["isFraud"] == etiqueta, "amount"]
        filas.append({
            "clase": nombre,
            "n": int(len(montos)),
            "mediana": float(montos.median()),
            "media": float(montos.mean()),
            "p99": float(montos.quantile(0.99)),
            "total": float(montos.sum()),
        })
    return pd.DataFrame(filas)


def volume_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Transacciones y fraudes por día simulado."""
    dia = (df["step"] - 1) // HOURS_PER_DAY
    tabla = df.assign(dia=dia).groupby("dia")["isFraud"].agg(n="size", fraude="sum")
    tabla["tasa"] = tabla["fraude"] / tabla["n"]
    return tabla.reset_index()


def plot_overview(df: pd.DataFrame, output_path: Path = OVERVIEW_FIGURE_PATH):
    """Las cuatro vistas en una sola figura de 2x2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    balance = class_balance(df)

    # --- 1. desbalance, en escala log porque si no la barra de fraude no se ve ---
    ax = axes[0, 0]
    ax.bar(["Legítimas", "Fraude"],
           [balance["n_total"] - balance["n_fraud"], balance["n_fraud"]],
           color=[LEGIT_COLOR, FRAUD_COLOR])
    ax.set_yscale("log")
    ax.set_ylabel("Transacciones (escala log)", color=INK_SECONDARY)
    ax.set_title(f"Desbalance: el fraude es el {balance['prevalence']:.2%} del dataset",
                 color=INK_PRIMARY, fontsize=12)
    for i, valor in enumerate([balance["n_total"] - balance["n_fraud"], balance["n_fraud"]]):
        ax.text(i, valor * 1.3, f"{valor:,}", ha="center", color=INK_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.grid(axis="x", visible=False)

    # --- 2. el fraude solo existe en dos tipos de transacción ---
    ax = axes[0, 1]
    tipos = fraud_by_type(df)
    colores = [FRAUD_COLOR if t > 0 else INK_MUTED for t in tipos["tasa"]]
    ax.barh(tipos["type"], tipos["tasa"] * 100, color=colores)
    for y, (tasa, n_fraude) in enumerate(zip(tipos["tasa"], tipos["fraude"])):
        etiqueta = f"{tasa:.2%} ({n_fraude:,})" if n_fraude else "sin fraude"
        ax.text(tasa * 100 + max(tipos["tasa"]) * 100 * 0.02, y, etiqueta,
                va="center", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, max(tipos["tasa"]) * 100 * 1.35)
    ax.set_xlabel("Tasa de fraude (%)", color=INK_SECONDARY)
    ax.set_title("El fraude solo ocurre en TRANSFER y CASH_OUT", color=INK_PRIMARY, fontsize=12)
    style_axes(ax)
    ax.grid(axis="y", visible=False)

    # --- 3. distribución del monto por clase ---
    ax = axes[1, 0]
    legitimos = df.loc[df["isFraud"] == 0, "amount"]
    fraudulentos = df.loc[df["isFraud"] == 1, "amount"]
    bins = np.geomspace(max(1.0, df["amount"].min()), df["amount"].max(), 60)
    ax.hist(legitimos, bins=bins, density=True, alpha=0.6, color=LEGIT_COLOR, label="Legítima")
    ax.hist(fraudulentos, bins=bins, density=True, alpha=0.6, color=FRAUD_COLOR, label="Fraude")
    ax.set_xscale("log")
    ax.set_xlabel("Monto (escala log)", color=INK_SECONDARY)
    ax.set_ylabel("Densidad", color=INK_SECONDARY)
    ax.set_title(f"El fraude mueve más plata: mediana {fraudulentos.median():,.0f} "
                 f"contra {legitimos.median():,.0f}", color=INK_PRIMARY, fontsize=12)
    style_axes(ax)
    ax.legend(frameon=False)

    # --- 4. el colapso de volumen que el split aleatorio esconde ---
    ax = axes[1, 1]
    dias = volume_by_day(df)
    ax.fill_between(dias["dia"], dias["n"], color=LEGIT_COLOR, alpha=0.35)
    ax.plot(dias["dia"], dias["n"], color=LEGIT_COLOR, linewidth=1.5)
    ax.set_xlabel("Día simulado", color=INK_SECONDARY)
    ax.set_ylabel("Transacciones por día", color=INK_SECONDARY)
    ax.set_title("El volumen se derrumba a fin de mes", color=INK_PRIMARY, fontsize=12)
    style_axes(ax)

    ax_fraude = ax.twinx()
    ax_fraude.plot(dias["dia"], dias["fraude"], color=FRAUD_COLOR, linewidth=1.5)
    ax_fraude.set_ylabel("Fraudes por día", color=FRAUD_COLOR)
    ax_fraude.tick_params(colors=FRAUD_COLOR)
    ax_fraude.spines["top"].set_visible(False)
    ax_fraude.set_ylim(0, dias["fraude"].max() * 1.2)

    fig.suptitle("PaySim: lo que hay que ver antes de modelar", color=INK_PRIMARY, fontsize=14)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("Cargando PaySim...")
    df = load_raw_data()

    balance = class_balance(df)
    print(f"\nTransacciones: {balance['n_total']:,} | fraude: {balance['n_fraud']:,} "
          f"({balance['prevalence']:.4%})")

    print("\nTasa de fraude por tipo de transacción:")
    print(fraud_by_type(df).to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\nMonto por clase:")
    print(amount_summary(df).to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    dias = volume_by_day(df)
    print(f"\nVolumen: día 0 = {dias['n'].iloc[0]:,} | día {int(dias['dia'].iloc[-1])} = "
          f"{dias['n'].iloc[-1]:,} | razón = {dias['n'].iloc[0] / dias['n'].iloc[-1]:,.0f}x")
    print(f"Fraudes por día: mediana = {dias['fraude'].median():.0f} "
          f"(estable mientras el volumen cae)")

    plot_overview(df)
    plt.close("all")
    print(f"\nFigura: {OVERVIEW_FIGURE_PATH}")

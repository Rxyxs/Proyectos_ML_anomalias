"""Módulo 8 — construir el paquete de scoring y medir qué cuesta usarlo.

Arma el artefacto desplegable a partir del split temporal del Módulo 5 y responde tres
preguntas que ningún módulo anterior toca, porque todas aparecen recién cuando el modelo
tiene que atender tráfico:

1. **¿Cuánto tarda en puntuar UNA transacción?** El Módulo 3 cronometró scoring por lotes de
   58.213 filas, que es la métrica de un backtest. En producción las transacciones llegan de
   a una, y ahí el costo fijo por llamada domina sobre el costo marginal por fila. Los
   números no se parecen.
2. **¿Cuántas alertas genera por día?** El paquete promete una tasa de falsas alarmas; con el
   volumen real del período eso se traduce en una carga de trabajo concreta.
3. **¿En qué se apoya el detector?** Si toda su capacidad descansa en una sola columna, es
   frágil ante un cambio de esa columna aunque su PR-AUC sea excelente.

Se empaqueta **Gaussian Mixture** y no el mejor por PR-AUC del split temporal. Deep SVDD gana
ahí (0.368 contra 0.263), pero el Módulo 5 mostró que su umbral promete 0,1% de falsas
alarmas y entrega 35%: es indesplegable tal cual. GMM queda segundo en ranking y primero en
lo que importa para operar — calibración (razón 1,4 contra 354) y dinero salvado.

Ejecución:

    python -m src.serving.run_serving
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler

from src.operations.temporal import get_temporal_data, period_index
from src.serving.explain import global_importance
from src.serving.package import DEFAULT_PACKAGE_PATH, DetectorPackage, build_package
from src.serving.predict import score_transactions
from src.unsupervised.families import GMMDensity

ALPHA = 0.01
N_LATENCIA = 200
N_ALERTAS_MOSTRADAS = 5


def medir_latencia(package: DetectorPackage, X: pd.DataFrame, n: int = N_LATENCIA) -> pd.DataFrame:
    """Latencia por transacción suelta contra latencia amortizada por lote.

    La primera es la que ve un sistema en línea; la segunda, la de un proceso por lotes. Se
    reportan juntas porque la diferencia entre ambas es el costo fijo por llamada, que un
    backtest nunca expone.
    """
    muestra = X.iloc[:n]

    # Una llamada por transacción, que es el caso en línea.
    inicio = time.perf_counter()
    for i in range(n):
        package.score(muestra.iloc[[i]])
    por_transaccion = (time.perf_counter() - inicio) / n

    # El mismo trabajo en una sola llamada.
    inicio = time.perf_counter()
    package.score(muestra)
    por_lote = (time.perf_counter() - inicio) / n

    # Con explicación, que es lo que cuesta una alerta accionable.
    inicio = time.perf_counter()
    score_transactions(package, muestra.iloc[:20], explain=True)
    con_explicacion = (time.perf_counter() - inicio) / 20

    return pd.DataFrame([
        {"modo": "de a una (en línea)", "ms_por_transaccion": por_transaccion * 1_000},
        {"modo": "en lote (backtest)", "ms_por_transaccion": por_lote * 1_000},
        {"modo": "con explicación", "ms_por_transaccion": con_explicacion * 1_000},
    ])


def carga_diaria(package: DetectorPackage, X: pd.DataFrame, steps, cutoff_step: int) -> pd.DataFrame:
    """Alertas por día que generaría el paquete sobre el tráfico real del período."""
    alertas = package.predict(X)
    periodos = period_index(steps, cutoff_step)

    filas = []
    for periodo in np.unique(periodos):
        seleccion = periodos == periodo
        if seleccion.sum() < 1_000:
            continue
        filas.append({
            "dia": int(periodo),
            "transacciones": int(seleccion.sum()),
            "alertas": int(alertas[seleccion].sum()),
            "tasa": float(alertas[seleccion].mean()),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    print("Cargando el split temporal del Módulo 5...")
    data = get_temporal_data()
    X_train, X_calib = data["X_train"], data["X_calib"]
    X_test, y_test, steps = data["X_test"], data["y_test"], data["steps_test"]

    scaler = RobustScaler().fit(X_train)
    detector = GMMDensity().fit(scaler.transform(X_train))

    package = build_package(
        detector, scaler, scaler.transform(X_calib), list(X_train.columns),
        alpha=ALPHA, detector_name="gmm_density",
        extra={"cutoff_step": int(data["cutoff_step"]), "n_train": int(len(X_train))},
    )
    ruta = package.save()

    print(f"\nPaquete guardado en {ruta}")
    print(f"  metadatos: {package.metadata}")
    print(f"  umbral (alpha={ALPHA}): {package.threshold:.4f}")

    # --- verificación de ida y vuelta: lo que se guarda es lo que se carga ---
    recargado = DetectorPackage.load(ruta)
    muestra = X_test.iloc[:500]
    if not np.allclose(recargado.score(muestra), package.score(muestra)):
        raise SystemExit("El paquete recargado no reproduce los scores originales.")
    print("  ida y vuelta verificada: el paquete recargado reproduce los scores")

    # --- 1. latencia ---
    print("\n=== Latencia ===")
    print(medir_latencia(recargado, X_test).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- 2. carga de alertas ---
    print("\n=== Alertas por día sobre el tráfico real ===")
    carga = carga_diaria(recargado, X_test, steps, data["cutoff_step"])
    print(carga.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nMediana de alertas por día: {carga['alertas'].median():.0f} "
          f"| tasa media: {carga['tasa'].mean():.4%} (prometida: {ALPHA:.2%})")

    detectadas = int(((recargado.predict(X_test)) & (np.asarray(y_test) == 1)).sum())
    print(f"Fraude capturado por las alertas: {detectadas} de {int(y_test.sum())} "
          f"({detectadas / max(1, int(y_test.sum())):.1%})")
    print(f"PR-AUC del detector empaquetado: "
          f"{average_precision_score(y_test, recargado.score(X_test)):.4f}")

    # --- 3. alertas explicadas ---
    print(f"\n=== Las {N_ALERTAS_MOSTRADAS} alertas más anómalas, explicadas ===")
    p_valores = recargado.p_values(X_test)
    top = X_test.iloc[np.argsort(p_valores)[:N_ALERTAS_MOSTRADAS]]
    explicadas = score_transactions(recargado, top, top_k=3)
    explicadas["es_fraude"] = np.asarray(y_test)[np.argsort(p_valores)[:N_ALERTAS_MOSTRADAS]]
    print(explicadas.to_string(float_format=lambda v: f"{v:.4g}"))

    # --- 4. en qué se apoya el detector ---
    print("\n=== En qué se apoya el detector (atribución global) ===")
    centro = recargado.scaler.transform(np.asarray(recargado.scaler.center_).reshape(1, -1)).ravel()
    importancia = global_importance(
        recargado.score_from_scaled,
        recargado.scaler.transform(X_test.iloc[:2_000]),
        centro,
        recargado.feature_names,
    )
    print(importancia.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

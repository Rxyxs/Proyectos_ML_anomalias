"""Módulo 5 — validación temporal: entrenar con el pasado, evaluar con el futuro.

Los módulos 2 a 4 comparten un split que muestrea filas **al azar** sobre un dataset que
tiene orden cronológico (`step` va de 1 a 743 horas, 31 días). Eso mete transacciones del
día 30 en el entrenamiento y del día 1 en la prueba: el modelo se evalúa sobre un pasado
que ya vio, y el número resultante es optimista respecto de lo que pasaría en producción,
donde siempre se predice hacia adelante.

Este módulo corrige eso. Corta por tiempo: entrena con las primeras horas y evalúa con las
posteriores, que es la única forma de estimar cómo envejece un detector.

Dos diferencias más con el split de los módulos anteriores, ambas deliberadas:

- **La prueba conserva la prevalencia real** (~0.24% de fraude en el período tardío) en vez
  de enriquecerse con todas las anomalías disponibles. El módulo 5 calcula umbrales,
  volúmenes de alerta y dinero salvado: sobre un conjunto enriquecido al 14% esas
  cantidades no significarían nada.
- **Se conservan `amount` y `step` sin escalar**, porque las métricas de costo se calculan
  en pesos y la evaluación por período necesita la marca de tiempo.

PaySim además trae un cambio de volumen severo —los primeros diez días concentran 3,19M
transacciones y los últimos diez apenas 298k— que un split aleatorio esconde por completo
y que este split expone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessing import clean_data
from src.features.build_features import build_features

TARGET_COLUMN = "isFraud"

# Fracción temporal de entrenamiento: el corte cae en el step que deja ese porcentaje de
# las horas por detrás. No se elige por cantidad de filas porque el volumen por hora no es
# constante — justamente el fenómeno que interesa observar.
TRAIN_TIME_FRACTION = 0.7

TRAIN_NORMAL_SIZE = 30_000

# Normales del período temprano reservadas para calibrar el umbral, disjuntas de las de
# ajuste. Calibrar sobre datos que el detector no usó para ajustarse es la práctica correcta
# por el mismo motivo que se separa validación de entrenamiento.
#
# Sirve además como ablación: si el umbral de un detector fallara por haberse calibrado
# sobre datos ya optimizados, el cuantil de este conjunto lo arreglaría. Sobre PaySim no lo
# arregla —Deep SVDD pasa de 0,389 a 0,354 de FPR observada cuando promete 0,001—, y esa
# falta de diferencia es informativa: descarta el sobreajuste y deja como única explicación
# el desplazamiento de la distribución entre el período de ajuste y el de evaluación.
CALIB_NORMAL_SIZE = 30_000

TEST_SIZE = 300_000


def temporal_cutoff(steps: pd.Series, train_fraction: float = TRAIN_TIME_FRACTION) -> int:
    """Step que separa el período de entrenamiento del de prueba."""
    return int(steps.quantile(train_fraction))


def get_temporal_data(
    train_normal_size: int = TRAIN_NORMAL_SIZE,
    calib_normal_size: int = CALIB_NORMAL_SIZE,
    test_size: int = TEST_SIZE,
    train_fraction: float = TRAIN_TIME_FRACTION,
    random_state: int = 42,
) -> dict:
    """Prepara el split temporal con prevalencia real en la prueba.

    Devuelve un diccionario con:
    - `X_train`: transacciones normales del período temprano (el ajuste nunca ve fraude);
    - `X_calib`: normales del mismo período temprano, **disjuntas** de `X_train`, para
      calibrar el umbral sobre datos que el detector no usó para ajustarse;
    - `X_test`, `y_test`: muestra del período tardío **sin enriquecer**;
    - `amounts_test`, `steps_test`: monto y hora sin escalar, para costo y análisis temporal;
    - `cutoff_step`: el step del corte.
    """
    df = load_raw_data()
    cutoff = temporal_cutoff(df["step"], train_fraction)

    df = clean_data(df)
    df = build_features(df)

    early = df[df["step"] <= cutoff]
    late = df[df["step"] > cutoff]

    normal_early = early[early[TARGET_COLUMN] == 0]
    reserved = normal_early.sample(
        n=min(train_normal_size + calib_normal_size, len(normal_early)), random_state=random_state
    )
    train = reserved.iloc[:train_normal_size]
    calib = reserved.iloc[train_normal_size:]

    # Muestreo simple del período tardío: preserva la proporción de fraude tal como ocurre.
    test = late.sample(n=min(test_size, len(late)), random_state=random_state)

    return {
        "X_train": train.drop(columns=[TARGET_COLUMN]),
        "X_calib": calib.drop(columns=[TARGET_COLUMN]),
        "X_test": test.drop(columns=[TARGET_COLUMN]),
        "y_test": test[TARGET_COLUMN],
        "amounts_test": test["amount"],
        "steps_test": test["step"],
        "cutoff_step": cutoff,
    }


def period_index(steps: pd.Series, cutoff_step: int, hours_per_period: int = 24) -> np.ndarray:
    """Agrupa cada transacción en períodos consecutivos desde el corte (por defecto, días)."""
    return ((np.asarray(steps) - cutoff_step - 1) // hours_per_period).astype(int)


def evaluate_by_period(
    y_true: pd.Series,
    scores: np.ndarray,
    steps: pd.Series,
    cutoff_step: int,
    metric,
    hours_per_period: int = 24,
    min_positives: int = 5,
    min_samples: int = 1_000,
) -> pd.DataFrame:
    """Aplica `metric(y, scores)` período a período para ver cómo envejece el detector.

    Se descartan los períodos con menos de `min_positives` fraudes o menos de `min_samples`
    transacciones. Ambos filtros son necesarios y el segundo no es opcional en PaySim: el
    volumen diario cae de ~62.000 transacciones a **23** hacia el final del mes, y sobre 23
    filas con prevalencia disparada cualquier detector obtiene métricas perfectas. Sin este
    filtro, la curva mostraría una "mejora" espectacular que es puro colapso muestral.

    Ojo con la métrica que se pase: PR-AUC depende de la prevalencia, y la prevalencia varía
    entre períodos, así que no es comparable a lo largo del tiempo. Para leer degradación
    conviene una métrica independiente de la prevalencia, como ROC-AUC.
    """
    periods = period_index(steps, cutoff_step, hours_per_period)
    y_arr = np.asarray(y_true)

    rows = []
    for period in np.unique(periods):
        selected = periods == period
        n_period = int(selected.sum())
        positives = int(y_arr[selected].sum())
        if positives < min_positives or n_period < min_samples:
            continue
        rows.append({
            "period": int(period),
            "n": n_period,
            "positives": positives,
            "prevalence": positives / n_period,
            "metric": float(metric(y_arr[selected], scores[selected])),
        })
    return pd.DataFrame(rows)


def volume_drift(steps: pd.Series, cutoff_step: int, hours_per_period: int = 24) -> pd.DataFrame:
    """Volumen de transacciones por período, para exponer el cambio de carga del dataset."""
    periods = period_index(steps, cutoff_step, hours_per_period)
    counts = pd.Series(periods).value_counts().sort_index()
    return pd.DataFrame({"period": counts.index.astype(int), "n": counts.to_numpy()})

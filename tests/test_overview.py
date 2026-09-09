"""Pruebas del panorama del dataset (src.data.overview).

Los números de esta sección del README sostienen decisiones de diseño de varios módulos —el
split temporal, las métricas en pesos, el filtro de volumen— así que conviene que las
funciones que los calculan estén fijadas por pruebas y no solo por una corrida.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.overview import (
    amount_summary,
    class_balance,
    fraud_by_type,
    volume_by_day,
)


@pytest.fixture
def transacciones() -> pd.DataFrame:
    """Diez transacciones: dos fraudes, ambos TRANSFER, repartidas en dos días."""
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5, 25, 26, 27, 28, 29],
        "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "PAYMENT", "TRANSFER",
                 "PAYMENT", "PAYMENT", "PAYMENT", "PAYMENT", "PAYMENT"],
        "amount": [1_000.0, 10.0, 20.0, 10.0, 2_000.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "isFraud": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    })


def test_el_balance_cuenta_la_prevalencia(transacciones):
    balance = class_balance(transacciones)
    assert balance == {"n_total": 10, "n_fraud": 2, "prevalence": 0.2}


def test_el_balance_no_falla_sin_fraude(transacciones):
    limpio = transacciones.assign(isFraud=0)
    assert class_balance(limpio)["prevalence"] == 0.0


def test_la_tasa_por_tipo_ordena_de_mayor_a_menor(transacciones):
    tabla = fraud_by_type(transacciones)
    assert list(tabla["tasa"]) == sorted(tabla["tasa"], reverse=True)
    assert tabla.iloc[0]["type"] == "TRANSFER"


def test_la_tasa_por_tipo_marca_los_tipos_sin_fraude(transacciones):
    tabla = fraud_by_type(transacciones).set_index("type")
    assert tabla.loc["PAYMENT", "fraude"] == 0
    assert tabla.loc["CASH_OUT", "fraude"] == 0
    assert tabla.loc["TRANSFER", "tasa"] == 1.0


def test_el_resumen_de_montos_separa_las_dos_clases(transacciones):
    resumen = amount_summary(transacciones).set_index("clase")

    assert resumen.loc["fraude", "n"] == 2
    assert resumen.loc["legítima", "n"] == 8
    assert resumen.loc["fraude", "mediana"] == 1_500.0
    # El monto del fraude debe superar ampliamente al legítimo: es lo que justifica las
    # métricas en pesos del Módulo 5.
    assert resumen.loc["fraude", "mediana"] > 10 * resumen.loc["legítima", "mediana"]


def test_el_volumen_por_dia_agrupa_de_a_24_horas(transacciones):
    dias = volume_by_day(transacciones)

    assert list(dias["dia"]) == [0, 1]
    assert list(dias["n"]) == [5, 5]
    assert list(dias["fraude"]) == [2, 0]


def test_el_volumen_por_dia_expone_una_caida_de_carga():
    # El fenómeno real de PaySim: el conteo de fraude se mantiene mientras el volumen cae,
    # así que la tasa se dispara sin que cambie nada del fraude.
    filas = ([{"step": 1, "type": "TRANSFER", "amount": 1.0, "isFraud": 0}] * 1_000
             + [{"step": 25, "type": "TRANSFER", "amount": 1.0, "isFraud": 0}] * 10)
    df = pd.DataFrame(filas)

    dias = volume_by_day(df)
    assert dias["n"].iloc[0] / dias["n"].iloc[-1] == 100.0


def test_la_tasa_diaria_sube_cuando_cae_el_volumen():
    filas = ([{"step": 1, "type": "TRANSFER", "amount": 1.0, "isFraud": 0}] * 999
             + [{"step": 1, "type": "TRANSFER", "amount": 1.0, "isFraud": 1}]
             + [{"step": 25, "type": "TRANSFER", "amount": 1.0, "isFraud": 0}] * 9
             + [{"step": 25, "type": "TRANSFER", "amount": 1.0, "isFraud": 1}])
    dias = volume_by_day(pd.DataFrame(filas))

    assert dias["fraude"].iloc[0] == dias["fraude"].iloc[-1] == 1
    assert dias["tasa"].iloc[-1] > 50 * dias["tasa"].iloc[0]

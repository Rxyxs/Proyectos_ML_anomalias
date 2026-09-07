"""Pruebas unitarias del split y la evaluación temporal (src.operations.temporal).

`get_temporal_data` no se prueba aquí porque descarga y procesa PaySim completo; lo que sí
se prueba es toda la aritmética de períodos y corte, que es donde un error de un índice
haría que el módulo mida el período equivocado sin que nada falle a la vista.
"""
import numpy as np
import pandas as pd

from src.operations.temporal import (
    evaluate_by_period,
    period_index,
    temporal_cutoff,
    volume_drift,
)


def test_el_corte_deja_la_fraccion_pedida_de_horas_por_detras():
    steps = pd.Series(np.arange(1, 101))
    assert temporal_cutoff(steps, 0.7) == 70


def test_los_periodos_empiezan_en_cero_justo_despues_del_corte():
    # La primera hora posterior al corte (101) abre el período 0, y el día dura 24 horas.
    steps = pd.Series([101, 124, 125, 148])
    np.testing.assert_array_equal(period_index(steps, cutoff_step=100), [0, 0, 1, 1])


def test_los_periodos_usan_la_ventana_configurada():
    steps = pd.Series([101, 102, 103, 104])
    np.testing.assert_array_equal(
        period_index(steps, cutoff_step=100, hours_per_period=2), [0, 0, 1, 1]
    )


def test_la_evaluacion_por_periodo_descarta_los_periodos_sin_positivos_suficientes():
    # Período 0 con 6 fraudes, período 1 con uno solo: el segundo se descarta porque con un
    # positivo cualquier métrica de ranking es ruido, no degradación.
    steps = pd.Series([101] * 20 + [130] * 20)
    y = pd.Series([1] * 6 + [0] * 14 + [1] + [0] * 19)
    scores = np.linspace(1, 0, 40)

    # min_samples=1 aísla el filtro que esta prueba verifica: el de positivos.
    tabla = evaluate_by_period(y, scores, steps, cutoff_step=100, metric=lambda a, b: 1.0,
                               min_positives=5, min_samples=1)

    assert list(tabla["period"]) == [0]
    assert tabla["positives"].iloc[0] == 6


def test_la_evaluacion_por_periodo_reporta_prevalencia_y_tamano():
    steps = pd.Series([101] * 100)
    y = pd.Series([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)

    tabla = evaluate_by_period(y, scores, steps, cutoff_step=100, metric=lambda a, b: 0.5,
                               min_samples=1)

    assert tabla["n"].iloc[0] == 100
    assert tabla["positives"].iloc[0] == 10
    assert tabla["prevalence"].iloc[0] == 0.1
    assert tabla["metric"].iloc[0] == 0.5


def test_la_evaluacion_por_periodo_aplica_la_metrica_solo_a_su_periodo():
    # El período 0 tiene un ranking perfecto y el 1 uno invertido: si la función mezclara
    # los períodos, ambos darían el mismo número intermedio.
    steps = pd.Series([101] * 10 + [130] * 10)
    y = pd.Series([1] * 5 + [0] * 5 + [0] * 5 + [1] * 5)
    scores = np.concatenate([np.linspace(1, 0, 10), np.linspace(1, 0, 10)])

    def fraccion_positiva_arriba(y_periodo, s_periodo):
        return float(y_periodo[np.argsort(s_periodo)[::-1][:5]].mean())

    tabla = evaluate_by_period(y, scores, steps, 100, fraccion_positiva_arriba, min_samples=1)

    assert tabla.loc[tabla.period == 0, "metric"].iloc[0] == 1.0
    assert tabla.loc[tabla.period == 1, "metric"].iloc[0] == 0.0


def test_el_volumen_por_periodo_cuenta_todas_las_transacciones():
    steps = pd.Series([101] * 7 + [130] * 3)
    volumen = volume_drift(steps, cutoff_step=100)

    assert list(volumen["period"]) == [0, 1]
    assert list(volumen["n"]) == [7, 3]
    assert volumen["n"].sum() == len(steps)


def test_el_volumen_expone_una_caida_de_carga():
    # Es el fenómeno que el split aleatorio esconde: el mismo detector enfrenta cargas muy
    # distintas según el día.
    steps = pd.Series([101] * 1000 + [130] * 50)
    volumen = volume_drift(steps, cutoff_step=100)

    assert volumen["n"].iloc[0] / volumen["n"].iloc[-1] == 20.0


def test_descarta_los_periodos_con_muy_pocas_transacciones():
    # El caso real de PaySim: el volumen diario cae de decenas de miles a 23 filas hacia el
    # final del mes. Sobre 23 filas cualquier detector saca métricas perfectas, así que un
    # período así no puede entrar en una curva de degradación.
    steps = pd.Series([101] * 5_000 + [130] * 23)
    y = pd.Series([1] * 50 + [0] * 4_950 + [1] * 12 + [0] * 11)
    scores = np.linspace(1, 0, 5_023)

    tabla = evaluate_by_period(y, scores, steps, cutoff_step=100, metric=lambda a, b: 1.0,
                               min_positives=5, min_samples=1_000)

    assert list(tabla["period"]) == [0], "el día de 23 transacciones debe quedar fuera"


def test_sin_filtro_de_volumen_entraria_el_periodo_degenerado():
    # Confirma que el filtro es lo que excluye al período chico y no otra condición: con
    # min_samples bajo, el mismo día vuelve a aparecer.
    steps = pd.Series([101] * 5_000 + [130] * 23)
    y = pd.Series([1] * 50 + [0] * 4_950 + [1] * 12 + [0] * 11)
    scores = np.linspace(1, 0, 5_023)

    tabla = evaluate_by_period(y, scores, steps, cutoff_step=100, metric=lambda a, b: 1.0,
                               min_positives=5, min_samples=10)

    assert list(tabla["period"]) == [0, 1]

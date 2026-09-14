"""Pruebas de la recalibración continua y el monitor sin etiquetas (Módulo 9).

Dos propiedades importan más que el resto. La primera es que no haya mirada hacia adelante:
si un día se calibra con sus propios scores, la tasa de alertas sale perfecta por
construcción y el experimento no mide nada. La segunda es el lazo de retroalimentación de
excluir las alertas de la ventana, que es fácil de introducir sin darse cuenta.
"""
import numpy as np
import pandas as pd
import pytest

from src.conformal.conformal import conformal_p_values, conformal_threshold
from src.serving.recalibration import RollingCalibrator, tail_ratio, uniformity_gap
from src.serving.run_recalibration import monitor_agreement, simulate, summarize

ALPHA = 0.01


def _alert_rate(calibrador: RollingCalibrator, scores: np.ndarray) -> float:
    return float((calibrador.p_values(scores) <= calibrador.alpha).mean())


# ---------------------------------------------------------------- monitores sin etiquetas

def test_la_razon_de_cola_ronda_uno_bajo_intercambiabilidad():
    n = 40_000
    rng = np.random.default_rng(0)
    p = conformal_p_values(rng.normal(size=n), rng.normal(size=n))
    # Error binomial de la fracción observada, más un término del mismo orden por la calibración.
    tolerancia = 8 * np.sqrt(ALPHA * (1 - ALPHA) / n) / ALPHA
    assert tail_ratio(p, ALPHA) == pytest.approx(1.0, abs=tolerancia)


def test_la_razon_de_cola_se_dispara_con_un_corrimiento():
    rng = np.random.default_rng(1)
    p = conformal_p_values(rng.normal(size=20_000), rng.normal(loc=1.5, size=20_000))
    assert tail_ratio(p, ALPHA) > 3


def test_la_brecha_uniforme_distingue_intercambiable_de_corrido():
    rng = np.random.default_rng(2)
    calibracion = rng.normal(size=20_000)
    igual = uniformity_gap(conformal_p_values(calibracion, rng.normal(size=20_000)))
    corrido = uniformity_gap(conformal_p_values(calibracion, rng.normal(loc=1.0, size=20_000)))

    assert igual < 0.03
    assert corrido > 5 * igual


def test_la_razon_de_cola_mezcla_deriva_con_prevalencia():
    """Sesgo documentado: sin etiquetas, más fraude se ve igual que peor calibración.

    Con la calibración de lo legítimo perfecta, sumar un 5% de transacciones extremas sube la
    razón aunque la FPR no cambie.
    """
    rng = np.random.default_rng(3)
    calibracion = rng.normal(size=20_000)
    legitimas = rng.normal(size=19_000)
    con_fraude = np.concatenate([legitimas, np.full(1_000, 50.0)])

    solo_legitimas = tail_ratio(conformal_p_values(calibracion, legitimas), ALPHA)
    mezclado = tail_ratio(conformal_p_values(calibracion, con_fraude), ALPHA)

    assert mezclado > solo_legitimas + 4


@pytest.mark.parametrize("funcion", [lambda p: tail_ratio(p, ALPHA), uniformity_gap])
def test_los_monitores_rechazan_una_entrada_vacia(funcion):
    with pytest.raises(ValueError, match="vacío"):
        funcion(np.array([]))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2])
def test_la_razon_de_cola_rechaza_un_alpha_invalido(alpha):
    with pytest.raises(ValueError, match="alpha debe estar"):
        tail_ratio(np.array([0.5]), alpha)


# ---------------------------------------------------------------- calibrador

def test_el_calibrador_sin_actualizar_reproduce_el_p_valor_conforme():
    rng = np.random.default_rng(4)
    calibracion, prueba = rng.normal(size=5_000), rng.normal(size=1_000)
    calibrador = RollingCalibrator(calibracion, window_size=10_000, alpha=ALPHA)

    np.testing.assert_allclose(calibrador.p_values(prueba), conformal_p_values(calibracion, prueba))
    assert calibrador.threshold == conformal_threshold(calibracion, ALPHA)


def test_la_ventana_no_supera_su_tamano_y_conserva_lo_mas_reciente():
    calibrador = RollingCalibrator(np.arange(100.0), window_size=150, alpha=ALPHA)
    calibrador.update(np.arange(100.0, 200.0))

    assert calibrador.n_calibration == 150
    # Se descartaron los 50 más viejos (0..49) y quedaron 50..199 en orden.
    np.testing.assert_array_equal(calibrador.scores_, np.arange(50.0, 200.0))


def test_la_ventana_inicial_se_recorta_al_tamano_pedido():
    calibrador = RollingCalibrator(np.arange(500.0), window_size=200, alpha=ALPHA)
    np.testing.assert_array_equal(calibrador.scores_, np.arange(300.0, 500.0))


def test_recalibrar_devuelve_la_tasa_de_alertas_tras_un_corrimiento():
    """El escenario del Módulo 8: el tráfico se corre y la tasa estática se dispara."""
    rng = np.random.default_rng(5)
    calibrador = RollingCalibrator(rng.normal(size=10_000), window_size=10_000, alpha=ALPHA)

    trafico_corrido = rng.normal(loc=1.5, size=10_000)
    antes = _alert_rate(calibrador, rng.normal(loc=1.5, size=10_000))

    calibrador.update(trafico_corrido)
    despues = _alert_rate(calibrador, rng.normal(loc=1.5, size=10_000))

    assert antes > 5 * ALPHA
    assert despues == pytest.approx(ALPHA, abs=0.006)


def test_excluir_alertas_saca_de_la_ventana_lo_que_se_habria_alertado():
    rng = np.random.default_rng(6)
    calibracion = rng.normal(size=10_000)
    nuevos = rng.normal(size=10_000)

    calibrador = RollingCalibrator(calibracion, window_size=20_000, alpha=ALPHA, exclude_alerts=True)
    alertados = int((conformal_p_values(calibracion, nuevos) <= ALPHA).sum())
    calibrador.update(nuevos)

    assert calibrador.n_calibration == 10_000 + (10_000 - alertados)


def test_excluir_alertas_cierra_un_lazo_que_infla_la_tasa():
    """Recortar la cola legítima en cada actualización baja el umbral una y otra vez.

    Con tráfico estable y sin ningún fraude, incluir todo mantiene la tasa en alpha; excluir lo
    alertado la hace crecer ronda tras ronda.
    """
    rng = np.random.default_rng(7)
    inicial = rng.normal(size=5_000)
    con_todo = RollingCalibrator(inicial, window_size=5_000, alpha=ALPHA)
    sin_alertas = RollingCalibrator(inicial, window_size=5_000, alpha=ALPHA, exclude_alerts=True)

    for _ in range(5):
        dia = rng.normal(size=5_000)
        con_todo.update(dia)
        sin_alertas.update(dia)

    prueba = rng.normal(size=50_000)
    assert _alert_rate(con_todo, prueba) == pytest.approx(ALPHA, abs=0.005)
    assert _alert_rate(sin_alertas, prueba) > 3 * ALPHA
    assert sin_alertas.threshold < con_todo.threshold


@pytest.mark.parametrize("kwargs, mensaje", [
    ({"window_size": 0}, "window_size"),
    ({"alpha": 1.5}, "alpha debe estar"),
])
def test_el_calibrador_valida_sus_parametros(kwargs, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        RollingCalibrator(np.ones(10), **kwargs)


def test_el_calibrador_rechaza_una_calibracion_vacia():
    with pytest.raises(ValueError, match="vacío"):
        RollingCalibrator(np.array([]))


# ---------------------------------------------------------------- simulación

@pytest.fixture
def escenario():
    """Tres días estables y tres corridos, con algo de fraude extremo cada día."""
    rng = np.random.default_rng(8)
    calibracion = rng.normal(size=6_000)
    scores, etiquetas, dias = [], [], []
    for dia in range(6):
        loc = 0.0 if dia < 3 else 1.5
        legitimas = rng.normal(loc=loc, size=3_000)
        fraude = rng.normal(loc=8.0, size=10)
        scores.append(np.concatenate([legitimas, fraude]))
        etiquetas.append(np.r_[np.zeros(3_000), np.ones(10)])
        dias.append(np.full(3_010, dia))
    return calibracion, np.concatenate(scores), np.concatenate(etiquetas), np.concatenate(dias)


def test_la_simulacion_no_mira_hacia_adelante(escenario):
    """El primer día todas las estrategias usan la misma calibración inicial.

    Si alguna actualizara la ventana antes de puntuar, su día 0 ya diferiría del estático.
    """
    tabla = simulate(*escenario, alpha=ALPHA, window_size=6_000)
    dia_0 = tabla[tabla["dia"] == 0].set_index("estrategia")

    assert dia_0["alertas"].nunique() == 1
    assert dia_0["umbral"].nunique() == 1


def test_la_ventana_se_acerca_a_lo_prometido_despues_del_corrimiento(escenario):
    tabla = simulate(*escenario, alpha=ALPHA, window_size=6_000)
    ultimo = tabla[tabla["dia"] == 5].set_index("estrategia")

    assert ultimo.loc["estatico", "fpr"] > 5 * ALPHA
    assert abs(ultimo.loc["ventana", "fpr"] - ALPHA) < abs(ultimo.loc["estatico", "fpr"] - ALPHA)


def test_el_resumen_tiene_una_fila_por_estrategia(escenario):
    tabla = simulate(*escenario, alpha=ALPHA, window_size=6_000)
    resumen = summarize(tabla, alpha=ALPHA, min_rows=1_000)

    assert list(resumen["estrategia"]) == [
        "estatico", "ventana", "ventana_oraculo", "ventana_sin_alertas"
    ]
    assert resumen["fraude_capturado"].between(0, 1).all()
    assert (resumen["alertas"] == tabla.groupby("estrategia", sort=False)["alertas"].sum().values).all()


def test_el_resumen_descarta_dias_chicos_de_las_tasas():
    tabla = pd.DataFrame({
        "estrategia": ["estatico"] * 2, "dia": [0, 1], "n": [5_000, 50],
        "tasa_alerta": [0.01, 0.90], "fpr": [0.01, 0.90], "alertas": [50, 45],
        "fraudes": [2, 3], "capturados": [1, 3],
    })
    resumen = summarize(tabla, alpha=ALPHA, min_rows=1_000).iloc[0]

    assert resumen["tasa_media"] == pytest.approx(0.01)
    assert resumen["alertas"] == 95, "las alertas de días chicos siguen contando"
    assert resumen["fraude_capturado"] == pytest.approx(4 / 5)


def test_el_monitor_sin_etiquetas_ordena_como_la_fpr_real(escenario):
    tabla = simulate(*escenario, alpha=ALPHA, window_size=6_000)
    acuerdo = monitor_agreement(tabla, estrategia="estatico", alpha=ALPHA, min_rows=1_000)

    assert acuerdo["dias"] == 6
    assert acuerdo["spearman"] > 0.7


def test_el_oraculo_aisla_el_costo_de_contaminar_la_ventana_con_fraude(escenario):
    """La variante oráculo alimenta la ventana solo con legítimas, con etiquetas al instante.

    No es desplegable —en producción las etiquetas llegan semanas tarde— pero separa el efecto
    de absorber la deriva, que es el buscado, del de meter fraude en la calibración.
    """
    tabla = simulate(*escenario, alpha=ALPHA, window_size=6_000)
    ultimo = tabla[tabla["dia"] == 5].set_index("estrategia")

    assert ultimo.loc["ventana_oraculo", "umbral"] < ultimo.loc["ventana", "umbral"]
    assert ultimo.loc["ventana_oraculo", "capturados"] >= ultimo.loc["ventana", "capturados"]

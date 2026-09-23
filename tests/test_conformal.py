"""Pruebas unitarias de la detección conforme (src.conformal.conformal).

La propiedad que importa es la garantía: bajo intercambiabilidad, rechazar con p <= alpha
produce una tasa de falsas alarmas en torno a alpha. Y su contracara, igual de importante:
cuando la intercambiabilidad se rompe, la garantía se rompe — si una prueba no verificara
eso, el módulo estaría vendiendo una promesa incondicional que no tiene.
"""
import numpy as np
import pytest
from scipy import stats

from src.conformal.conformal import (
    ConformalDetector,
    conformal_p_values,
    conformal_threshold,
    coverage_report,
)
from src.unsupervised.families import HBOS


def test_los_p_valores_viven_en_el_intervalo_valido():
    rng = np.random.default_rng(0)
    p = conformal_p_values(rng.normal(size=1_000), rng.normal(size=500))

    assert (p > 0).all(), "el +1 del numerador impide que un p-valor sea exactamente 0"
    assert (p <= 1).all()


def test_un_score_mas_alto_da_un_p_valor_mas_bajo():
    calibracion = np.arange(100, dtype=float)
    p = conformal_p_values(calibracion, np.array([10.0, 50.0, 90.0]))

    assert p[0] > p[1] > p[2]


def test_el_p_valor_minimo_lo_fija_el_tamano_de_calibracion():
    # Con n puntos de calibración, el p-valor más chico posible es 1/(n+1): no se puede
    # afirmar una significancia mayor que la que la muestra permite resolver.
    calibracion = np.arange(99, dtype=float)
    p = conformal_p_values(calibracion, np.array([1e9]))

    assert p[0] == pytest.approx(1 / 100)


def test_la_garantia_se_cumple_bajo_intercambiabilidad():
    # Calibración y prueba del mismo sorteo. La garantía es marginal, así que la tasa
    # observada fluctúa alrededor de alpha; se admite un margen por ruido muestral.
    rng = np.random.default_rng(42)
    calibracion = rng.normal(size=20_000)
    prueba = rng.normal(size=20_000)

    p = conformal_p_values(calibracion, prueba)
    for alpha in (0.01, 0.05):
        observada = float((p <= alpha).mean())
        assert observada == pytest.approx(alpha, rel=0.25)


def test_la_garantia_se_rompe_cuando_hay_desplazamiento():
    # Contracara de la prueba anterior: la garantía es condicional a la intercambiabilidad,
    # y el módulo no debe presentarla como incondicional.
    rng = np.random.default_rng(3)
    calibracion = rng.normal(size=20_000)
    desplazada = rng.normal(loc=1.5, size=20_000)

    p = conformal_p_values(calibracion, desplazada)
    assert float((p <= 0.01).mean()) > 0.05


def test_el_umbral_conforme_reproduce_el_rechazo_por_p_valor():
    rng = np.random.default_rng(7)
    calibracion = rng.normal(size=5_000)
    prueba = rng.normal(size=3_000)

    umbral = conformal_threshold(calibracion, 0.01)
    p = conformal_p_values(calibracion, prueba)

    np.testing.assert_array_equal(prueba >= umbral, p <= 0.01)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5, 2.0])
def test_el_umbral_conforme_rechaza_un_alpha_invalido(alpha):
    with pytest.raises(ValueError, match="alpha debe estar"):
        conformal_threshold(np.zeros(10), alpha)


def test_un_alpha_inalcanzable_devuelve_umbral_infinito():
    # Con 10 puntos de calibración el p-valor más chico es 1/11 = 0.09: pedir 0.001 es
    # imposible, y el umbral correcto es el que no marca nada.
    assert conformal_threshold(np.arange(10, dtype=float), 0.001) == np.inf


def test_rechaza_una_calibracion_vacia():
    with pytest.raises(ValueError, match="no puede estar vacío"):
        conformal_p_values(np.array([]), np.array([1.0]))


def test_el_reporte_mide_la_fpr_solo_sobre_las_legitimas():
    rng = np.random.default_rng(11)
    calibracion = rng.normal(size=10_000)
    legitimas = rng.normal(size=5_000)

    solo_legitimas = coverage_report(calibracion, legitimas, np.zeros(5_000, dtype=int))

    con_fraude = coverage_report(
        calibracion,
        np.concatenate([legitimas, np.full(500, 50.0)]),
        np.concatenate([np.zeros(5_000, dtype=int), np.ones(500, dtype=int)]),
    )

    np.testing.assert_allclose(
        solo_legitimas["fpr_observada"], con_fraude["fpr_observada"], rtol=1e-9
    )


def test_el_reporte_expresa_la_desviacion_como_razon():
    rng = np.random.default_rng(5)
    reporte = coverage_report(rng.normal(size=10_000), rng.normal(size=5_000),
                              np.zeros(5_000, dtype=int), alphas=(0.01,))

    fila = reporte.iloc[0]
    assert fila["razon_fpr_alpha"] == pytest.approx(fila["fpr_observada"] / fila["alpha"])


def test_el_detector_conforme_envuelve_cualquier_detector():
    rng = np.random.default_rng(0)
    entrenamiento = rng.normal(size=(2_000, 4))
    calibracion = rng.normal(size=(2_000, 4))
    prueba = np.vstack([rng.normal(size=(100, 4)), np.full((10, 4), 25.0)])

    conforme = ConformalDetector(HBOS(), alpha=0.05).fit(entrenamiento, calibracion)
    p = conforme.p_values(prueba)

    assert p.shape == (110,)
    assert p[-10:].mean() < p[:100].mean(), "las anomalías deben recibir p-valores más bajos"
    assert conforme.predict(prueba)[-10:].all()


def test_el_detector_conforme_exige_ajuste_previo():
    with pytest.raises(ValueError, match="antes de calcular p-valores"):
        ConformalDetector(HBOS()).p_values(np.zeros((3, 4)))


# --------------------------------------------------- Día 17: uniformidad bajo H0

def test_los_p_valores_son_uniformes_bajo_intercambiabilidad():
    """La garantía P(p(x) <= alpha) <= alpha para todo alpha, en muestra finita,
    es exactamente la afirmación de que los p-valores conformes son (super)uniformes
    en [0, 1] bajo H0 (un punto legítimo, intercambiable con la calibración). Un
    test de Kolmogorov-Smirnov contra Uniform(0,1) lo confirma directamente en vez
    de solo chequear un par de cuantiles alpha sueltos."""
    rng = np.random.default_rng(17)
    calibracion = rng.normal(size=20_000)
    legitimas = rng.normal(size=20_000)

    p = conformal_p_values(calibracion, legitimas)
    estadistico, p_valor_ks = stats.kstest(p, "uniform")

    # No se rechaza H0 (uniforme) a un nivel de significancia convencional. Con
    # 20.000 puntos el test tiene potencia de sobra para detectar un desvío real;
    # un p-valor de KS chico aca significaría que la implementación no es uniforme.
    assert p_valor_ks > 0.01, (
        f"KS rechazó uniformidad bajo H0 (estadístico={estadistico:.4f}, "
        f"p={p_valor_ks:.4g}) -- los p-valores conformes deberían ser uniformes "
        f"bajo intercambiabilidad"
    )


def test_los_p_valores_se_sesgan_hacia_cero_bajo_anomalias():
    """Contracara de la prueba anterior: bajo H1 (un punto genuinamente anómalo,
    no intercambiable con la calibración), los p-valores tienen que dejar de ser
    uniformes y concentrarse cerca de 0. Se verifica con el mismo test de KS
    (ahora se espera que SÍ rechace uniformidad) y con dominancia estocástica
    directa contra los p-valores de tráfico legítimo."""
    rng = np.random.default_rng(23)
    calibracion = rng.normal(size=20_000)
    legitimas = rng.normal(size=5_000)
    anomalias = rng.normal(loc=6.0, scale=1.0, size=5_000)  # muy por fuera de la calibración

    p_legit = conformal_p_values(calibracion, legitimas)
    p_anom = conformal_p_values(calibracion, anomalias)

    _, p_valor_ks_legit = stats.kstest(p_legit, "uniform")
    _, p_valor_ks_anom = stats.kstest(p_anom, "uniform")

    assert p_valor_ks_legit > 0.01, "las legítimas deberían seguir pasando el chequeo de uniformidad"
    assert p_valor_ks_anom < 1e-10, "las anómalas NO deberían superar el chequeo de uniformidad"

    assert np.median(p_anom) < np.median(p_legit)
    assert np.median(p_anom) < 0.01, "las anomalías deberían concentrarse cerca de p=0"

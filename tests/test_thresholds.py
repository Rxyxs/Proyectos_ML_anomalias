"""Pruebas unitarias de la selección de umbral (src.operations.thresholds).

La propiedad central que se verifica es la que hace útil al umbral por cuantil: sobre datos
que siguen la misma distribución que el entrenamiento, `alpha` debe reproducirse como tasa
de falsos positivos. Que eso se rompa con datos posteriores es el hallazgo del módulo, pero
tiene que valer cuando no hay drift, o la regla no serviría de nada.
"""
import numpy as np
import pytest

from src.operations.thresholds import (
    capacity_threshold,
    cost_optimal_threshold,
    empirical_alarm_rate,
    quantile_threshold,
    threshold_report,
)


def test_el_umbral_por_cuantil_reproduce_alpha_sin_drift():
    n = 50_000
    rng = np.random.default_rng(42)
    train = rng.normal(size=n)
    test_legitimo = rng.normal(size=n)

    for alpha in (0.001, 0.01, 0.05):
        umbral = quantile_threshold(train, alpha)
        observado = empirical_alarm_rate(test_legitimo, umbral)
        # La tolerancia sale del ruido de muestreo, no de un porcentaje elegido a dedo: con
        # n muestras, la fracción observada tiene error binomial sqrt(alpha(1-alpha)/n), y
        # el cuantil de entrenamiento aporta un error del mismo orden. Cuatro desviaciones
        # cubren ambos sin volver la prueba inútilmente laxa para los alphas grandes.
        tolerancia = 4 * np.sqrt(alpha * (1 - alpha) / n) * 2
        assert observado == pytest.approx(alpha, abs=tolerancia)


def test_un_alpha_mas_chico_exige_un_umbral_mas_alto():
    rng = np.random.default_rng(0)
    train = rng.normal(size=10_000)
    assert quantile_threshold(train, 0.001) > quantile_threshold(train, 0.05)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_rechaza_un_alpha_fuera_del_intervalo(alpha):
    with pytest.raises(ValueError, match="alpha debe estar"):
        quantile_threshold(np.zeros(10), alpha)


def test_el_umbral_por_capacidad_entrega_la_cantidad_de_alertas_pedida():
    scores = np.arange(1000, dtype=float)
    umbral = capacity_threshold(scores, alerts_per_period=10, n_periods=5)

    assert int((scores >= umbral).sum()) == 50


def test_el_umbral_por_capacidad_no_falla_si_se_pide_mas_de_lo_que_hay():
    scores = np.arange(10, dtype=float)
    umbral = capacity_threshold(scores, alerts_per_period=100, n_periods=10)

    assert int((scores >= umbral).sum()) == 10


def test_el_umbral_optimo_evita_revisar_cuando_no_conviene():
    # Un solo fraude barato y mucho ruido caro de revisar: revisar de más destruye valor,
    # así que el óptimo debe alertar poco.
    rng = np.random.default_rng(1)
    y = np.zeros(500, dtype=int)
    y[0] = 1
    scores = rng.normal(size=500)
    scores[0] = 10.0
    amounts = np.full(500, 100.0)

    resultado = cost_optimal_threshold(y, scores, amounts, review_cost=1_000.0)
    assert resultado["n_alerts"] < 50


def test_el_umbral_optimo_amplia_la_revision_cuando_el_fraude_es_caro():
    # Mismo escenario pero con fraude carísimo: conviene revisar mucho más.
    rng = np.random.default_rng(1)
    y = np.zeros(500, dtype=int)
    y[:20] = 1
    scores = rng.normal(size=500)
    scores[:20] += 1.0
    amounts = np.full(500, 100.0)
    amounts[:20] = 1e7

    barato = cost_optimal_threshold(y, scores, amounts, review_cost=1_000.0)
    assert barato["n_alerts"] > 20


def test_el_reporte_de_umbral_mide_la_fpr_solo_sobre_legitimas():
    # Si el reporte contara el fraude dentro de la FPR, agregar fraude de score altísimo la
    # movería. No debe moverla: la tasa de falsos positivos es sobre transacciones legítimas.
    rng = np.random.default_rng(7)
    train = rng.normal(size=20_000)
    legitimas = rng.normal(size=5_000)

    solo_legitimas = threshold_report(train, legitimas, np.zeros(5_000, dtype=int))

    con_fraude_scores = np.concatenate([legitimas, np.full(500, 50.0)])
    con_fraude_y = np.concatenate([np.zeros(5_000, dtype=int), np.ones(500, dtype=int)])
    con_fraude = threshold_report(train, con_fraude_scores, con_fraude_y)

    np.testing.assert_allclose(
        solo_legitimas["alpha_observado"], con_fraude["alpha_observado"], rtol=1e-9
    )


def test_el_reporte_de_umbral_detecta_el_drift():
    # Escenario del módulo: los datos posteriores están desplazados respecto al
    # entrenamiento, así que el umbral calibrado antes dispara muchas más alertas.
    rng = np.random.default_rng(3)
    train = rng.normal(size=20_000)
    test_desplazado = rng.normal(loc=1.5, size=20_000)

    reporte = threshold_report(train, test_desplazado, np.zeros(20_000, dtype=int))
    assert (reporte["alpha_observado"] > reporte["alpha"] * 3).all()


def test_el_reporte_devuelve_una_fila_por_alpha():
    rng = np.random.default_rng(0)
    reporte = threshold_report(rng.normal(size=1_000), rng.normal(size=500),
                               np.zeros(500, dtype=int), alphas=(0.01, 0.05))

    assert len(reporte) == 2
    assert list(reporte.columns) == [
        "alpha", "threshold", "alpha_observado", "alertas",
        "fraude_detectado", "recall", "precision",
    ]

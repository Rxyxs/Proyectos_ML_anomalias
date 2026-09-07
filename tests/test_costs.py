"""Pruebas unitarias de las métricas en pesos (src.operations.costs).

El punto de estas funciones es que un fraude caro no vale lo mismo que uno barato, así que
las pruebas están construidas para que una métrica por conteo y una por monto den
resultados distintos: si alguien las reemplazara por su versión contadora, fallarían.
"""
import numpy as np
import pytest

from src.operations.costs import (
    amount_concentration,
    break_even_review_cost,
    net_savings,
    savings_curve,
    value_weighted_recall,
)


def test_recall_en_monto_difiere_del_recall_por_conteo():
    # Cuatro fraudes: uno carísimo y tres baratos. Atrapar solo el caro es 25% de recall
    # por conteo pero 97% del dinero — la diferencia que justifica todo el módulo.
    y = np.array([1, 1, 1, 1])
    amounts = np.array([970_000.0, 10_000.0, 10_000.0, 10_000.0])
    flagged = np.array([True, False, False, False])

    assert value_weighted_recall(y, flagged, amounts) == pytest.approx(0.97)


def test_recall_en_monto_ignora_los_legitimos_marcados():
    # Marcar transacciones legítimas cuesta plata en la revisión, pero no cambia cuánto del
    # monto defraudado se recuperó.
    y = np.array([1, 0, 0])
    amounts = np.array([100.0, 1e9, 1e9])

    assert value_weighted_recall(y, np.array([True, True, True]), amounts) == pytest.approx(1.0)
    assert value_weighted_recall(y, np.array([True, False, False]), amounts) == pytest.approx(1.0)


def test_recall_en_monto_es_cero_sin_fraude():
    assert value_weighted_recall(np.zeros(3), np.ones(3, dtype=bool), np.ones(3)) == 0.0


def test_ahorro_neto_cobra_la_revision_de_los_falsos_positivos():
    # Se recupera un fraude de 5.000 y se revisan 3 alertas a 1.000 cada una: 5.000 - 3.000.
    y = np.array([1, 0, 0])
    amounts = np.array([5_000.0, 100.0, 100.0])
    flagged = np.array([True, True, True])

    assert net_savings(y, flagged, amounts, review_cost=1_000.0) == pytest.approx(2_000.0)


def test_ahorro_neto_puede_ser_negativo_si_se_revisa_de_mas():
    y = np.array([1, 0, 0, 0, 0])
    amounts = np.array([1_000.0, 10.0, 10.0, 10.0, 10.0])

    assert net_savings(y, np.ones(5, dtype=bool), amounts, review_cost=1_000.0) < 0


def test_ahorro_neto_aplica_la_tasa_de_recuperacion():
    y = np.array([1])
    amounts = np.array([10_000.0])
    flagged = np.array([True])

    completo = net_savings(y, flagged, amounts, review_cost=0.0, recovery_rate=1.0)
    parcial = net_savings(y, flagged, amounts, review_cost=0.0, recovery_rate=0.5)

    assert completo == pytest.approx(10_000.0)
    assert parcial == pytest.approx(5_000.0)


def test_curva_de_ahorro_alcanza_recall_total_al_revisar_todo():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    scores = rng.normal(size=300) + y
    amounts = rng.lognormal(10, 1, 300)

    curva = savings_curve(y, scores, amounts, review_cost=1.0)

    assert curva["revisadas"].iloc[0] == 1
    assert curva["recall"].iloc[-1] == pytest.approx(1.0)
    assert curva["recall_en_monto"].iloc[-1] == pytest.approx(1.0)


def test_la_curva_de_ahorro_recorre_el_ranking_en_orden():
    # Con un score perfecto, revisar más nunca puede detectar menos fraudes.
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([9.0, 8.0, 1.0, 0.5, 0.2, 0.1])
    amounts = np.full(6, 1_000.0)

    curva = savings_curve(y, scores, amounts, review_cost=1.0, n_points=6)
    assert curva["fraudes_detectados"].is_monotonic_increasing


def test_concentracion_del_monto_detecta_una_cola_pesada():
    # Un fraude de 900 contra nueve de 100/9: el 10% más caro concentra el 90%.
    amounts = np.array([900.0] + [100.0 / 9] * 9)
    y = np.ones(10, dtype=int)

    resultado = amount_concentration(y, amounts, top_fraction=0.1)
    assert resultado["n_fraud"] == 10
    assert resultado["share_of_amount"] == pytest.approx(0.9, abs=0.01)


def test_concentracion_sin_fraude_no_falla():
    resultado = amount_concentration(np.zeros(5), np.ones(5))
    assert resultado["n_fraud"] == 0
    assert resultado["share_of_amount"] == 0.0


def test_rechaza_montos_con_largo_distinto():
    with pytest.raises(ValueError, match="mismo largo"):
        value_weighted_recall(np.array([1, 0]), np.array([True, False]), np.array([1.0]))


def test_el_costo_de_equilibrio_es_la_perdida_esperada_por_transaccion():
    # Un fraude de 10.000 entre 10 transacciones: la pérdida esperada por transacción es
    # 1.000, así que revisar por debajo de eso conviene siempre.
    y = np.array([1] + [0] * 9)
    amounts = np.array([10_000.0] + [100.0] * 9)

    assert break_even_review_cost(y, amounts) == pytest.approx(1_000.0)


def test_por_debajo_del_equilibrio_conviene_revisar_todo():
    # Es la propiedad que vuelve degenerado al óptimo económico: si revisar cuesta menos que
    # la pérdida esperada, marcar el 100% del tráfico gana plata.
    rng = np.random.default_rng(5)
    y = (rng.random(1_000) < 0.05).astype(int)
    amounts = rng.lognormal(10, 0.5, 1_000)

    equilibrio = break_even_review_cost(y, amounts)
    todo = np.ones(1_000, dtype=bool)

    assert net_savings(y, todo, amounts, review_cost=equilibrio * 0.9) > 0
    assert net_savings(y, todo, amounts, review_cost=equilibrio * 1.1) < 0


def test_el_equilibrio_escala_con_la_tasa_de_recuperacion():
    y = np.array([1, 0])
    amounts = np.array([1_000.0, 0.0])

    assert break_even_review_cost(y, amounts, recovery_rate=1.0) == pytest.approx(500.0)
    assert break_even_review_cost(y, amounts, recovery_rate=0.5) == pytest.approx(250.0)

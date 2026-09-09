"""Pruebas del apilado semi-supervisado y el aprendizaje activo (src.adaptive).

El resultado que estos módulos vienen a demostrar es que **cómo se gasta el presupuesto de
etiquetado importa más que su tamaño**. Varias pruebas están construidas para fallar si esa
diferencia desapareciera, y otras cubren el caso degenerado que aparece con presupuestos
chicos: no encontrar un solo positivo y quedarse sin clasificador.
"""
import numpy as np
import pytest

from src.adaptive.active import STRATEGIES, run_active_learning, select_batch
from src.adaptive.stacking import (
    build_training_set,
    fit_classifier,
    select_for_review,
    stack_scores,
)


@pytest.fixture
def pool():
    """Pool con prevalencia baja donde el score no supervisado sí informa."""
    rng = np.random.default_rng(0)
    n = 4_000
    y = (rng.random(n) < 0.02).astype(int)
    X = rng.normal(size=(n, 4)) + y[:, None] * 2.0
    scores = rng.normal(size=n) + y * 3.0
    return X, y, scores


def test_el_apilado_agrega_una_columna_por_detector():
    X = np.zeros((10, 4))
    apilado = stack_scores(X, {"a": np.ones(10), "b": np.zeros(10)})
    assert apilado.shape == (10, 6)


def test_el_apilado_ordena_los_detectores_de_forma_estable():
    # Sin un orden fijo, dos llamadas podrían producir columnas en distinto orden y el
    # clasificador entrenado con una no serviría para puntuar con la otra.
    X = np.zeros((5, 2))
    uno = stack_scores(X, {"zeta": np.ones(5), "alfa": np.full(5, 2.0)})
    otro = stack_scores(X, {"alfa": np.full(5, 2.0), "zeta": np.ones(5)})
    np.testing.assert_array_equal(uno, otro)


def test_el_apilado_sin_detectores_devuelve_las_features_originales():
    X = np.arange(12, dtype=float).reshape(4, 3)
    np.testing.assert_array_equal(stack_scores(X, {}), X)


def test_el_apilado_rechaza_scores_de_otro_largo():
    with pytest.raises(ValueError, match="mismas filas"):
        stack_scores(np.zeros((10, 2)), {"a": np.ones(7)})


def test_etiquetar_la_cola_encuentra_mucho_mas_fraude_que_el_azar(pool):
    _, y, scores = pool
    top = select_for_review(scores, 100, "top")
    azar = select_for_review(scores, 100, "random")

    assert int(y[top].sum()) > 5 * int(y[azar].sum())


def test_la_seleccion_por_score_devuelve_las_mas_altas():
    scores = np.arange(100, dtype=float)
    elegidas = select_for_review(scores, 10, "top")
    np.testing.assert_array_equal(np.sort(elegidas), np.arange(90, 100))


def test_la_seleccion_no_pide_mas_de_lo_que_hay():
    assert len(select_for_review(np.arange(5, dtype=float), 50, "top")) == 5


def test_rechaza_una_estrategia_de_revision_desconocida():
    with pytest.raises(ValueError, match="Estrategia desconocida"):
        select_for_review(np.zeros(10), 5, "corazonada")


def test_el_conjunto_de_entrenamiento_suma_fondo_negativo(pool):
    X, y, scores = pool
    revisadas = select_for_review(scores, 50, "top")
    X_train, y_train = build_training_set(X, y, revisadas, background_size=500)

    assert len(y_train) == 550
    # Las revisadas conservan su etiqueta real; el fondo entra como negativo.
    np.testing.assert_array_equal(y_train[:50], y[revisadas])
    assert y_train[50:].sum() == 0


def test_el_fondo_no_repite_transacciones_ya_revisadas(pool):
    X, y, scores = pool
    revisadas = select_for_review(scores, 100, "top")
    X_train, _ = build_training_set(X, y, revisadas, background_size=300)

    assert len(X_train) == 400, "revisadas y fondo deben ser disjuntos"


def test_sin_positivos_no_devuelve_clasificador():
    # Es el caso degenerado del muestreo al azar con prevalencia muy baja: hay que poder
    # reportarlo, no disimularlo con un modelo entrenado sobre una sola clase.
    X = np.random.default_rng(0).normal(size=(200, 3))
    assert fit_classifier(X, np.zeros(200, dtype=int)) is None


def test_sin_negativos_tampoco_devuelve_clasificador():
    X = np.random.default_rng(0).normal(size=(200, 3))
    assert fit_classifier(X, np.ones(200, dtype=int)) is None


def test_el_arranque_en_frio_usa_el_score_no_supervisado(pool):
    # Sin modelo todavía no hay incertidumbre que medir, así que toda estrategia debe caer al
    # score no supervisado — que es lo que los Módulos 2 a 6 dejan disponible.
    _, _, scores = pool
    vacio = np.array([], dtype=int)

    por_duda = select_batch("uncertainty", 20, scores, vacio, predicted_proba=None)
    por_score = select_batch("top_score", 20, scores, vacio, predicted_proba=None)

    np.testing.assert_array_equal(np.sort(por_duda), np.sort(por_score))


def test_la_incertidumbre_elige_probabilidades_cercanas_a_un_medio(pool):
    _, _, scores = pool
    proba = np.linspace(0.0, 1.0, len(scores))

    elegidas = select_batch("uncertainty", 10, scores, np.array([], dtype=int), proba)
    assert np.abs(proba[elegidas] - 0.5).max() < 0.02


def test_no_vuelve_a_elegir_lo_ya_revisado(pool):
    _, _, scores = pool
    ya = np.arange(100)
    batch = select_batch("top_score", 30, scores, ya)

    assert len(np.intersect1d(batch, ya)) == 0


def test_el_hibrido_mezcla_score_alto_con_incertidumbre(pool):
    _, _, scores = pool
    rng = np.random.default_rng(1)
    proba = rng.random(len(scores))

    batch = select_batch("hybrid", 20, scores, np.array([], dtype=int), proba)
    assert len(batch) == 20
    assert len(np.unique(batch)) == 20, "no debe repetir índices entre las dos mitades"


def test_rechaza_una_estrategia_activa_desconocida(pool):
    _, _, scores = pool
    with pytest.raises(ValueError, match="Estrategia desconocida"):
        select_batch("adivinanza", 5, scores, np.array([], dtype=int), np.zeros(len(scores)))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_el_circuito_activo_acumula_etiquetas_sin_repetir(strategy, pool):
    X, y, scores = pool
    rng = np.random.default_rng(2)
    X_test = rng.normal(size=(400, 4))
    y_test = (rng.random(400) < 0.05).astype(int)

    tabla = run_active_learning(X, y, X_test, y_test, scores, strategy,
                                n_rounds=3, batch_size=20)

    assert list(tabla["etiquetas"]) == [20, 40, 60]
    assert tabla["fraudes_encontrados"].is_monotonic_increasing

"""Pruebas unitarias de la combinación de detectores (src.unsupervised.ensemble)."""
import numpy as np
import pytest

from src.unsupervised.ensemble import (
    ENSEMBLE_STRATEGIES,
    build_ensembles,
    rank_average,
    zscore_average,
    zscore_max,
)


@pytest.fixture
def scores_de_dos_detectores() -> dict[str, np.ndarray]:
    """Dos detectores en escalas muy distintas que coinciden en cuál es la última fila."""
    return {
        "detector_a": np.array([0.1, 0.2, 0.3, 9.0]),
        "detector_b": np.array([1000.0, 2000.0, 3000.0, 90000.0]),
    }


def test_build_ensembles_aplica_las_tres_estrategias(scores_de_dos_detectores):
    combinados = build_ensembles(scores_de_dos_detectores)
    assert set(combinados) == set(ENSEMBLE_STRATEGIES)
    for nombre, scores in combinados.items():
        assert scores.shape == (4,), nombre


@pytest.mark.parametrize("estrategia", [rank_average, zscore_average, zscore_max])
def test_todas_las_estrategias_coinciden_en_la_fila_mas_anomala(estrategia, scores_de_dos_detectores):
    combinado = estrategia(scores_de_dos_detectores)
    assert int(np.argmax(combinado)) == 3


def test_rank_average_queda_acotado_entre_cero_y_uno(scores_de_dos_detectores):
    combinado = rank_average(scores_de_dos_detectores)
    assert (combinado > 0).all() and (combinado <= 1).all()


def test_rank_average_ignora_la_escala_de_cada_detector():
    # El mismo orden con magnitudes distintas debe producir exactamente el mismo resultado:
    # es la propiedad por la que se usa rangos y no los scores crudos.
    orden_a = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([10.0, 20.0, 30.0])}
    orden_b = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([1e6, 2e6, 3e6])}
    np.testing.assert_allclose(rank_average(orden_a), rank_average(orden_b))


def test_rank_average_reparte_el_rango_promedio_entre_empates():
    # Detectores discretos como HBOS producen muchos empates; deben recibir el mismo score
    # combinado en vez de un orden arbitrario según su posición en el arreglo.
    combinado = rank_average({"x": np.array([5.0, 5.0, 1.0, 9.0])})
    assert combinado[0] == combinado[1]


def test_zscore_max_basta_con_un_detector_convencido():
    # La fila 1 es anómala solo para el detector B. El promedio la diluye contra el voto
    # en contra de A; el máximo la conserva.
    scores = {
        "a": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.0, 50.0, 0.0, 0.0, 0.0]),
    }
    assert zscore_max(scores)[1] > zscore_average(scores)[1]


def test_zscore_average_conserva_la_magnitud_relativa():
    # Con un solo detector, estandarizar preserva el orden y las distancias relativas,
    # a diferencia de rank_average que las aplana.
    scores = {"a": np.array([0.0, 1.0, 100.0])}
    combinado = zscore_average(scores)
    assert combinado[2] - combinado[1] > combinado[1] - combinado[0]


def test_rechaza_detectores_con_distinto_numero_de_filas():
    with pytest.raises(ValueError, match="mismas filas"):
        rank_average({"a": np.array([1.0, 2.0]), "b": np.array([1.0, 2.0, 3.0])})


def test_rechaza_un_conjunto_vacio_de_detectores():
    with pytest.raises(ValueError, match="al menos un detector"):
        rank_average({})

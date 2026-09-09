"""Pruebas unitarias de Half-Space Trees (src.adaptive.hs_trees).

La propiedad que distingue a este detector de los otros quince es `update_window`: adaptarse
a un cambio de distribución cuesta una pasada lineal. Varias pruebas apuntan justo ahí, y una
verifica lo que **no** debe cambiar al actualizar — la estructura de los árboles.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.adaptive.hs_trees import HalfSpaceTrees, build_hs_trees
from src.unsupervised.models import anomaly_score


@pytest.fixture
def nube() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(2_000, 5))


def test_expone_la_api_homogenea_del_repositorio(nube):
    detector = build_hs_trees().fit(nube)
    assert hasattr(detector, "fit")
    assert hasattr(detector, "score_samples")
    assert detector.score_samples(nube).shape == (len(nube),)


def test_puntua_mas_alto_un_punto_extremo(nube):
    detector = HalfSpaceTrees().fit(nube)
    scores = anomaly_score(detector, np.array([[0.0] * 5, [15.0] * 5]))

    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_separa_anomalias_de_una_nube_gaussiana(nube):
    rng = np.random.default_rng(1)
    prueba = np.vstack([rng.normal(size=(300, 5)), rng.normal(loc=6.0, size=(30, 5))])
    y = np.r_[np.zeros(300), np.ones(30)]

    detector = HalfSpaceTrees().fit(nube)
    assert roc_auc_score(y, anomaly_score(detector, prueba)) > 0.9


def test_la_estructura_de_los_arboles_no_depende_de_los_datos():
    # Es la idea central del método: los cortes se sortean antes de ver un dato, así que dos
    # ajustes con la misma semilla sobre datos distintos comparten estructura.
    rng = np.random.default_rng(0)
    uno = HalfSpaceTrees(random_state=5).fit(rng.normal(size=(500, 4)))
    otro = HalfSpaceTrees(random_state=5).fit(rng.uniform(-10, 10, size=(500, 4)))

    np.testing.assert_array_equal(uno.split_dim_, otro.split_dim_)
    np.testing.assert_allclose(uno.split_val_, otro.split_val_)


def test_actualizar_la_ventana_conserva_los_arboles(nube):
    detector = HalfSpaceTrees(random_state=3).fit(nube)
    dims_antes = detector.split_dim_.copy()
    vals_antes = detector.split_val_.copy()

    detector.update_window(np.random.default_rng(9).normal(loc=4.0, size=(1_000, 5)))

    np.testing.assert_array_equal(detector.split_dim_, dims_antes)
    np.testing.assert_allclose(detector.split_val_, vals_antes)


def test_actualizar_la_ventana_cambia_el_perfil_de_masa(nube):
    detector = HalfSpaceTrees(random_state=3).fit(nube)
    masa_antes = detector.mass_.copy()

    detector.update_window(np.random.default_rng(9).normal(loc=4.0, size=(1_000, 5)))

    assert not np.allclose(detector.mass_, masa_antes)


def test_actualizar_la_ventana_recupera_el_rendimiento_tras_un_corrimiento():
    # El escenario que motiva el módulo: el tráfico se corre y el detector ajustado una vez
    # se degrada. Refrescar la ventana con datos recientes, sin etiquetas, lo recupera.
    rng = np.random.default_rng(7)
    original = rng.normal(size=(3_000, 5))
    corrido = rng.normal(loc=4.0, size=(3_000, 5))

    prueba = np.vstack([rng.normal(loc=4.0, size=(400, 5)), rng.normal(loc=10.0, size=(40, 5))])
    y = np.r_[np.zeros(400), np.ones(40)]

    detector = HalfSpaceTrees(random_state=1).fit(original)
    antes = roc_auc_score(y, anomaly_score(detector, prueba))

    detector.update_window(corrido)
    despues = roc_auc_score(y, anomaly_score(detector, prueba))

    assert despues >= antes


def test_no_permite_actualizar_sin_ajuste_previo():
    with pytest.raises(ValueError, match="antes de actualizar"):
        HalfSpaceTrees().update_window(np.zeros((10, 3)))


def test_es_reproducible_con_la_misma_semilla(nube):
    punto = np.array([[8.0] * 5])
    primero = anomaly_score(HalfSpaceTrees(random_state=11).fit(nube), punto)
    segundo = anomaly_score(HalfSpaceTrees(random_state=11).fit(nube), punto)
    np.testing.assert_allclose(primero, segundo)


def test_mas_arboles_dan_un_score_mas_estable(nube):
    # Un solo árbol es un detector pésimo por construcción; la calidad sale del ensemble.
    rng = np.random.default_rng(4)
    prueba = np.vstack([rng.normal(size=(200, 5)), rng.normal(loc=6.0, size=(20, 5))])
    y = np.r_[np.zeros(200), np.ones(20)]

    pocos = [roc_auc_score(y, anomaly_score(HalfSpaceTrees(n_trees=1, random_state=s).fit(nube), prueba))
             for s in range(6)]
    muchos = [roc_auc_score(y, anomaly_score(HalfSpaceTrees(n_trees=25, random_state=s).fit(nube), prueba))
              for s in range(6)]

    assert np.std(muchos) < np.std(pocos)


def test_el_limite_de_tamano_corta_el_descenso(nube):
    # Con un límite alto la búsqueda se detiene antes, así que los scores no pueden coincidir
    # con los de un límite que deja bajar hasta la hoja.
    punto = np.array([[3.0] * 5])
    corto = anomaly_score(HalfSpaceTrees(size_limit=500, random_state=2).fit(nube), punto)
    largo = anomaly_score(HalfSpaceTrees(size_limit=0, random_state=2).fit(nube), punto)

    assert not np.allclose(corto, largo)


def test_tolera_una_columna_constante():
    # Un rango cero dividiría por cero al normalizar sin el epsilon de la implementación.
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=500), np.ones(500)])

    scores = anomaly_score(HalfSpaceTrees().fit(X), X[:10])
    assert np.isfinite(scores).all()


def _datos_con_ruido(n_ruido: int, semilla: int = 0):
    """Señal concentrada en 2 features, más `n_ruido` columnas irrelevantes.

    La señal se genera siempre con el mismo generador, así que es idéntica entre llamadas; el
    ruido sale de uno aparte. Sin esa separación, consumir el generador para el ruido movería
    también la señal y la comparación mediría dos cambios a la vez.
    """
    señal = np.random.default_rng(semilla)
    entrenamiento = señal.normal(size=(4_000, 2))
    prueba = np.vstack([señal.normal(size=(400, 2)), señal.normal(loc=5.0, size=(40, 2))])
    y = np.r_[np.zeros(400), np.ones(40)]

    if n_ruido == 0:
        return entrenamiento, prueba, y

    ruido = np.random.default_rng(semilla + 1_000)
    return (
        np.column_stack([entrenamiento, ruido.normal(size=(4_000, n_ruido))]),
        np.column_stack([prueba, ruido.normal(size=(440, n_ruido))]),
        y,
    )


def test_la_estructura_aleatoria_se_diluye_con_features_irrelevantes():
    """Explica el flojo desempeño sobre PaySim, y de paso el de LODA en el Módulo 6.

    Ambos construyen su estructura sin mirar los datos, así que reparten su capacidad entre
    todas las dimensiones por igual. Si la señal vive en unas pocas features —como en PaySim,
    donde está en `errorBalanceOrig` y `errorBalanceDest`— agregar columnas irrelevantes la
    diluye. Un detector que estima la densidad a partir de los datos no sufre eso.
    """
    from src.unsupervised.families import GMMDensity

    limpio_tr, limpio_te, y = _datos_con_ruido(0)
    ruidoso_tr, ruidoso_te, _ = _datos_con_ruido(30)

    caida_hst = (
        roc_auc_score(y, anomaly_score(HalfSpaceTrees(random_state=1).fit(limpio_tr), limpio_te))
        - roc_auc_score(y, anomaly_score(HalfSpaceTrees(random_state=1).fit(ruidoso_tr), ruidoso_te))
    )
    caida_gmm = (
        roc_auc_score(y, anomaly_score(GMMDensity().fit(limpio_tr), limpio_te))
        - roc_auc_score(y, anomaly_score(GMMDensity().fit(ruidoso_tr), ruidoso_te))
    )

    assert caida_hst > 0.05, "la estructura aleatoria debe degradarse con dimensiones irrelevantes"
    assert caida_gmm < 0.01, "el modelo de densidad debe aguantar el mismo ruido"

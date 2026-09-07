"""Pruebas unitarias de las familias complementarias de detectores (src.unsupervised.families).

Cada prueba comprueba la propiedad que justifica incluir esa familia en el benchmark, no
solo que el código corra: al detector de covarianza se le exige detectar una correlación
rota, al de reconstrucción un punto fuera del subespacio, etc. Todo con datos sintéticos,
sin depender de la descarga de PaySim.
"""
import numpy as np
import pytest

from src.unsupervised.families import (
    ECOD,
    HBOS,
    LODA,
    FastABOD,
    GMMDensity,
    KNNDistance,
    OneClassSVMApprox,
    PCAReconstruction,
    RobustMahalanobis,
    build_detectors,
)
from src.unsupervised.models import anomaly_score


@pytest.fixture
def normal_data() -> np.ndarray:
    """Nube gaussiana estándar de 600 filas y 4 columnas."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(600, 4))


def test_build_detectors_expone_la_api_homogenea():
    detectores = build_detectors()
    assert len(detectores) == 9
    for nombre, detector in detectores.items():
        assert hasattr(detector, "fit"), nombre
        assert hasattr(detector, "score_samples"), nombre


@pytest.mark.parametrize("factory", [HBOS, ECOD, KNNDistance, RobustMahalanobis, GMMDensity,
                                    OneClassSVMApprox, LODA, FastABOD])
def test_detectores_puntuan_mas_alto_un_punto_extremo(factory, normal_data):
    detector = factory().fit(normal_data)
    scores = anomaly_score(detector, np.array([[0.0, 0.0, 0.0, 0.0], [30.0, 30.0, 30.0, 30.0]]))

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


@pytest.mark.parametrize("factory", [HBOS, ECOD, KNNDistance, RobustMahalanobis, GMMDensity,
                                    OneClassSVMApprox, LODA, FastABOD])
def test_detectores_devuelven_un_score_por_fila(factory, normal_data):
    detector = factory().fit(normal_data)
    assert anomaly_score(detector, normal_data).shape == (len(normal_data),)


def test_pca_detecta_un_punto_fuera_del_subespacio_principal():
    # Datos que viven en un plano de 2D embebido en 3D: la tercera coordenada es una
    # combinación lineal de las dos primeras, así que PCA la reconstruye sin error.
    rng = np.random.default_rng(0)
    base = rng.normal(size=(400, 2))
    X = np.column_stack([base, base[:, 0] + base[:, 1]])

    detector = PCAReconstruction(n_components=2).fit(X)
    en_el_plano = np.array([[1.0, 1.0, 2.0]])
    fuera_del_plano = np.array([[1.0, 1.0, 9.0]])

    assert anomaly_score(detector, fuera_del_plano)[0] > anomaly_score(detector, en_el_plano)[0]


def test_mahalanobis_detecta_una_correlacion_rota_que_el_z_score_por_columna_no_ve():
    # Dos features fuertemente correlacionadas. El punto (2.5, -2.5) tiene ambos valores
    # dentro del rango observado por separado, pero su combinación no ocurre nunca: eso es
    # exactamente lo que un baseline por columna (MAD-z) no puede detectar.
    rng = np.random.default_rng(7)
    x = rng.normal(size=800)
    X = np.column_stack([x, x + rng.normal(scale=0.05, size=800)])

    detector = RobustMahalanobis().fit(X)
    coherente = np.array([[2.5, 2.5]])
    incoherente = np.array([[2.5, -2.5]])

    assert anomaly_score(detector, incoherente)[0] > anomaly_score(detector, coherente)[0]


def test_hbos_es_determinista_y_no_tiene_estado_aleatorio(normal_data):
    primero = anomaly_score(HBOS().fit(normal_data), normal_data)
    segundo = anomaly_score(HBOS().fit(normal_data), normal_data)
    np.testing.assert_allclose(primero, segundo)


def test_hbos_asigna_score_finito_a_valores_fuera_del_rango_de_entrenamiento(normal_data):
    # np.digitize manda los valores fuera de rango al bin extremo; sin el clip, el índice
    # se saldría del arreglo de densidades.
    detector = HBOS().fit(normal_data)
    scores = anomaly_score(detector, np.array([[-500.0, -500.0, 500.0, 500.0]]))
    assert np.isfinite(scores).all()


def test_ecod_produce_probabilidades_de_cola_estrictamente_entre_cero_y_uno(normal_data):
    detector = ECOD().fit(normal_data)
    izquierda, derecha = detector._tail_probabilities(np.array([[-1e9, 0.0, 0.0, 1e9]]))

    for cola in (izquierda, derecha):
        assert (cola > 0).all(), "una probabilidad de cola en 0 haría infinito el log del score"
        assert (cola <= 1).all()


def test_ecod_no_necesita_hiperparametros_y_detecta_ambas_colas(normal_data):
    detector = ECOD().fit(normal_data)
    scores = anomaly_score(detector, np.array([
        [0.0, 0.0, 0.0, 0.0],      # centro de la distribución
        [-40.0, -40.0, -40.0, -40.0],  # cola izquierda
        [40.0, 40.0, 40.0, 40.0],      # cola derecha
    ]))

    assert scores[1] > scores[0]
    assert scores[2] > scores[0]


def test_knn_distance_usa_la_distancia_al_k_esimo_vecino():
    # Un cúmulo denso en el origen: un punto lejano debe quedar por encima de cualquiera
    # de los puntos del cúmulo.
    rng = np.random.default_rng(3)
    X = rng.normal(scale=0.1, size=(300, 2))

    detector = KNNDistance(n_neighbors=5).fit(X)
    scores = anomaly_score(detector, np.vstack([X[:10], [[10.0, 10.0]]]))

    assert scores[-1] > scores[:-1].max()


def test_gmm_detecta_el_hueco_entre_dos_modos():
    # Densidad bimodal: el punto intermedio es "normal" en cada coordenada pero cae en una
    # zona de probabilidad casi nula, algo que un detector de distancia global no ve.
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(loc=-5.0, scale=0.3, size=(300, 2)),
                   rng.normal(loc=5.0, scale=0.3, size=(300, 2))])

    detector = GMMDensity(n_components=2).fit(X)
    scores = anomaly_score(detector, np.array([[-5.0, -5.0], [0.0, 0.0]]))

    assert scores[1] > scores[0]


def test_one_class_svm_aproximado_es_reproducible(normal_data):
    punto = np.array([[15.0, 15.0, 15.0, 15.0]])
    primero = anomaly_score(OneClassSVMApprox(random_state=1).fit(normal_data), punto)
    segundo = anomaly_score(OneClassSVMApprox(random_state=1).fit(normal_data), punto)
    np.testing.assert_allclose(primero, segundo)


def test_loda_captura_dependencias_que_hbos_no_ve():
    # HBOS arma histogramas sobre las features originales, así que asume independencia y no
    # puede ver una correlación rota. LODA los arma sobre proyecciones aleatorias y sí.
    rng = np.random.default_rng(7)
    base = rng.normal(size=3000)
    X = np.column_stack([base, base + rng.normal(scale=0.05, size=3000)])

    coherente = np.array([[2.0, 2.0]])
    incoherente = np.array([[2.0, -2.0]])

    loda = LODA(n_projections=200).fit(X)
    assert anomaly_score(loda, incoherente)[0] > anomaly_score(loda, coherente)[0]


def test_loda_es_reproducible_con_la_misma_semilla(normal_data):
    punto = np.array([[12.0, 12.0, 12.0, 12.0]])
    primero = anomaly_score(LODA(random_state=3).fit(normal_data), punto)
    segundo = anomaly_score(LODA(random_state=3).fit(normal_data), punto)
    np.testing.assert_allclose(primero, segundo)


def test_las_proyecciones_de_loda_son_dispersas(normal_data):
    # El paper usa ~sqrt(d) entradas no nulas por proyección; con 4 features son 2.
    loda = LODA(n_projections=50).fit(normal_data)
    no_nulas = (loda.projections_ != 0).sum(axis=1)
    np.testing.assert_array_equal(no_nulas, 2)


def test_la_atribucion_de_loda_senala_la_feature_responsable():
    # Solo la tercera columna es anómala en el punto consultado; su aporte debe dominar.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 5))
    punto = np.zeros((1, 5))
    punto[0, 2] = 12.0

    importancia = LODA(n_projections=300, random_state=0).fit(X).feature_importance(punto)

    assert importancia.shape == (1, 5)
    assert int(np.argmax(importancia[0])) == 2


def test_abod_devuelve_varianza_baja_para_un_punto_al_borde():
    # Un punto lejano ve toda la nube en la misma dirección: la varianza de sus ángulos
    # colapsa. score_samples devuelve esa varianza, y más baja significa más anómalo.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 3))

    abod = FastABOD(n_neighbors=15).fit(X)
    scores = abod.score_samples(np.vstack([np.zeros((1, 3)), np.full((1, 3), 30.0)]))

    assert scores[1] < scores[0]
    assert anomaly_score(abod, np.full((1, 3), 30.0))[0] > anomaly_score(abod, np.zeros((1, 3)))[0]


def test_abod_procesa_por_bloques_sin_cambiar_el_resultado(normal_data):
    # El troceado existe por memoria, no por semántica: partir en bloques no puede alterar
    # el score de ninguna fila.
    entero = FastABOD(n_neighbors=10, chunk_size=10_000).fit(normal_data).score_samples(normal_data)
    troceado = FastABOD(n_neighbors=10, chunk_size=37).fit(normal_data).score_samples(normal_data)
    np.testing.assert_allclose(entero, troceado, rtol=1e-10)


def test_abod_tolera_un_punto_identico_a_uno_de_entrenamiento(normal_data):
    # Un vecino a distancia cero divide por cero sin el epsilon de la implementación.
    abod = FastABOD(n_neighbors=10).fit(normal_data)
    scores = abod.score_samples(normal_data[:5])
    assert np.isfinite(scores).all()

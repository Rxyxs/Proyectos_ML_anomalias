"""Contrato común de los modelos de detección de anomalías (Día 15, endurecido Día 16).

Tres propiedades que todo detector del repositorio (los de src.unsupervised.models
y los de src.unsupervised.families) debería cumplir para ser seguro de enchufar en
src.serving.predict, verificadas explícitamente en vez de asumidas:

1. score_samples devuelve un score finito por fila, con la forma y el dtype que el
   resto del pipeline espera (un vector 1-D de floats, largo == n_filas).
2. Bajo desbalance extremo (99% inliers / 1% outliers), la fracción de alertas que
   produce IsolationForest.predict() se mantiene cerca del contamination con el que
   se ajustó -- no se dispara ni colapsa a cero solo porque el set de prueba está
   desbalanceado.
3. Ante NaN/Inf en la entrada, TODOS los detectores rechazan con ValueError. El Día
   15 encontró que HBOS, ECOD, LODA y MADBaseline no fallaban solos -- devolvían un
   score no finito o silenciosamente "razonable" sin avisar. El Día 16 les agregó
   assert_finite() (src.unsupervised.models) a los cuatro. El único que sigue sin
   poder validarse a sí mismo es IsolationForest (es código de scikit-learn, no
   propio); por eso se prueba tanto que llamado directo sigue sin validar como que,
   a través de DetectorPackage -- el único camino real hacia producción -- sí rechaza,
   porque la validación vive ahí, no en cada detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.serving.package import build_package
from src.serving.predict import score_transactions
from src.unsupervised.families import build_detectors
from src.unsupervised.models import (
    MADBaseline,
    anomaly_score,
    build_isolation_forest,
    build_lof,
    build_mad_baseline,
)


@pytest.fixture
def normal_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(500, 4))


def _model_builders():
    """(nombre, factory sin argumentos) para cada detector "nombrado" de models.py."""
    return [
        ("isolation_forest", build_isolation_forest),
        ("lof", build_lof),
        ("mad_baseline", build_mad_baseline),
    ]


# --------------------------------------------------------------------- 1. forma/dtype

@pytest.mark.parametrize("nombre,factory", _model_builders())
def test_score_samples_devuelve_un_vector_1d_de_floats_finitos(nombre, factory, normal_data):
    detector = factory().fit(normal_data)
    scores = anomaly_score(detector, normal_data)

    assert scores.ndim == 1, nombre
    assert scores.shape[0] == normal_data.shape[0], nombre
    assert np.issubdtype(scores.dtype, np.floating), nombre
    assert np.isfinite(scores).all(), nombre


def test_los_nueve_detectores_de_families_devuelven_un_score_por_fila(normal_data):
    detectores = build_detectors()
    assert len(detectores) == 9

    for nombre, detector in detectores.items():
        detector.fit(normal_data)
        scores = anomaly_score(detector, normal_data)
        assert scores.shape == (normal_data.shape[0],), nombre
        assert np.isfinite(scores).all(), nombre


@pytest.mark.parametrize("nombre,factory", _model_builders())
def test_score_samples_sobre_una_sola_fila_no_pierde_la_dimension(nombre, factory, normal_data):
    """Un lote de una transaccion (el caso de score_transactions en produccion) no
    debe degradar a un escalar: sigue siendo un vector de largo 1."""
    detector = factory().fit(normal_data)
    una_fila = normal_data[:1]
    scores = anomaly_score(detector, una_fila)
    assert scores.shape == (1,), nombre


# ------------------------------------------------------ 2. desbalance extremo 99/1

def test_isolation_forest_mantiene_la_proporcion_de_alertas_bajo_desbalance_extremo():
    """99% inliers / 1% outliers en el set de prueba: la fraccion marcada por
    predict() debe seguir cerca del contamination con que se ajusto el detector,
    no dispararse ni colapsar a cero solo por el desbalance del set de prueba."""
    rng = np.random.default_rng(7)
    contamination = 0.01

    X_train = rng.normal(size=(2000, 4))
    detector = build_isolation_forest(contamination=contamination).fit(X_train)

    n_in, n_out = 990, 10
    X_in = rng.normal(size=(n_in, 4))
    X_out = rng.normal(loc=15.0, size=(n_out, 4))
    X_test = np.vstack([X_in, X_out])
    y_test = np.array([0] * n_in + [1] * n_out)

    preds = detector.predict(X_test)
    flagged = preds == -1

    assert flagged.sum() + (~flagged).sum() == len(X_test)
    # Tolerancia amplia (2x): esto no es un test de calibracion exacta del
    # contamination, solo comprueba que el desbalance del set de prueba no
    # descalibra la proporcion de alertas por si solo.
    assert 0.0 < flagged.mean() < 4 * contamination

    # Los outliers extremos inyectados tienen que quedar dentro de lo marcado --
    # si esto fallara, el "mantenimiento de proporciones" de arriba seria vacuo.
    assert flagged[y_test == 1].mean() == 1.0


def test_mad_baseline_separa_outliers_extremos_bajo_desbalance_99_1():
    """Mismo desbalance 99/1, pero contra el baseline sin hiperparametro de
    contamination: los outliers extremos deben quedar entre los peores scores,
    proporcionalmente a su 1% real en el set de prueba."""
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(1000, 4))
    detector = build_mad_baseline().fit(X_train)

    n_in, n_out = 990, 10
    X_in = rng.normal(size=(n_in, 4))
    X_out = rng.normal(loc=20.0, size=(n_out, 4))
    X_test = np.vstack([X_in, X_out])
    y_test = np.array([0] * n_in + [1] * n_out)

    scores = anomaly_score(detector, X_test)
    top_1_pct = np.argsort(scores)[-n_out:]
    recall_top_1_pct = y_test[top_1_pct].mean()

    assert recall_top_1_pct == 1.0


# ------------------------------------------------------------- 3. NaN / Inf de entrada

def _con_nan_e_inf(X: np.ndarray) -> np.ndarray:
    X_bad = X[:3].copy()
    X_bad[0, 0] = np.nan
    X_bad[1, 1] = np.inf
    X_bad[2, 2] = -np.inf
    return X_bad


# Los 9 detectores de families.py: 6 ya rechazaban por apoyarse en scikit-learn
# (GMM, kNN, MCD, Nystroem, PCA), y desde el Día 16 los 3 hechos a mano (HBOS,
# ECOD, LODA) tambien -- vía assert_finite() en su propio score_samples.
_TODOS_LOS_DETECTORES_DE_FAMILIES = [
    "gmm_density", "knn_distance", "robust_mahalanobis", "ocsvm_nystroem", "abod",
    "pca_reconstruction", "hbos", "ecod", "loda",
]


def test_los_nueve_detectores_de_families_rechazan_nan_e_inf(normal_data):
    detectores = build_detectors()
    assert set(detectores) == set(_TODOS_LOS_DETECTORES_DE_FAMILIES)

    X_bad = _con_nan_e_inf(normal_data)
    for nombre in _TODOS_LOS_DETECTORES_DE_FAMILIES:
        detector = detectores[nombre].fit(normal_data)
        with pytest.raises(ValueError):
            anomaly_score(detector, X_bad)


def test_mad_baseline_rechaza_nan_e_inf(normal_data):
    """Antes del Día 16, el baseline hecho a mano propagaba NaN/Inf en silencio --
    era el gap mas claro del audit del Día 15, porque es el unico detector sin
    ninguna dependencia de scikit-learn que le regale la validacion gratis. Ahora
    score_samples llama assert_finite() igual que HBOS/ECOD/LODA."""
    detector = MADBaseline().fit(normal_data)
    X_bad = _con_nan_e_inf(normal_data)
    with pytest.raises(ValueError, match="NaN o valores infinitos"):
        detector.score_samples(X_bad)


def test_lof_rechaza_nan(normal_data):
    detector = build_lof().fit(normal_data)
    X_bad = _con_nan_e_inf(normal_data)
    with pytest.raises(ValueError):
        anomaly_score(detector, X_bad)


def test_isolation_forest_llamado_directo_sigue_sin_validar(normal_data):
    """IsolationForest es de scikit-learn: no se puede tocar su código para que
    valide. Documenta el limite exacto de lo que el Día 16 puede arreglar por su
    cuenta -- la próxima prueba confirma que la capa de serving, que sí es código
    propio, lo cubre igual."""
    detector = build_isolation_forest().fit(normal_data)
    X_bad = _con_nan_e_inf(normal_data)
    scores = anomaly_score(detector, X_bad)
    assert scores.shape == (3,)


def test_isolation_forest_a_traves_del_paquete_de_serving_si_rechaza_nan(normal_data):
    """El mismo IsolationForest que no valida solo, sí rechaza NaN/Inf cuando se
    puntúa a través de DetectorPackage -- src.serving.package.to_scaled_matrix
    llama assert_finite() antes de entregarle nada al detector, así que no importa
    cuál de los detectores del repositorio esté empaquetado."""
    scaler = RobustScaler().fit(normal_data)
    detector = build_isolation_forest().fit(scaler.transform(normal_data))
    calibracion = scaler.transform(normal_data)
    columnas = [f"f{i}" for i in range(normal_data.shape[1])]

    package = build_package(
        detector, scaler, calibracion, columnas,
        detector_name="isolation_forest",
    )

    X_bad = pd.DataFrame(_con_nan_e_inf(normal_data), columns=columnas)
    with pytest.raises(ValueError, match="NaN o valores infinitos"):
        package.score(X_bad)

    with pytest.raises(ValueError, match="NaN o valores infinitos"):
        score_transactions(package, X_bad, explain=False)

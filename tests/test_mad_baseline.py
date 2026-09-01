"""Pruebas unitarias para el baseline estadístico MAD-z (src.unsupervised.models)."""
import numpy as np

from src.unsupervised.models import MADBaseline, anomaly_score, build_mad_baseline


def test_build_mad_baseline_returns_mad_baseline_instance():
    model = build_mad_baseline()
    assert isinstance(model, MADBaseline)


def test_mad_baseline_fit_computes_median_and_mad():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    model = MADBaseline().fit(X)

    assert model.median_ is not None
    np.testing.assert_allclose(model.median_, [3.0, 30.0])
    assert model.mad_ is not None
    assert (model.mad_ > 0).all()


def test_mad_baseline_flags_extreme_point_as_more_anomalous():
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=1.0, size=(200, 3))
    model = MADBaseline().fit(normal)

    typical_point = np.zeros((1, 3))
    extreme_point = np.full((1, 3), 50.0)

    typical_score = anomaly_score(model, typical_point)[0]
    extreme_score = anomaly_score(model, extreme_point)[0]

    assert extreme_score > typical_score


def test_anomaly_score_higher_is_more_anomalous_convention():
    # score_samples del MADBaseline es -max|z|; anomaly_score invierte el signo,
    # así que debe quedar en la misma convención que Isolation Forest/LOF.
    X_train = np.array([[0.0], [1.0], [-1.0], [0.5], [-0.5]])
    model = MADBaseline().fit(X_train)
    scores = anomaly_score(model, np.array([[0.0], [100.0]]))
    assert scores[1] > scores[0]

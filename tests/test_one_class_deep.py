"""Pruebas unitarias de los modelos profundos de una clase (src.deep.one_class).

Cada prueba verifica la propiedad de diseño que hace válido al modelo, no solo que corra:
que Deep SVDD no colapse a una solución trivial, que el VAE penalice el término latente, y
que ambos respeten la convención de score del resto del repositorio.
"""
import numpy as np
import pytest
import torch

from src.deep.one_class import (
    VAE,
    DeepSVDDDetector,
    DeepSVDDNet,
    VAEDetector,
    build_deep_detectors,
)
from src.unsupervised.models import anomaly_score


@pytest.fixture
def normal_data() -> np.ndarray:
    """Nube gaussiana estándar de 400 filas y 5 columnas."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(400, 5)).astype("float32")


@pytest.fixture
def extremos() -> np.ndarray:
    """Un punto en el centro de la distribución y otro claramente fuera."""
    return np.array([[0.0] * 5, [20.0] * 5], dtype="float32")


def test_build_deep_detectors_expone_la_api_homogenea():
    detectores = build_deep_detectors()
    assert set(detectores) == {"vae", "deep_svdd"}
    for nombre, detector in detectores.items():
        assert hasattr(detector, "fit"), nombre
        assert hasattr(detector, "score_samples"), nombre


@pytest.mark.parametrize("factory", [VAEDetector, DeepSVDDDetector])
def test_detectores_puntuan_mas_alto_un_punto_extremo(factory, normal_data, extremos):
    detector = factory(epochs=5).fit(normal_data)
    scores = anomaly_score(detector, extremos)

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


@pytest.mark.parametrize("factory", [VAEDetector, DeepSVDDDetector])
def test_detectores_devuelven_un_score_por_fila(factory, normal_data):
    detector = factory(epochs=3).fit(normal_data)
    assert anomaly_score(detector, normal_data).shape == (len(normal_data),)


@pytest.mark.parametrize("factory", [VAEDetector, DeepSVDDDetector])
def test_detectores_son_reproducibles_con_la_misma_semilla(factory, normal_data, extremos):
    primero = anomaly_score(factory(epochs=3, random_state=7).fit(normal_data), extremos)
    segundo = anomaly_score(factory(epochs=3, random_state=7).fit(normal_data), extremos)
    np.testing.assert_allclose(primero, segundo)


@pytest.mark.parametrize("factory", [VAEDetector, DeepSVDDDetector])
def test_rechaza_una_activacion_desconocida(factory, normal_data):
    with pytest.raises(ValueError, match="Activación desconocida"):
        factory(activation="no_existe", epochs=1).fit(normal_data)


def test_deep_svdd_no_tiene_sesgos_para_evitar_el_colapso():
    # Con sesgos, la red podría mapear toda entrada al centro y ganar el objetivo sin
    # aprender nada. La ausencia de bias es lo que vuelve inalcanzable esa solución.
    red = DeepSVDDNet(n_features=5)
    capas = [m for m in red.net if isinstance(m, torch.nn.Linear)]

    assert capas, "la red debe tener capas lineales"
    for capa in capas:
        assert capa.bias is None


def test_deep_svdd_no_colapsa_a_scores_identicos(normal_data):
    # Un colapso de la hiperesfera se manifiesta como todos los puntos a la misma distancia
    # del centro: el detector deja de ordenar y el score pierde todo poder discriminante.
    detector = DeepSVDDDetector(epochs=10).fit(normal_data)
    scores = anomaly_score(detector, normal_data)
    assert scores.std() > 1e-6


def test_deep_svdd_aleja_el_centro_del_origen(normal_data):
    # Componentes del centro en ~0 permiten a la red anularlas con pesos pequeños; el
    # detector las empuja fuera de esa vecindad al inicializar.
    detector = DeepSVDDDetector(eps=0.1, epochs=3).fit(normal_data)
    assert (detector.center_.abs() >= 0.1 - 1e-6).all()


def test_vae_produce_media_y_logvarianza_separadas():
    modelo = VAE(n_features=5, latent=3)
    mu, logvar = modelo.encode(torch.zeros(7, 5))
    assert mu.shape == (7, 3)
    assert logvar.shape == (7, 3)


def test_vae_reparametriza_de_forma_estocastica():
    # Con la misma media y log-varianza, dos llamadas deben dar muestras distintas: si no,
    # el término KL no estaría regularizando nada.
    torch.manual_seed(0)
    modelo = VAE(n_features=4, latent=2)
    mu, logvar = torch.zeros(10, 2), torch.zeros(10, 2)
    assert not torch.allclose(modelo.reparameterize(mu, logvar), modelo.reparameterize(mu, logvar))


def test_vae_con_beta_cero_ignora_el_termino_latente(normal_data, extremos):
    # beta=0 elimina la divergencia KL del score, dejando solo reconstrucción. Debe seguir
    # produciendo scores válidos, y distintos de los del ELBO completo.
    sin_kl = anomaly_score(VAEDetector(beta=0.0, epochs=5).fit(normal_data), extremos)
    con_kl = anomaly_score(VAEDetector(beta=1.0, epochs=5).fit(normal_data), extremos)

    assert np.isfinite(sin_kl).all()
    assert not np.allclose(sin_kl, con_kl)


def test_vae_promedia_varios_muestreos_latentes(normal_data, extremos):
    # El VAE es estocástico: promediar muestreos debe reducir la varianza del score entre
    # llamadas frente a un único muestreo.
    detector = VAEDetector(epochs=5, n_samples=16).fit(normal_data)
    primero = detector.score_samples(extremos)
    segundo = detector.score_samples(extremos)
    np.testing.assert_allclose(primero, segundo)

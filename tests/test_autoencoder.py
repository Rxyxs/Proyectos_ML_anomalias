"""Pruebas unitarias para el autoencoder PyTorch (src.unsupervised.autoencoder).

Entrenamiento con datos sintéticos pequeños y pocas épocas: valida forma/comportamiento,
no calidad de convergencia (eso se reporta en el pipeline completo, no en CI rápida).
"""
import numpy as np
import pytest

from src.unsupervised.autoencoder import Autoencoder, reconstruction_error, train_autoencoder


@pytest.mark.parametrize("activation", ["relu", "gelu", "swish"])
def test_autoencoder_forward_preserves_shape(activation):
    model = Autoencoder(n_features=6, activation=activation)
    x = __import__("torch").rand(4, 6)
    out = model(x)
    assert out.shape == x.shape


def test_autoencoder_rejects_unknown_activation():
    with pytest.raises(ValueError):
        Autoencoder(n_features=6, activation="tanh-unsupported")


def test_train_autoencoder_returns_fitted_model():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(64, 5)).astype("float32")
    model = train_autoencoder(X_train, activation="relu", epochs=2, batch_size=16)
    assert isinstance(model, Autoencoder)


def test_reconstruction_error_flags_outlier_higher_than_inlier():
    rng = np.random.default_rng(1)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(128, 4)).astype("float32")
    model = train_autoencoder(X_train, activation="relu", epochs=15, batch_size=32)

    inlier = X_train[:5]
    outlier = np.full((5, 4), 50.0, dtype="float32")

    inlier_error = reconstruction_error(model, inlier).mean()
    outlier_error = reconstruction_error(model, outlier).mean()

    assert outlier_error > inlier_error

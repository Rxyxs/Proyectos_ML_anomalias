"""Módulo 4 — modelos profundos de una clase, entrenados solo con transacciones normales.

El benchmark del Módulo 3 dejó dos preguntas abiertas sobre el enfoque profundo:

1. **El autoencoder apenas superó a su ablación lineal** (PR-AUC 0.581 contra 0.487 de PCA).
   ¿El problema es la profundidad, o el *objetivo*? Un autoencoder minimiza error de
   reconstrucción, que no es lo mismo que separar lo anómalo: una red con capacidad
   suficiente aprende a reconstruir bien también las anomalías. **Deep SVDD** ataca eso
   optimizando directamente el objetivo de una clase — comprimir lo normal en una esfera.

2. **El mejor detector fue Gaussian Mixture**, un modelo de densidad. Su contraparte
   profunda no es el autoencoder sino el **VAE**: en vez de un único punto reconstruido,
   estima una distribución latente y puntúa con la verosimilitud (ELBO), lo que le permite
   penalizar tanto una reconstrucción mala como un código latente improbable.

Ambos exponen la convención `fit` / `score_samples` (más bajo = más anómalo) del resto del
repositorio, así que entran directo en el benchmark del Módulo 3 y son comparables fila a
fila con los otros once detectores.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.unsupervised.autoencoder import ACTIVATIONS


# Rango admitido para la log-varianza: exp(±10) se mantiene lejos del desborde en float32.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0

# Norma máxima del gradiente. Las colas de montos y saldos producen lotes con pérdidas
# órdenes de magnitud mayores que el resto; sin recorte, un solo lote destruye los pesos.
MAX_GRAD_NORM = 5.0


def _as_tensor(X) -> torch.Tensor:
    return torch.tensor(np.asarray(X, dtype=np.float32))


def _batches(X: torch.Tensor, batch_size: int, shuffle: bool = True):
    dataset = torch.utils.data.TensorDataset(X)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class VAE(nn.Module):
    """Autoencoder variacional: el encoder produce media y log-varianza de la latente."""

    def __init__(self, n_features: int, latent: int = 4, hidden: int | None = None, activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activación desconocida: {activation}. Usar una de {list(ACTIVATIONS)}")
        act_cls = ACTIVATIONS[activation]
        hidden = hidden or max(latent * 2, n_features // 2)

        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), act_cls())
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            act_cls(),
            nn.Linear(hidden, n_features),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        # La red predice log-varianza en vez de varianza para que la salida pueda ser
        # negativa sin restricciones y exp() garantice una varianza positiva. El clamp
        # acota ese exponente: sobre PaySim escalado las features llegan a |x|~1900, y sin
        # el límite exp(logvar) se desborda a inf y toda la pérdida se vuelve NaN.
        return self.to_mu(h), torch.clamp(self.to_logvar(h), min=LOGVAR_MIN, max=LOGVAR_MAX)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Truco de reparametrización: muestrea z = mu + sigma*eps para poder retropropagar."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def _elbo_terms(reconstruction: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (error de reconstrucción, divergencia KL) por fila, sin promediar el batch."""
    # Media sobre features, no suma: con 15 columnas escaladas cuya suma de cuadrados por
    # fila llega a 3,7e6, el término sumado domina al KL por seis órdenes de magnitud y el
    # entrenamiento diverge. La media lo deja en la misma escala que el autoencoder del
    # Módulo 2, que sí converge sobre estos datos.
    recon = torch.mean((reconstruction - x) ** 2, dim=1)
    # KL cerrado entre N(mu, sigma^2) y N(0, 1), sumado sobre las dimensiones latentes.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon, kl


class VAEDetector:
    """Detector por ELBO negativo: penaliza reconstrucción mala *y* latente improbable.

    `beta` pondera el término KL. Con beta=1 es el ELBO estándar; se deja configurable
    porque en datos tabulares muy asimétricos un KL demasiado dominante colapsa la latente
    y el detector deja de distinguir (posterior collapse).

    El score se promedia sobre `n_samples` muestreos de la latente: el VAE es estocástico y
    un solo pase daría un score con varianza innecesaria entre corridas.
    """

    def __init__(self, latent: int = 4, activation: str = "relu", beta: float = 1.0, epochs: int = 30,
                 batch_size: int = 256, lr: float = 1e-3, n_samples: int = 8, random_state: int = 42):
        self.latent = latent
        self.activation = activation
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_samples = n_samples
        self.random_state = random_state
        self.model_: VAE | None = None

    def fit(self, X) -> "VAEDetector":
        torch.manual_seed(self.random_state)
        X_tensor = _as_tensor(X)
        self.model_ = VAE(X_tensor.shape[1], latent=self.latent, activation=self.activation)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in _batches(X_tensor, self.batch_size):
                optimizer.zero_grad()
                reconstruction, mu, logvar = self.model_(batch)
                recon, kl = _elbo_terms(reconstruction, batch, mu, logvar)
                loss = (recon + self.beta * kl).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        return self

    @torch.no_grad()
    def score_samples(self, X) -> np.ndarray:
        """ELBO por fila (más bajo = más anómalo), promediado sobre varios muestreos latentes."""
        self.model_.eval()
        X_tensor = _as_tensor(X)
        torch.manual_seed(self.random_state)

        total = torch.zeros(X_tensor.shape[0])
        for _ in range(self.n_samples):
            reconstruction, mu, logvar = self.model_(X_tensor)
            recon, kl = _elbo_terms(reconstruction, X_tensor, mu, logvar)
            total += recon + self.beta * kl
        negative_elbo = total / self.n_samples
        return -negative_elbo.numpy()


class DeepSVDDNet(nn.Module):
    """Red de proyección de Deep SVDD: sin sesgos, para evitar el colapso trivial.

    Si las capas tuvieran término de sesgo, la red podría minimizar el objetivo mapeando
    *toda* entrada al centro c — solución perfecta y completamente inútil (hypersphere
    collapse). Quitar los sesgos hace que esa solución constante sea inalcanzable.
    """

    def __init__(self, n_features: int, latent: int = 4, hidden: int | None = None, activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activación desconocida: {activation}. Usar una de {list(ACTIVATIONS)}")
        act_cls = ACTIVATIONS[activation]
        hidden = hidden or max(latent * 2, n_features // 2)

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden, bias=False),
            act_cls(),
            nn.Linear(hidden, latent, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSVDDDetector:
    """Deep SVDD (Ruff et al., 2018): aprende una esfera mínima que contiene lo normal.

    A diferencia del autoencoder, no reconstruye nada: entrena una proyección que acerca
    todas las transacciones normales a un centro fijo `c`, y el anomaly score es la
    distancia cuadrada a ese centro. El objetivo es directamente el de una clase, no un
    proxy de reconstrucción.

    `c` se fija con el promedio de las salidas de la red **sin entrenar** y luego no se
    actualiza: si se dejara aprender, la red y el centro convergerían juntos al colapso.
    Las componentes de `c` demasiado cercanas a cero se alejan a ±eps por la misma razón.
    """

    def __init__(self, latent: int = 4, activation: str = "relu", epochs: int = 30, batch_size: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-6, eps: float = 0.1, random_state: int = 42):
        self.latent = latent
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.random_state = random_state
        self.model_: DeepSVDDNet | None = None
        self.center_: torch.Tensor | None = None

    @torch.no_grad()
    def _init_center(self, X_tensor: torch.Tensor) -> torch.Tensor:
        self.model_.eval()
        center = self.model_(X_tensor).mean(dim=0)
        # Una componente del centro en ~0 permite que la red la anule con pesos pequeños y
        # gane objetivo sin aprender nada; se la empuja fuera de la vecindad de cero.
        center[center.abs() < self.eps] = self.eps
        return center

    def fit(self, X) -> "DeepSVDDDetector":
        torch.manual_seed(self.random_state)
        X_tensor = _as_tensor(X)
        self.model_ = DeepSVDDNet(X_tensor.shape[1], latent=self.latent, activation=self.activation)
        self.center_ = self._init_center(X_tensor)

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in _batches(X_tensor, self.batch_size):
                optimizer.zero_grad()
                distances = torch.sum((self.model_(batch) - self.center_) ** 2, dim=1)
                distances.mean().backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        return self

    @torch.no_grad()
    def score_samples(self, X) -> np.ndarray:
        """Distancia cuadrada al centro, negada (más bajo = más anómalo)."""
        self.model_.eval()
        projected = self.model_(_as_tensor(X))
        distances = torch.sum((projected - self.center_) ** 2, dim=1)
        return -distances.numpy()


def build_deep_detectors() -> dict:
    """Instancia los dos detectores profundos de una clase a nivel de transacción."""
    return {"vae": VAEDetector(), "deep_svdd": DeepSVDDDetector()}

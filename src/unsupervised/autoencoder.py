"""Autoencoder (PyTorch) para detección de anomalías por error de reconstrucción.

Se entrena únicamente con transacciones normales; el autoencoder aprende a comprimir y
reconstruir el patrón "normal". Una anomalía, al no encajar en ese patrón, produce un
error de reconstrucción (MSE) más alto — esa es la señal de anomaly score.

Se compara el mismo arquitectura con tres funciones de activación (ReLU, GELU, Swish/SiLU)
para ilustrar su efecto en la calidad de la reconstrucción sobre este dataset tabular.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # SiLU == Swish (x * sigmoid(x))
}


class Autoencoder(nn.Module):
    """Autoencoder totalmente conectado, simétrico, con cuello de botella central."""

    def __init__(self, n_features: int, activation: str = "relu", bottleneck: int = 4):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activación desconocida: {activation}. Usar una de {list(ACTIVATIONS)}")
        act_cls = ACTIVATIONS[activation]
        hidden = max(bottleneck * 2, n_features // 2)

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            act_cls(),
            nn.Linear(hidden, bottleneck),
            act_cls(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            act_cls(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_train: np.ndarray,
    activation: str = "relu",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    random_state: int = 42,
) -> Autoencoder:
    """Entrena un autoencoder sobre datos normales (MSE de reconstrucción)."""
    torch.manual_seed(random_state)
    n_features = X_train.shape[1]
    model = Autoencoder(n_features, activation=activation)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def reconstruction_error(model: Autoencoder, X) -> np.ndarray:
    """Anomaly score = error de reconstrucción (MSE por fila). Más alto = más anómalo."""
    model.eval()
    X_tensor = torch.tensor(np.asarray(X, dtype=np.float32))
    reconstruction = model(X_tensor)
    errors = torch.mean((reconstruction - X_tensor) ** 2, dim=1)
    return errors.numpy()

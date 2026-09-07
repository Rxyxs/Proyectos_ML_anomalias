"""Módulo 4 — detección de anomalías sobre la *historia* de una cuenta destino.

Los trece detectores de los Módulos 2 y 3 comparten una limitación estructural: tratan cada
transacción como una fila independiente. Una cuenta mula no es anómala por una transacción
suya en particular —cada monto individual puede ser perfectamente normal— sino por la
*secuencia*: recibe transferencias con una cadencia y un patrón de saldos que no se parecen
a los de una cuenta legítima.

Este módulo cambia la representación de los datos, no solo el modelo. Reconstruye la
historia ordenada de cada cuenta destino y la puntúa completa con un autoencoder GRU
entrenado únicamente con cuentas limpias.

Dos decisiones sobre los datos que condicionan todo lo demás:

- **Se agrupa por `nameDest`, no por `nameOrig`.** En PaySim el 99,85% de las cuentas de
  origen aparece una sola vez (ninguna llega a cinco transacciones): no existe una historia
  que modelar del lado emisor. Las cuentas destino sí acumulan — 280.200 tienen cinco o más
  transacciones, cubriendo el 56,5% del dataset.
- **Se descartan las cuentas de comercio (`M`).** El fraude de PaySim nunca tiene como
  destino un comercio: las 8.213 transacciones fraudulentas van a cuentas `C`. Dejar los
  comercios adentro le regalaría al modelo un atajo trivial (todo `M` es normal) y una
  métrica inflada que no mide nada.

**Advertencia de comparabilidad:** este detector puntúa *cuentas*, no transacciones. Sus
métricas no son comparables con las del benchmark del Módulo 3 y no deben ponerse en la
misma tabla — el universo evaluado y la unidad de la etiqueta son distintos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.loader import load_raw_data

MIN_SEQUENCE_LENGTH = 5
MAX_SEQUENCE_LENGTH = 20

TRAIN_NORMAL_ACCOUNTS = 40_000
TEST_NORMAL_ACCOUNTS = 40_000

FEATURE_NAMES = [
    "log_amount",
    "delta_step",
    "log_oldbalance_dest",
    "log_newbalance_dest",
    "is_transfer",
    "is_cash_out",
    "is_other_type",
]


def _step_features(df: pd.DataFrame) -> np.ndarray:
    """Features por transacción de la secuencia, ya en escala logarítmica donde corresponde."""
    delta_step = df.groupby("nameDest", sort=False)["step"].diff().fillna(0.0)

    return np.column_stack([
        np.log1p(df["amount"].to_numpy()),
        # Cadencia: horas desde la transacción anterior hacia la misma cuenta. Es la señal
        # que solo existe en la secuencia y que un modelo fila a fila no puede ver.
        delta_step.to_numpy(),
        np.log1p(df["oldbalanceDest"].to_numpy()),
        np.log1p(df["newbalanceDest"].to_numpy()),
        (df["type"] == "TRANSFER").to_numpy(),
        (df["type"] == "CASH_OUT").to_numpy(),
        (~df["type"].isin(["TRANSFER", "CASH_OUT"])).to_numpy(),
    ]).astype(np.float32)


def _account_positions(df: pd.DataFrame, min_length: int):
    """Ordena por cuenta y calcula, para cada fila, su posición contada desde la más reciente.

    Devuelve `(df_ordenado, codes, counts, position_from_end, long_enough)`. Se comparte
    entre la construcción de secuencias y la comprobación de cobertura para que ambas midan
    exactamente sobre el mismo universo de cuentas.
    """
    df = df[df["nameDest"].str.startswith("C")]
    # mergesort es estable: preserva el orden original entre transacciones del mismo step,
    # que es lo más cercano a un orden cronológico real dentro de una misma hora.
    df = df.sort_values(["nameDest", "step"], kind="mergesort").reset_index(drop=True)

    codes, _ = pd.factorize(df["nameDest"])
    counts = np.bincount(codes)
    long_enough = counts >= min_length

    group_start = np.repeat(np.cumsum(counts) - counts, counts)
    position_from_end = np.repeat(counts, counts) - 1 - (np.arange(len(df)) - group_start)
    return df, codes, counts, position_from_end, long_enough


def fraud_window_coverage(
    df: pd.DataFrame,
    min_length: int = MIN_SEQUENCE_LENGTH,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> dict:
    """Cuántas transacciones fraudulentas sobreviven al truncado de la secuencia.

    Truncar a las `max_length` transacciones más recientes puede dejar el fraude fuera de la
    ventana modelada. Si eso pasara masivamente, un mal resultado del detector no diría nada
    sobre el método: estaría midiendo un recorte mal elegido. Esta función permite descartar
    esa explicación con un número en vez de con una suposición.
    """
    df, codes, _, position_from_end, long_enough = _account_positions(df, min_length)

    eligible = long_enough[codes] & (df["isFraud"].to_numpy() == 1)

    inside = int((position_from_end[eligible] < max_length).sum())
    total = int(eligible.sum())
    return {
        "fraud_transactions": total,
        "inside_window": inside,
        "coverage": inside / total if total else 0.0,
        "median_position_from_end": float(np.median(position_from_end[eligible])) if total else 0.0,
    }


def build_account_sequences(
    df: pd.DataFrame,
    min_length: int = MIN_SEQUENCE_LENGTH,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte transacciones crudas en secuencias por cuenta destino.

    Devuelve `(sequences, mask, labels)`:
    - `sequences` de forma (n_cuentas, max_length, n_features), con relleno **al inicio**
      para que la transacción más reciente quede siempre en la última posición;
    - `mask` de forma (n_cuentas, max_length), 1 en las posiciones reales y 0 en el relleno;
    - `labels` con 1 si *alguna* transacción recibida por la cuenta fue fraudulenta.

    De las cuentas con más de `max_length` transacciones se conservan las más recientes.
    """
    df, codes, counts, position_from_end, long_enough = _account_positions(df, min_length)

    if not long_enough.any():
        raise ValueError(f"Ninguna cuenta destino alcanza {min_length} transacciones.")

    keep = long_enough[codes] & (position_from_end < max_length)

    # Reindexa las cuentas conservadas a 0..n-1 y ubica cada transacción en su ranura,
    # dejando el relleno al inicio de la secuencia.
    account_index = np.full(len(counts), -1, dtype=np.int64)
    account_index[long_enough] = np.arange(long_enough.sum())
    rows = account_index[codes[keep]]
    slots = max_length - 1 - position_from_end[keep]

    features = _step_features(df)[keep]
    n_accounts = int(long_enough.sum())

    sequences = np.zeros((n_accounts, max_length, features.shape[1]), dtype=np.float32)
    mask = np.zeros((n_accounts, max_length), dtype=np.float32)
    sequences[rows, slots] = features
    mask[rows, slots] = 1.0

    fraud_by_account = np.zeros(len(counts), dtype=np.int64)
    np.maximum.at(fraud_by_account, codes, df["isFraud"].to_numpy())
    labels = fraud_by_account[long_enough]

    return sequences, mask, labels


def get_sequence_data(
    train_size: int = TRAIN_NORMAL_ACCOUNTS,
    test_size: int = TEST_NORMAL_ACCOUNTS,
    random_state: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Prepara el split a nivel de cuenta: entrenamiento 100% limpio, prueba mixta.

    Devuelve `((seq_train, mask_train), (seq_test, mask_test), y_test)`. El entrenamiento
    solo contiene cuentas sin una sola transacción fraudulenta; la prueba mezcla cuentas
    limpias con *todas* las cuentas marcadas como fraude.
    """
    df = load_raw_data()
    sequences, mask, labels = build_account_sequences(df)

    rng = np.random.default_rng(random_state)
    normal_idx = np.flatnonzero(labels == 0)
    fraud_idx = np.flatnonzero(labels == 1)

    chosen = rng.permutation(normal_idx)[: train_size + test_size]
    train_idx = chosen[:train_size]
    test_normal_idx = chosen[train_size:]
    test_idx = rng.permutation(np.concatenate([test_normal_idx, fraud_idx]))

    return (
        (sequences[train_idx], mask[train_idx]),
        (sequences[test_idx], mask[test_idx]),
        labels[test_idx],
    )


class SequenceAutoencoder(nn.Module):
    """Autoencoder GRU: comprime la historia completa de una cuenta y la reconstruye.

    El encoder recorre la secuencia y se queda con su último estado oculto — un resumen de
    tamaño fijo de toda la historia. El decoder tiene que reconstruir la secuencia entera a
    partir de ese único vector, así que el cuello de botella fuerza a la red a quedarse con
    el patrón de comportamiento, no con las transacciones sueltas.
    """

    def __init__(self, n_features: int, hidden: int = 16, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden, batch_first=True)
        self.to_bottleneck = nn.Linear(hidden, bottleneck)
        self.decoder = nn.GRU(bottleneck, hidden, batch_first=True)
        self.to_output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, last_hidden = self.encoder(x)
        code = self.to_bottleneck(last_hidden[-1])
        # El mismo código se repite en cada paso: el decoder no recibe la entrada original,
        # así que no puede copiarla y está obligado a reconstruir desde el resumen.
        repeated = code.unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.to_output(decoded)


def _masked_errors(reconstruction: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Error cuadrático medio por secuencia, contando solo las posiciones reales."""
    squared = ((reconstruction - x) ** 2).mean(dim=2)
    valid = mask.sum(dim=1).clamp(min=1.0)
    return (squared * mask).sum(dim=1) / valid


def train_sequence_autoencoder(
    sequences: np.ndarray,
    mask: np.ndarray,
    hidden: int = 16,
    bottleneck: int = 8,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    random_state: int = 42,
) -> SequenceAutoencoder:
    """Entrena el autoencoder secuencial con cuentas limpias, ignorando el relleno."""
    torch.manual_seed(random_state)
    model = SequenceAutoencoder(sequences.shape[2], hidden=hidden, bottleneck=bottleneck)

    dataset = torch.utils.data.TensorDataset(torch.tensor(sequences), torch.tensor(mask))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch, batch_mask in loader:
            optimizer.zero_grad()
            loss = _masked_errors(model(batch), batch, batch_mask).mean()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def sequence_step_errors(model: SequenceAutoencoder, sequences: np.ndarray, mask: np.ndarray,
                         batch_size: int = 1024) -> np.ndarray:
    """Error de reconstrucción por paso, de forma (n_cuentas, max_length), con relleno en 0.

    Se expone aparte del score agregado porque la forma de resumir esos errores en un solo
    número es una decisión de diseño discutible —una única transacción anómala se diluye al
    promediarla sobre veinte pasos— y conviene poder medir el efecto en vez de suponerlo.
    """
    model.eval()
    errors = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.tensor(sequences[start : start + batch_size])
        batch_mask = torch.tensor(mask[start : start + batch_size])
        squared = ((model(batch) - batch) ** 2).mean(dim=2)
        errors.append((squared * batch_mask).numpy())
    return np.concatenate(errors)


def aggregate_step_errors(step_errors: np.ndarray, mask: np.ndarray, how: str = "mean") -> np.ndarray:
    """Resume los errores por paso en un score por cuenta. Más alto = más anómalo.

    - `mean`: promedio sobre los pasos reales — el comportamiento global de la cuenta;
    - `max`: el paso peor reconstruido — sensible a una única transacción anómala;
    - `top3`: promedio de los tres peores — intermedio, menos sensible al ruido que `max`;
    - `last`: solo la transacción más reciente.
    """
    valid = mask.sum(axis=1).clip(min=1.0)
    if how == "mean":
        return step_errors.sum(axis=1) / valid
    if how == "max":
        return step_errors.max(axis=1)
    if how == "top3":
        return np.sort(step_errors, axis=1)[:, -3:].mean(axis=1)
    if how == "last":
        return step_errors[:, -1]
    raise ValueError(f"Agregación desconocida: {how}. Usar una de ['mean', 'max', 'top3', 'last'].")


def sequence_anomaly_score(model: SequenceAutoencoder, sequences: np.ndarray, mask: np.ndarray,
                           batch_size: int = 1024, how: str = "mean") -> np.ndarray:
    """Anomaly score por cuenta = error de reconstrucción de su historia. Más alto = más anómalo."""
    step_errors = sequence_step_errors(model, sequences, mask, batch_size=batch_size)
    return aggregate_step_errors(step_errors, mask, how=how)

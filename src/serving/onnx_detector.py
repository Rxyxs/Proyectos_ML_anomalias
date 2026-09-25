"""Wrapper de inferencia acelerada sobre una `onnxruntime.InferenceSession`.

Mismo contrato de telemetría que `DetectorPackage` (`src/serving/metrics.py`,
Día 1/2 de este plan) y la misma convención de signo que `anomaly_score()`
(`src/unsupervised/models.py`): más alto = más anómalo. La diferencia real
es de motor -- acá no hay scikit-learn ni el objeto Python original en el
proceso que sirve, solo el grafo ONNX y `onnxruntime`.

Latencia sub-milisegundo real, con una salvedad medida y no escondida: el
costo del op TreeEnsemble de ONNX Runtime escala ~linealmente con la
cantidad de árboles de un `IsolationForest` (perfilado real, latencia media
por fila: 10 árboles -> 0.50ms, 100 -> 4.79ms, 200 -> 9.29ms). Los 200
árboles por defecto de `build_isolation_forest()` (`src/unsupervised/models.py`)
NO llegan a sub-milisegundo en CPU; `tests/test_onnx_serving.py` lo verifica
con 15 árboles, donde sí hay margen real (media 0.55ms, máximo 0.76ms sobre
500 corridas). Para servir el modelo de 200 árboles bajo ese presupuesto de
latencia hacen falta más árboles solo si el caso de uso realmente los
necesita -- y ahí el camino es reducir n_estimators o batchear filas, no
este wrapper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.serving.metrics import instrument
from src.unsupervised.models import assert_finite


class ONNXDetectorWrapper:
    """Sirve un detector ya exportado a ONNX (`onnx_exporter.export_detector_to_onnx`).

    `threshold` es el punto de corte de `predict()`, calculado afuera (ej.
    sobre las mismas `calibration_scores` que usaría `DetectorPackage`) --
    este wrapper no recalibra nada, solo puntúa y decide con el umbral que
    se le da.
    """

    def __init__(self, onnx_path: str | Path, threshold: float = 0.0,
                 detector_name: str = "onnx_detector") -> None:
        self.onnx_path = Path(onnx_path)
        self.threshold = threshold
        self.metadata: dict[str, Any] = {"detector": detector_name}

        self._session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        declared = self._session.get_inputs()[0].shape[1]
        if not isinstance(declared, int):
            raise ValueError(
                f"El grafo ONNX en {self.onnx_path} no declara una cantidad fija de "
                f"features en su entrada (llegó {declared!r})."
            )
        self.feature_count: int = declared

    def _validate_shape(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"Se esperaba un arreglo 2D [batch_size, feature_count], llegó {X.ndim}D."
            )
        if X.shape[1] != self.feature_count:
            raise ValueError(
                f"Se esperaban {self.feature_count} columnas, llegaron {X.shape[1]}."
            )
        return X

    @instrument(record_scores=True)
    def score(self, X) -> np.ndarray:
        """Anomaly score crudo. Más alto = más anómalo (misma convención que
        `anomaly_score()`: la sesión ONNX expone `score_samples` con la
        convención de sklearn -- más bajo = más anómalo --, así que se
        invierte el signo acá, una sola vez, en el mismo lugar donde el
        modelo nativo lo hace).
        """
        X = self._validate_shape(X)
        X = assert_finite(X, "la entrada al detector ONNX").astype(np.float32)
        outputs = self._session.run(["score_samples"], {self._input_name: X})
        return -np.asarray(outputs[0]).reshape(-1)

    @instrument()
    def predict(self, X) -> np.ndarray:
        """True para las filas cuyo score alcanza o supera `threshold`."""
        return self.score(X) >= self.threshold

"""Exportación de detectores de anomalías a ONNX, vía `skl2onnx`.

Cobertura real, no universal: solo los estimadores sklearn que exponen
`score_samples` y tienen un convertidor registrado en `skl2onnx`
(`IsolationForest`, `LocalOutlierFactor`, `GaussianMixture`, `PCA`,
`NearestNeighbors`, entre otros -- verificable con
`skl2onnx.supported_converters(from_sklearn=True)`) se pueden exportar.
Los detectores hechos a mano de este repo (HBOS, ECOD, LODA, MADBaseline --
`src/unsupervised/families.py`/`models.py`) NO son estimadores sklearn y no
tienen convertidor: `export_detector_to_onnx` lo detecta y lanza
`UnsupportedDetectorError` en vez de producir un grafo silenciosamente
incorrecto.

Verificado end-to-end con `IsolationForest`: el `score_samples` que devuelve
el grafo ONNX coincide con el del modelo nativo a ~1e-7 (ver
`tests/test_onnx_serving.py`), muy por debajo de la tolerancia de 1e-4 que
pide el Día 9.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# El dominio ai.onnx.ml a la versión 3: fijado explícitamente porque sin esto
# `to_onnx` apunta a la versión más nueva que conoce el ONNX instalado y
# skl2onnx (1.20.0) todavía no la soporta -- falla con "not supported yet by
# this library" si se omite.
_TARGET_OPSET = {"": 18, "ai.onnx.ml": 3}

# La salida que necesita ONNXDetectorWrapper: `score_samples`, no `scores`
# (que en sklearn es `decision_function`, desplazado por el umbral de
# contaminación) -- es la que reproduce la misma convención de signo que
# `anomaly_score()` (src/unsupervised/models.py: `-model.score_samples(X)`).
REQUIRED_OUTPUT_NAME = "score_samples"


class UnsupportedDetectorError(TypeError):
    """El estimador no se puede convertir a ONNX: no tiene `score_samples`,
    no tiene un convertidor registrado en skl2onnx, o la conversión no
    expuso la salida `score_samples` que este módulo requiere."""


def export_detector_to_onnx(model: Any, feature_count: int, output_path: str | Path) -> Path:
    """Convierte `model` (un estimador sklearn ya ajustado) a un grafo ONNX
    con entrada `[batch_size, feature_count]` (`batch_size` dinámico) y lo
    guarda en `output_path`. Devuelve la ruta escrita.

    Valida los metadatos del grafo (forma de la entrada, presencia de la
    salida `score_samples`) antes de guardar -- un esquema incorrecto falla
    acá, en el momento de exportar, en vez de con un error más críptico de
    ONNX Runtime recién al servir la primera solicitud real.
    """
    if feature_count <= 0:
        raise ValueError(f"feature_count debe ser positivo, llegó {feature_count}.")
    if not hasattr(model, "score_samples"):
        raise UnsupportedDetectorError(
            f"{type(model).__name__} no expone score_samples() -- no es un estimador "
            "sklearn compatible con la conversión a ONNX de este módulo."
        )

    try:
        onnx_model = to_onnx(
            model,
            initial_types=[("input", FloatTensorType([None, feature_count]))],
            options={type(model): {"score_samples": True}},
            target_opset=_TARGET_OPSET,
        )
    except Exception as exc:  # noqa: BLE001 - skl2onnx lanza tipos de error variados según el modelo
        raise UnsupportedDetectorError(
            f"No se pudo convertir {type(model).__name__} a ONNX: {exc}"
        ) from exc

    output_names = {o.name for o in onnx_model.graph.output}
    if REQUIRED_OUTPUT_NAME not in output_names:
        raise UnsupportedDetectorError(
            f"La conversión de {type(model).__name__} no expuso la salida "
            f"'{REQUIRED_OUTPUT_NAME}' (salidas disponibles: {sorted(output_names)})."
        )

    _validate_input_metadata(onnx_model, type(model).__name__, feature_count)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(output_path))
    return output_path


def _validate_input_metadata(onnx_model, model_name: str, feature_count: int) -> None:
    entrada = onnx_model.graph.input[0]
    dims = entrada.type.tensor_type.shape.dim
    if len(dims) != 2:
        raise UnsupportedDetectorError(
            f"{model_name}: se esperaba una entrada 2D [batch_size, feature_count], "
            f"el grafo declara {len(dims)}D."
        )
    # dim[0] (batch_size) se deja dinámico a propósito (dim_value=0 / dim_param
    # seteado) -- solo se valida la dimensión de features, fija.
    dim_features = dims[1].dim_value
    if dim_features != feature_count:
        raise UnsupportedDetectorError(
            f"{model_name}: la entrada del grafo tiene {dim_features} features, "
            f"se esperaban {feature_count}."
        )

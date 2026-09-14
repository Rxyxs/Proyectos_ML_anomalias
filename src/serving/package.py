"""Módulo 8 — el paquete de scoring: lo que efectivamente se despliega.

Los siete módulos anteriores entrenan y evalúan dieciséis detectores, pero ninguno deja algo
con lo que puntuar una transacción nueva. Es el hueco que contradice todo el hilo de los
Módulos 5 a 7: se calcularon umbrales, se midió cuánta plata salva cada punto de operación y
se discutió recalibración, sin que existiera un objeto capaz de recibir una transacción y
responder.

Un modelo serializado no alcanza. Para decidir sobre una transacción —y poder explicar la
decisión— hacen falta seis cosas, y si alguna viaja por separado el sistema se rompe en
silencio:

1. **El detector ajustado** — el modelo propiamente dicho.
2. **El escalador** — ajustado *solo* con los datos de entrenamiento. Reajustarlo en
   producción sobre el tráfico del día cambiaría la escala bajo los pies del detector.
3. **Los scores de calibración** — de un conjunto retenido, para convertir un score crudo en
   un p-valor conforme (Módulo 6). Sin ellos el score es un número sin unidades.
4. **El nivel alpha** — la tasa de falsas alarmas que el sistema promete.
5. **El orden y nombre de las columnas** — un `DataFrame` con las mismas columnas en otro
   orden produce scores plausibles y completamente equivocados. Es el modo de falla más
   difícil de detectar de todos, porque nada tira una excepción.
6. **Un fondo de filas normales y los grupos de columnas dependientes** — para explicar por
   qué se disparó una alerta. Una explicación calculada contra otro fondo, o sin respetar las
   dependencias entre columnas, no es la misma explicación.

`DetectorPackage` las mantiene juntas en un solo archivo y valida la quinta en cada llamada.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.conformal.conformal import conformal_p_values, conformal_threshold
from src.serving.explain import infer_dependency_groups
from src.unsupervised.models import anomaly_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_PATH = PROJECT_ROOT / "data" / "processed" / "detector_package.joblib"


@dataclass
class DetectorPackage:
    """Todo lo necesario para puntuar una transacción, en un solo objeto serializable."""

    detector: object
    scaler: object
    calibration_scores: np.ndarray
    feature_names: list[str]
    alpha: float = 0.01
    # Muestra de filas normales, ya escaladas, contra la que se explica una alerta. Va dentro
    # del paquete porque una explicación calculada con otro fondo no es la misma explicación.
    background: np.ndarray | None = None
    # Columnas que no pueden ocluirse por separado sin producir un punto imposible. Viajan con
    # el paquete porque dependen de como se construyeron las features, no del detector.
    groups: list[list[int]] | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ validación

    def to_scaled_matrix(self, X) -> np.ndarray:
        """Valida columnas y devuelve la matriz escalada, en el orden del entrenamiento.

        Acepta un DataFrame —y entonces reordena por nombre— o un arreglo, en cuyo caso solo
        puede verificar la cantidad de columnas. Pasar un DataFrame es lo recomendado
        precisamente porque permite la verificación fuerte.
        """
        if isinstance(X, pd.DataFrame):
            faltantes = [c for c in self.feature_names if c not in X.columns]
            if faltantes:
                raise ValueError(f"Faltan columnas en la entrada: {faltantes}")
            sobrantes = [c for c in X.columns if c not in self.feature_names]
            if sobrantes:
                raise ValueError(
                    f"Columnas inesperadas: {sobrantes}. El paquete espera exactamente "
                    f"{len(self.feature_names)} columnas."
                )
            X = X[self.feature_names]
            # Se entrega al escalador en la misma forma en que fue ajustado. scikit-learn
            # advierte cuando recibe nombres habiendo sido ajustado sin ellos, y también al
            # revés; mirar `feature_names_in_` evita las dos advertencias en vez de cambiar
            # una por la otra.
            if not hasattr(self.scaler, "feature_names_in_"):
                X = X.to_numpy()
        else:
            X = np.asarray(X)
            if X.ndim != 2 or X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Se esperaban {len(self.feature_names)} columnas y llegaron "
                    f"{X.shape[1] if X.ndim == 2 else X.shape}. Pasar un DataFrame permite "
                    f"validar además el orden."
                )
            # Caso simétrico al anterior: el escalador espera nombres y le llega un arreglo.
            if hasattr(self.scaler, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names)
        return self.scaler.transform(X)

    # ------------------------------------------------------------------ scoring

    def score(self, X) -> np.ndarray:
        """Anomaly score crudo. Más alto = más anómalo."""
        return anomaly_score(self.detector, self.to_scaled_matrix(X))

    def score_from_scaled(self, X_scaled) -> np.ndarray:
        """Puntúa una matriz **ya escalada**, sin volver a pasar por el escalador.

        La explicación por oclusión modifica los datos en el espacio escalado y vuelve a
        puntuar muchas veces; escalar de nuevo en cada pasada sería incorrecto además de caro.
        """
        return anomaly_score(self.detector, np.asarray(X_scaled, dtype=float))

    def p_values(self, X) -> np.ndarray:
        """P-valor conforme por transacción. Más bajo = más anómalo.

        A diferencia del score crudo, el p-valor sí tiene unidades interpretables: es la
        fracción del tráfico legítimo de calibración que resulta al menos tan anómala.
        """
        return conformal_p_values(self.calibration_scores, self.score(X))

    @property
    def threshold(self) -> float:
        """Score a partir del cual se dispara una alerta, dado el alpha del paquete."""
        return conformal_threshold(self.calibration_scores, self.alpha)

    def predict(self, X) -> np.ndarray:
        """True para las transacciones que hay que alertar."""
        return self.p_values(X) <= self.alpha

    # ------------------------------------------------------------------ persistencia

    def save(self, path: Path = DEFAULT_PACKAGE_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = DEFAULT_PACKAGE_PATH) -> "DetectorPackage":
        return joblib.load(Path(path))


def build_package(
    detector,
    scaler,
    X_calibration_scaled: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.01,
    detector_name: str = "desconocido",
    background_size: int = 50,
    random_state: int = 42,
    extra: dict | None = None,
) -> DetectorPackage:
    """Arma el paquete a partir de un detector ya ajustado y su conjunto de calibración.

    `X_calibration_scaled` tiene que venir de datos que el detector **no** usó para ajustarse
    y del mismo período que el entrenamiento; es la condición de intercambiabilidad de la que
    depende la garantía del p-valor.

    De ese mismo conjunto se toma el fondo con el que se explican las alertas. `background_size`
    controla el compromiso: más filas dan una atribución más estable y cuestan una pasada de
    scoring más por feature, así que solo pesa al explicar, no al puntuar.
    """
    X_calibration_scaled = np.asarray(X_calibration_scaled, dtype=float)
    rng = np.random.default_rng(random_state)
    n_fondo = min(background_size, len(X_calibration_scaled))
    fondo = X_calibration_scaled[rng.choice(len(X_calibration_scaled), n_fondo, replace=False)]

    metadata = {
        "detector": detector_name,
        "alpha": alpha,
        "n_calibration": int(len(X_calibration_scaled)),
        "n_background": int(n_fondo),
        "n_features": len(feature_names),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    metadata.update(extra or {})

    return DetectorPackage(
        detector=detector,
        scaler=scaler,
        calibration_scores=anomaly_score(detector, X_calibration_scaled),
        feature_names=list(feature_names),
        background=fondo,
        groups=infer_dependency_groups(list(feature_names)),
        alpha=alpha,
        metadata=metadata,
    )

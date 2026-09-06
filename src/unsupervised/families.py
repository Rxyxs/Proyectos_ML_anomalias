"""Módulo 3 — familias complementarias de detección de anomalías.

El módulo 2 cubre tres enfoques (aislamiento por particiones, densidad local y z-score
robusto por columna) más el autoencoder. Cada uno falla de una forma distinta, y ninguno
domina a los demás en datos tabulares: por eso conviene tener representada cada *familia*
de detectores, no varios modelos de la misma familia.

Aquí se agregan las que faltaban, con la misma convención de la API de scikit-learn
(fit / score_samples, donde **más bajo = más anómalo**) para que anomaly_score() de
src.unsupervised.models funcione igual sobre todas:

| Detector             | Familia                  | Qué anomalía detecta bien                            |
|----------------------|--------------------------|------------------------------------------------------|
| PCAReconstruction    | Reconstrucción lineal    | Filas fuera del subespacio principal (ablación del AE)|
| GMMDensity           | Densidad paramétrica     | Zonas de baja probabilidad bajo una mezcla gaussiana  |
| RobustMahalanobis    | Covarianza robusta (MCD) | Correlaciones rotas entre features                    |
| KNNDistance          | Distancia global         | Puntos lejos de cualquier vecindario denso            |
| OneClassSVMApprox    | Frontera con kernel      | Filas fuera de la envolvente aprendida de lo normal   |
| HBOS                 | Histogramas por feature  | Valores raros en columnas individuales (muy rápido)   |
| ECOD                 | Colas de la CDF empírica | Colas extremas, sin hiperparámetros ni distancias     |

HBOS y ECOD se implementan desde cero (son de coste lineal y no requieren una dependencia
extra); el resto se apoya en scikit-learn.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

EPS = 1e-12


class PCAReconstruction:
    """Error de reconstrucción con PCA: la contraparte lineal del autoencoder.

    Proyecta cada fila sobre los primeros componentes principales (ajustados solo con
    transacciones normales) y la reconstruye. Si lo normal vive aproximadamente en un
    subespacio lineal, una anomalía queda fuera y su reconstrucción falla.

    Sirve como ablación honesta del autoencoder: si la red neuronal no supera este
    baseline, su no-linealidad no está aportando nada en este dataset.
    """

    def __init__(self, n_components: float | int = 0.9, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca_: PCA | None = None

    def fit(self, X) -> "PCAReconstruction":
        X = np.asarray(X, dtype=float)
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state).fit(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        reconstruction = self.pca_.inverse_transform(self.pca_.transform(X))
        return -np.mean((X - reconstruction) ** 2, axis=1)


class RobustMahalanobis:
    """Distancia de Mahalanobis sobre covarianza robusta (Minimum Covariance Determinant).

    El baseline MAD-z mira cada columna por separado, así que no ve una fila cuyos valores
    son todos plausibles individualmente pero cuya *combinación* es imposible (por ejemplo,
    un monto alto con saldo de origen en cero). Mahalanobis sí: mide la distancia teniendo
    en cuenta la matriz de covarianza completa.

    Se estima con MCD en vez de la covarianza empírica porque esta última se contamina con
    los mismos outliers que se quieren detectar (efecto de enmascaramiento).

    Sobre PaySim, scikit-learn advierte que la matriz de covarianza no es de rango completo:
    las features son linealmente dependientes por construcción — las dummies de `type_*`
    suman 1, y `errorBalanceOrig`/`errorBalanceDest` son combinaciones lineales exactas de
    `amount` y los saldos. La advertencia es esperable y no invalida el detector: MCD
    resuelve el sistema con la pseudo-inversa, que ignora las direcciones degeneradas y
    calcula la distancia sobre el subespacio de rango completo.
    """

    def __init__(self, support_fraction: float = 0.9, random_state: int = 42):
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.estimator_: MinCovDet | None = None

    def fit(self, X) -> "RobustMahalanobis":
        X = np.asarray(X, dtype=float)
        self.estimator_ = MinCovDet(
            support_fraction=self.support_fraction, random_state=self.random_state
        ).fit(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return -self.estimator_.mahalanobis(X)


class KNNDistance:
    """Distancia al k-ésimo vecino más cercano — el detector de distancia global clásico.

    LOF compara la densidad *local* de un punto con la de sus vecinos, así que puede
    considerar normal a un punto aislado si sus vecinos también lo están. Este detector no:
    la distancia cruda al k-ésimo vecino del set normal es una medida global de rareza.
    """

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.index_: NearestNeighbors | None = None

    def fit(self, X) -> "KNNDistance":
        X = np.asarray(X, dtype=float)
        self.index_ = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        distances, _ = self.index_.kneighbors(X)
        return -distances[:, -1]


class OneClassSVMApprox:
    """One-Class SVM con kernel RBF aproximado (Nyström + SGD).

    Aprende una frontera que encierra la región de lo normal. El OneClassSVM exacto es
    O(n^2)-O(n^3) y no termina en un set de 30k filas en tiempo razonable; la aproximación
    de Nyström mapea los datos a un espacio de características explícito donde
    SGDOneClassSVM resuelve el mismo problema en tiempo lineal.
    """

    def __init__(self, nu: float = 0.05, n_components: int = 300, random_state: int = 42):
        self.nu = nu
        self.n_components = n_components
        self.random_state = random_state
        self.pipeline_ = None

    def fit(self, X) -> "OneClassSVMApprox":
        X = np.asarray(X, dtype=float)
        self.pipeline_ = make_pipeline(
            Nystroem(
                gamma=1.0 / X.shape[1],
                n_components=self.n_components,
                random_state=self.random_state,
            ),
            SGDOneClassSVM(nu=self.nu, random_state=self.random_state),
        ).fit(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        # decision_function ya es más baja cuanto más anómalo (fuera de la frontera).
        return self.pipeline_.decision_function(np.asarray(X, dtype=float))


class HBOS:
    """Histogram-Based Outlier Score: densidad por histograma, feature por feature.

    Asume independencia entre columnas — supuesto falso pero deliberado: a cambio el
    detector es de coste lineal, no calcula distancias y su score se descompone por
    feature, así que un analista puede ver *qué columna* disparó la alerta.

    Score = suma sobre features de log(1 / densidad del bin donde cae el valor).
    """

    def __init__(self, n_bins: int | None = None):
        self.n_bins = n_bins
        self.bin_edges_: list[np.ndarray] = []
        self.bin_density_: list[np.ndarray] = []

    def fit(self, X) -> "HBOS":
        X = np.asarray(X, dtype=float)
        n_bins = self.n_bins or max(5, int(np.sqrt(X.shape[0])))
        self.bin_edges_, self.bin_density_ = [], []

        for column in X.T:
            counts, edges = np.histogram(column, bins=n_bins)
            width = np.diff(edges)
            # Densidad normalizada a su máximo: hace comparables features con escalas
            # distintas sin depender del ancho absoluto del bin.
            density = counts / (width + EPS)
            density = density / (density.max() + EPS)
            self.bin_edges_.append(edges)
            self.bin_density_.append(density)
        return self

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        total = np.zeros(X.shape[0])

        for j, column in enumerate(X.T):
            edges, density = self.bin_edges_[j], self.bin_density_[j]
            # np.digitize sobre los bordes interiores mapea los valores fuera del rango
            # visto en entrenamiento al bin extremo, cuya densidad ya es muy baja.
            idx = np.clip(np.digitize(column, edges[1:-1]), 0, len(density) - 1)
            total += np.log(1.0 / (density[idx] + EPS))
        return -total


class ECOD:
    """ECOD — Empirical Cumulative distribution based Outlier Detection.

    Para cada feature estima su CDF empírica con los datos normales y mide qué tan lejos
    está un valor en la cola: una probabilidad de cola diminuta significa un valor extremo.
    Los aportes por feature se suman en escala logarítmica (equivale a asumir independencia,
    igual que HBOS, pero sin discretizar en bins).

    Se calculan tres agregados —cola izquierda, cola derecha y cola elegida por el signo de
    la asimetría de cada feature— y el score final es el máximo de los tres, como en el
    paper original (Li et al., 2022). No tiene un solo hiperparámetro que ajustar, lo que
    lo vuelve un baseline difícil de descartar por "mala configuración".
    """

    def __init__(self):
        self.train_sorted_: np.ndarray | None = None
        self.skew_sign_: np.ndarray | None = None
        self.n_train_: int = 0

    @staticmethod
    def _skewness(X: np.ndarray) -> np.ndarray:
        centered = X - X.mean(axis=0)
        std = centered.std(axis=0) + EPS
        return np.mean((centered / std) ** 3, axis=0)

    def fit(self, X) -> "ECOD":
        X = np.asarray(X, dtype=float)
        self.n_train_ = X.shape[0]
        self.train_sorted_ = np.sort(X, axis=0)
        self.skew_sign_ = np.sign(self._skewness(X))
        return self

    def _tail_probabilities(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """CDF empírica izquierda y derecha, columna a columna, contra los datos de entrenamiento."""
        n = self.n_train_
        left = np.empty_like(X, dtype=float)
        right = np.empty_like(X, dtype=float)

        for j in range(X.shape[1]):
            train_column = self.train_sorted_[:, j]
            # P(Xtrain <= x) y P(Xtrain >= x); se suma 1 al numerador y al denominador para
            # que ningún valor fuera del rango visto obtenga probabilidad 0 (log infinito).
            left[:, j] = (np.searchsorted(train_column, X[:, j], side="right") + 1) / (n + 1)
            right[:, j] = (n - np.searchsorted(train_column, X[:, j], side="left") + 1) / (n + 1)
        return left, right

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        left, right = self._tail_probabilities(X)

        outlier_left = -np.log(left).sum(axis=1)
        outlier_right = -np.log(right).sum(axis=1)
        # Cola "automática": en una feature con asimetría positiva la cola informativa es la
        # derecha, y viceversa; el paper usa el signo de la asimetría para elegirla.
        skew_tail = np.where(self.skew_sign_ < 0, np.log(left), np.log(right))
        outlier_skew = -skew_tail.sum(axis=1)

        score = np.maximum.reduce([outlier_left, outlier_right, outlier_skew])
        return -score


class GMMDensity:
    """Densidad bajo una mezcla de gaussianas: el score es la log-verosimilitud negada.

    Es el enfoque generativo: en vez de trazar una frontera, modela *cómo se genera* lo
    normal como una mezcla de k gaussianas, y una transacción es anómala si resulta
    improbable bajo ese modelo. reg_covar evita covarianzas singulares por las columnas
    dummy de type_*, que son casi constantes dentro de algunos componentes.
    """

    def __init__(self, n_components: int = 8, reg_covar: float = 1e-4, random_state: int = 42):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.gmm_: GaussianMixture | None = None

    def fit(self, X) -> "GMMDensity":
        X = np.asarray(X, dtype=float)
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        ).fit(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        # GaussianMixture.score_samples ya devuelve log-verosimilitud: más baja = más anómalo.
        return self.gmm_.score_samples(np.asarray(X, dtype=float))


def build_detectors() -> dict:
    """Instancia las siete familias complementarias con su configuración por defecto."""
    return {
        "pca_reconstruction": PCAReconstruction(),
        "gmm_density": GMMDensity(),
        "robust_mahalanobis": RobustMahalanobis(),
        "knn_distance": KNNDistance(),
        "ocsvm_nystroem": OneClassSVMApprox(),
        "hbos": HBOS(),
        "ecod": ECOD(),
    }

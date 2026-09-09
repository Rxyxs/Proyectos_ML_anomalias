"""Módulo 7 — Half-Space Trees: detección en streaming que se adapta a la deriva.

Los Módulos 5 y 6 llegaron a la misma conclusión por dos caminos distintos: **ningún umbral
fijo aprendido del pasado sobrevive**. El cuantil de entrenamiento de Deep SVDD prometía
0,1% de falsas alarmas y entregaba 35%, y el p-valor conforme —que arregla la calibración
cuando hay intercambiabilidad— se rompe exactamente igual cuando la distribución se corre.
Ninguno de los dos módulos propuso el detector que sí se adapta. Este lo hace.

Los quince detectores anteriores comparten un supuesto: se ajustan una vez y se usan para
siempre. Half-Space Trees (Tan, Ting y Liu, 2011) está diseñado al revés — para operar sobre
un flujo, manteniendo un **perfil de masa** de una ventana reciente que se refresca sola.
Cuando el tráfico cambia, el detector cambia con él sin reentrenar nada.

La idea es astuta y barata: los árboles se construyen **antes de ver un solo dato**. Cada
nodo parte el espacio por la mitad en una dimensión al azar, así que la estructura no depende
de la muestra; lo único que se aprende es cuántos puntos de la ventana caen en cada nodo. Un
punto que aterriza en una región de masa baja es anómalo. Actualizar el modelo es recontar
—una pasada lineal— en vez de reajustar.

Esa construcción ciega tiene una consecuencia que conviene entender: sin datos que guíen los
cortes, un solo árbol es un detector pésimo. La calidad sale del ensemble, igual que en LODA.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


class HalfSpaceTrees:
    """Ensemble de árboles de mitades con perfil de masa sobre una ventana deslizante.

    A diferencia del resto del repositorio, este detector expone además `update_window`, que
    reemplaza el perfil de masa por el de datos más recientes **sin tocar la estructura de
    los árboles**. Es lo que lo vuelve utilizable en un flujo: adaptarse cuesta una pasada
    lineal, no un reajuste.

    `size_limit` implementa el corte del paper: la búsqueda se detiene en el primer nodo cuya
    masa cae por debajo del límite, en vez de bajar siempre hasta la hoja. Sin ese corte, en
    regiones poco pobladas todos los nodos profundos tienen masa cero y el score deja de
    distinguir entre "raro" y "rarísimo".
    """

    def __init__(self, n_trees: int = 25, max_depth: int = 8, size_limit: int = 5,
                 random_state: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.size_limit = size_limit
        self.random_state = random_state

        self.split_dim_: np.ndarray | None = None
        self.split_val_: np.ndarray | None = None
        self.mass_: np.ndarray | None = None
        self.data_min_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None

    # ------------------------------------------------------------------ estructura

    def _build_structure(self, n_features: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """Construye los cortes de todos los árboles sin mirar los datos.

        Devuelve dos matrices (n_trees, n_nodos_internos): dimensión y valor de corte. Los
        nodos se numeran como un árbol binario completo — el hijo izquierdo de `i` es
        `2i+1` y el derecho `2i+2` — lo que permite recorrer todos los puntos a la vez con
        aritmética de índices en vez de recursión.
        """
        n_internal = 2 ** self.max_depth - 1
        split_dim = np.zeros((self.n_trees, n_internal), dtype=np.int64)
        split_val = np.zeros((self.n_trees, n_internal), dtype=float)

        for tree in range(self.n_trees):
            # Espacio de trabajo inicial, sobre datos ya normalizados a [0, 1]. La receta es
            # la del paper: un punto q al azar y un rango simétrico lo bastante ancho como
            # para cubrir el dominio, de modo que árboles distintos partan por lugares
            # distintos aunque la estructura no dependa de la muestra.
            q = rng.uniform(0.0, 1.0, size=n_features)
            half = 2.0 * np.maximum(q, 1.0 - q)
            lower = q - half
            upper = q + half

            # Recorrido por niveles: cada nodo hereda el espacio de su padre partido al medio.
            bounds = {0: (lower, upper)}
            for node in range(n_internal):
                node_lower, node_upper = bounds[node]
                dim = int(rng.integers(n_features))
                midpoint = 0.5 * (node_lower[dim] + node_upper[dim])

                split_dim[tree, node] = dim
                split_val[tree, node] = midpoint

                left_upper = node_upper.copy()
                left_upper[dim] = midpoint
                right_lower = node_lower.copy()
                right_lower[dim] = midpoint

                for child, child_bounds in (
                    (2 * node + 1, (node_lower, left_upper)),
                    (2 * node + 2, (right_lower, node_upper)),
                ):
                    if child < n_internal:
                        bounds[child] = child_bounds
        return split_dim, split_val

    # ------------------------------------------------------------------ recorrido

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.data_min_) / self.data_range_

    def _traverse(self, X_norm: np.ndarray, tree: int) -> np.ndarray:
        """Índice de nodo alcanzado por cada punto en cada nivel, de forma (n_filas, profundidad+1)."""
        node = np.zeros(X_norm.shape[0], dtype=np.int64)
        path = np.empty((X_norm.shape[0], self.max_depth + 1), dtype=np.int64)
        path[:, 0] = node

        n_internal = 2 ** self.max_depth - 1
        for depth in range(self.max_depth):
            dims = self.split_dim_[tree, node]
            vals = self.split_val_[tree, node]
            go_left = X_norm[np.arange(X_norm.shape[0]), dims] < vals
            node = np.where(go_left, 2 * node + 1, 2 * node + 2)
            # Los nodos del último nivel no son internos; se recortan para poder indexarlos
            # en el arreglo de masas sin salirse.
            node = np.minimum(node, 2 ** (self.max_depth + 1) - 2)
            path[:, depth + 1] = node
            if depth + 1 >= self.max_depth:
                break
            node = np.minimum(node, n_internal - 1)
        return path

    def _mass_profile(self, X_norm: np.ndarray) -> np.ndarray:
        """Cuenta cuántos puntos de la ventana alcanzan cada nodo de cada árbol."""
        n_nodes = 2 ** (self.max_depth + 1) - 1
        mass = np.zeros((self.n_trees, n_nodes), dtype=float)

        for tree in range(self.n_trees):
            path = self._traverse(X_norm, tree)
            for depth in range(self.max_depth + 1):
                counts = np.bincount(path[:, depth], minlength=n_nodes)
                mass[tree] += counts
        return mass

    # ------------------------------------------------------------------ API pública

    def fit(self, X) -> "HalfSpaceTrees":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # La normalización usa el rango de la ventana inicial. Es deliberado: al actualizar
        # la ventana se conserva, de modo que un corrimiento del tráfico se refleja en la
        # masa (que es lo que se quiere medir) y no se absorbe silenciosamente en la escala.
        self.data_min_ = X.min(axis=0)
        self.data_range_ = np.ptp(X, axis=0) + EPS

        self.split_dim_, self.split_val_ = self._build_structure(X.shape[1], rng)
        self.mass_ = self._mass_profile(self._normalize(X))
        return self

    def update_window(self, X) -> "HalfSpaceTrees":
        """Reemplaza el perfil de masa con el de una ventana nueva, sin reconstruir árboles.

        Es la operación que distingue a este detector de los otros quince: adaptarse a un
        cambio de distribución cuesta una pasada sobre la ventana nueva.
        """
        if self.split_dim_ is None:
            raise ValueError("Hay que llamar a fit() antes de actualizar la ventana.")
        self.mass_ = self._mass_profile(self._normalize(np.asarray(X, dtype=float)))
        return self

    def score_samples(self, X) -> np.ndarray:
        """Score de masa acumulada. Más bajo = más anómalo, como el resto del repositorio."""
        X_norm = self._normalize(np.asarray(X, dtype=float))
        total = np.zeros(X_norm.shape[0])

        for tree in range(self.n_trees):
            path = self._traverse(X_norm, tree)
            masses = self.mass_[tree][path]

            # Corte del paper: se detiene en el primer nodo con masa por debajo del límite.
            below = masses <= self.size_limit
            first_below = np.where(below.any(axis=1), below.argmax(axis=1), self.max_depth)

            rows = np.arange(X_norm.shape[0])
            total += masses[rows, first_below] * (2.0 ** first_below)

        return total


def build_hs_trees(**kwargs) -> HalfSpaceTrees:
    """Instancia Half-Space Trees con la configuración por defecto del módulo."""
    return HalfSpaceTrees(**kwargs)

"""Módulo 8 — por qué se disparó esta alerta.

Un analista que recibe una transacción marcada necesita saber qué la marcó. Sin eso la alerta
no es accionable: no se puede confirmar, no se puede descartar rápido y no se puede explicar
a un cliente que reclama.

De los dieciséis detectores, solo LODA trae atribución propia (Módulo 6) y HBOS y ECOD la
admitirían por construcción. Los otros trece no. Esta implementación es **agnóstica al
modelo**: no mira el interior del detector, solo lo consulta.

El método es oclusión. Se reemplaza una feature por valores típicos y se vuelve a puntuar. Si
el score se desploma, esa feature sostenía la anomalía; si no se mueve, no aportaba nada.

La versión ingenua de eso **no funciona sobre PaySim**, y encontrarlo costó dos intentos:

1. Ocluir contra la mediana del entrenamiento dio aportes de -397.000 para las columnas que
   más pesan — o sea "quitar esta columna vuelve la transacción mucho más rara", que no
   explica nada.
2. Promediar sobre un fondo de cincuenta filas normales, la corrección estándar que hace
   SHAP, tampoco lo arregló: los negativos crecieron a -909.000.

La causa no era contra qué se sustituye sino **que se sustituye de a una columna**. Las
features de PaySim son linealmente dependientes por construcción: `errorBalanceOrig` es
exactamente `newbalanceOrig + amount - oldbalanceOrg`, y las cinco dummies de `type_*` suman
1. Mover una sola produce saldos que no cierran o una transacción sin tipo, y un modelo de
densidad manda ese punto imposible a una región de probabilidad casi nula.

La solución es ocluir por **grupos** de columnas dependientes (`infer_dependency_groups`), lo
que mantiene la consistencia interna a costa de una explicación más gruesa. Los dos límites
que siguen en pie:

- **Mide contribución marginal, no causalidad**, igual que cualquier método de permutación.
- **La atribución es por grupo, no por columna.** Todas las del grupo se movieron juntas, así
  que repartir el efecto entre ellas sería inventar una precisión que los datos no permiten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def infer_dependency_groups(feature_names: list[str]) -> list[list[int]]:
    """Agrupa las columnas que no pueden moverse por separado sin producir un punto imposible.

    Sobre PaySim quedan tres grupos:

    - **monetario**: montos, saldos, sus discrepancias y los indicadores de saldo en cero.
      Van todas juntas porque `amount` aparece en las dos fórmulas de error, así que enlaza el
      lado de origen con el de destino y no admite separarlos.
    - **tipo de transacción**: las cinco dummies de `type_*`, que suman 1. Mover una sola
      produce una transacción sin tipo, o con dos.
    - **tiempo**: `step`, que no participa de ninguna restricción.

    La granularidad es gruesa a propósito. Podría afinarse con columnas independientes entre
    sí, pero acá "la alerta la disparó el patrón de saldos" es una explicación honesta y
    accionable, mientras que atribuirla a `newbalanceOrig` en particular sería inventar una
    precisión que la estructura de los datos no permite.
    """
    monetario, tipo, resto = [], [], []
    for j, nombre in enumerate(feature_names):
        if nombre.startswith("type_"):
            tipo.append(j)
        elif any(clave in nombre for clave in ("amount", "balance", "Balance")):
            monetario.append(j)
        else:
            resto.append(j)

    return [g for g in (monetario, tipo, *[[j] for j in resto]) if g]


def occlusion_attribution(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str] | None = None,
    groups: list[list[int]] | None = None,
) -> np.ndarray:
    """Cuánto cae el anomaly score al reemplazar cada feature —o grupo— por valores típicos.

    Devuelve una matriz (n_filas, n_features) donde el valor positivo significa que esa
    columna estaba **sosteniendo** la anomalía. Un valor negativo significa lo contrario: sin
    esa columna la transacción se vería aún más rara.

    `baseline` puede ser un vector de valores típicos o una **matriz de fondo** con varias
    filas normales; con fondo se promedia sobre sustituciones que sí ocurren en los datos, la
    misma corrección que hace SHAP al integrar sobre una distribución de referencia.

    `groups` es lo que decide si la explicación sirve, y sobre PaySim no es opcional. Las
    features del dataset son **linealmente dependientes por construcción**:
    `errorBalanceOrig` es exactamente `newbalanceOrig + amount - oldbalanceOrg`, y las cinco
    dummies de `type_*` suman 1. Ocluir una sola columna de un conjunto así produce siempre un
    punto imposible —saldos que no cierran, una transacción sin tipo— y un modelo de densidad
    lo manda a una región de probabilidad casi nula.

    El síntoma es inconfundible y aparece en el Módulo 8: la atribución de las columnas que
    más pesan sale negativa y enorme (-909.000 para `newbalanceOrig` con Gaussian Mixture),
    o sea "quitar esta columna vuelve la transacción muchísimo más rara", que no explica nada.
    Cambiar la mediana por un fondo de cincuenta filas no lo corrige: el problema no es contra
    qué se sustituye sino que se sustituye de a una. Ocluir el grupo entero mantiene la
    consistencia interna y devuelve una atribución legible.

    Sin `groups` cada feature es su propio grupo, que es el comportamiento clásico y el
    correcto cuando las columnas son independientes.
    """
    X_scaled = np.asarray(X_scaled, dtype=float)
    fondo = np.atleast_2d(np.asarray(baseline, dtype=float))

    if fondo.shape[1] != X_scaled.shape[1]:
        raise ValueError(
            f"La línea de base tiene {fondo.shape[1]} valores y los datos "
            f"{X_scaled.shape[1]} columnas."
        )
    if feature_names is not None and len(feature_names) != X_scaled.shape[1]:
        raise ValueError(
            f"Se recibieron {len(feature_names)} nombres para {X_scaled.shape[1]} columnas."
        )

    if groups is None:
        groups = [[j] for j in range(X_scaled.shape[1])]
    else:
        cubiertas = [j for grupo in groups for j in grupo]
        if sorted(cubiertas) != list(range(X_scaled.shape[1])):
            raise ValueError(
                "Los grupos deben cubrir cada columna exactamente una vez; se recibieron "
                f"{len(cubiertas)} índices para {X_scaled.shape[1]} columnas."
            )

    score_original = np.asarray(score_fn(X_scaled), dtype=float)
    atribucion = np.zeros_like(X_scaled)

    for grupo in groups:
        acumulado = np.zeros(X_scaled.shape[0])
        for fila_fondo in fondo:
            ocluido = X_scaled.copy()
            ocluido[:, grupo] = fila_fondo[grupo]
            acumulado += np.asarray(score_fn(ocluido), dtype=float)

        # Todas las columnas del grupo reciben el mismo aporte: se movieron juntas, así que
        # no hay forma de repartir el efecto entre ellas sin inventar una atribución.
        atribucion[:, grupo] = (score_original - acumulado / len(fondo))[:, None]

    return atribucion


def explain_rows(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
    groups: list[list[int]] | None = None,
) -> list[list[tuple[str, float]]]:
    """Los `top_k` grupos más responsables de cada fila, ordenados de mayor a menor aporte.

    Con grupos, todas las columnas de uno comparten el mismo aporte, así que se devuelve una
    entrada por grupo en vez de repetir el mismo número tres veces. El nombre del grupo son
    sus columnas unidas por `+`, acortado cuando son muchas.
    """
    atribucion = occlusion_attribution(score_fn, X_scaled, baseline, feature_names, groups)
    if groups is None:
        groups = [[j] for j in range(atribucion.shape[1])]

    etiquetas = [_group_label(grupo, feature_names) for grupo in groups]
    representantes = [grupo[0] for grupo in groups]

    explicaciones = []
    for fila in atribucion:
        aportes = fila[representantes]
        orden = np.argsort(aportes)[::-1][:top_k]
        explicaciones.append([(etiquetas[g], float(aportes[g])) for g in orden])
    return explicaciones


def _group_label(grupo: list[int], feature_names: list[str], max_nombres: int = 2) -> str:
    """Nombre legible de un grupo: sus columnas, acortadas si son más de `max_nombres`."""
    nombres = [feature_names[j] for j in grupo]
    if len(nombres) <= max_nombres:
        return "+".join(nombres)
    return f"{nombres[0]}+{len(nombres) - 1} más"


def attribution_frame(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str],
    groups: list[list[int]] | None = None,
) -> pd.DataFrame:
    """La matriz de atribución como DataFrame, con las features como columnas."""
    return pd.DataFrame(
        occlusion_attribution(score_fn, X_scaled, baseline, feature_names, groups),
        columns=list(feature_names),
    )


def global_importance(
    score_fn,
    X_scaled: np.ndarray,
    baseline: np.ndarray,
    feature_names: list[str],
    groups: list[list[int]] | None = None,
) -> pd.DataFrame:
    """Aporte promedio de cada grupo sobre un conjunto de transacciones.

    Promediar la atribución absoluta sobre muchas alertas da una lectura global: en qué se
    apoya el detector en general, no solo en un caso. Útil para detectar que un detector
    depende de un único grupo, que es una fragilidad operativa aunque la métrica sea buena.

    Devuelve una fila por grupo, no por columna: como todas las columnas de un grupo comparten
    aporte, listarlas por separado repetiría el mismo número tantas veces como columnas tenga
    y escondería los grupos chicos debajo.

    Conviene calcularlo sobre **alertas**, no sobre tráfico al azar. Sobre transacciones
    normales el aporte sale negativo —sustituir un patrón normal por otro produce un híbrido
    menos típico— y eso no dice nada sobre qué dispara las alertas.
    """
    atribucion = occlusion_attribution(score_fn, X_scaled, baseline, feature_names, groups)
    if groups is None:
        groups = [[j] for j in range(atribucion.shape[1])]

    representantes = [grupo[0] for grupo in groups]
    tabla = pd.DataFrame({
        "grupo": [_group_label(grupo, feature_names) for grupo in groups],
        "columnas": [len(grupo) for grupo in groups],
        "aporte_medio": atribucion[:, representantes].mean(axis=0),
        "aporte_absoluto_medio": np.abs(atribucion[:, representantes]).mean(axis=0),
    })
    return tabla.sort_values("aporte_absoluto_medio", ascending=False).reset_index(drop=True)

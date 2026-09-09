"""Módulo 7 — apilado semi-supervisado: qué hacer cuando llegan unas pocas etiquetas.

El repositorio vive en dos extremos. El Módulo 1 es supervisado y usa las 6,3 millones de
etiquetas del dataset; los Módulos 2 a 6 son no supervisados y no usan ninguna. El caso real
está en el medio y no se parece a ninguno de los dos: un equipo de fraude tiene un puñado de
casos confirmados —los que alcanzó a revisar— y millones de transacciones sin revisar.

Este módulo cubre ese caso con la idea de XGBOD (Zhao y Hryniewicki, 2018): usar los scores
de los quince detectores no supervisados como **features adicionales** de un clasificador
supervisado. Los detectores ya destilaron la estructura de "lo normal" sin gastar una sola
etiqueta; el clasificador solo tiene que aprender a combinarlos, que es un problema mucho
más chico y por lo tanto necesita muchísimas menos etiquetas.

Dos detalles que hacen la comparación honesta:

- **No se normalizan los scores.** Los árboles de decisión son invariantes a transformaciones
  monótonas, así que rank-normalizar no cambiaría nada; hacerlo sugeriría un cuidado que no
  aporta.
- **El presupuesto de etiquetado se gasta como en la vida real.** Etiquetar 50 transacciones
  al azar sobre una prevalencia del 0,08% da 0,04 fraudes esperados: nada con qué entrenar.
  Un equipo real etiqueta la cola de alertas que revisa, así que el conjunto etiquetado está
  sesgado hacia scores altos — y ese sesgo es lo que lo vuelve utilizable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Transacciones sin revisar que se agregan como negativas. Un equipo trata como legítimo
# todo lo que no alcanzó a mirar; una fracción diminuta (~0,08%) es fraude no detectado, y
# ese ruido de etiqueta forma parte del problema real.
BACKGROUND_SIZE = 50_000


def stack_scores(X, scores_by_detector: dict[str, np.ndarray]) -> np.ndarray:
    """Concatena las features originales con el score de cada detector, en orden estable."""
    base = np.asarray(X, dtype=float)
    if not scores_by_detector:
        return base

    extra = np.column_stack([scores_by_detector[name] for name in sorted(scores_by_detector)])
    if extra.shape[0] != base.shape[0]:
        raise ValueError(
            f"Los scores deben cubrir las mismas filas que X; {extra.shape[0]} vs {base.shape[0]}"
        )
    return np.column_stack([base, extra])


def select_for_review(scores: np.ndarray, budget: int, strategy: str = "top",
                      random_state: int = 42) -> np.ndarray:
    """Índices de las transacciones que el equipo alcanza a revisar con su presupuesto.

    - `top`: las de score más alto, que es el orden en que se trabaja una cola de alertas;
    - `random`: una muestra al azar, incluida como contraste — sobre una prevalencia del
      0,08% casi no devuelve positivos, y esa es exactamente la lección.
    """
    scores = np.asarray(scores, dtype=float)
    budget = min(budget, len(scores))

    if strategy == "top":
        return np.argsort(scores)[::-1][:budget]
    if strategy == "random":
        rng = np.random.default_rng(random_state)
        return rng.choice(len(scores), size=budget, replace=False)
    raise ValueError(f"Estrategia desconocida: {strategy}. Usar 'top' o 'random'.")


def build_training_set(
    X_pool: np.ndarray,
    y_pool,
    reviewed_idx: np.ndarray,
    background_size: int = BACKGROUND_SIZE,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Arma el conjunto de entrenamiento: lo revisado con su etiqueta real, más fondo negativo.

    El fondo son transacciones no revisadas que se asumen legítimas. Es lo que hace un equipo
    real, y arrastra el ruido de etiqueta correspondiente: alguna de ellas es fraude que
    nadie miró.
    """
    y_arr = np.asarray(y_pool)
    rng = np.random.default_rng(random_state)

    unreviewed = np.setdiff1d(np.arange(len(y_arr)), reviewed_idx, assume_unique=False)
    background = rng.choice(unreviewed, size=min(background_size, len(unreviewed)), replace=False)

    idx = np.concatenate([reviewed_idx, background])
    labels = np.concatenate([y_arr[reviewed_idx], np.zeros(len(background), dtype=int)])
    return X_pool[idx], labels


def fit_classifier(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """XGBoost con peso de clase, o `None` si el conjunto no tiene ambas clases.

    Con presupuestos muy chicos y muestreo al azar puede no aparecer un solo positivo. Eso no
    es un error a silenciar: es el resultado, y el llamador debe poder reportarlo.
    """
    from xgboost import XGBClassifier

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def label_budget_curve(
    X_pool_raw: np.ndarray,
    X_pool_stacked: np.ndarray,
    y_pool,
    X_test_raw: np.ndarray,
    X_test_stacked: np.ndarray,
    y_test,
    ranking_scores: np.ndarray,
    budgets=(50, 100, 200, 500, 1_000, 5_000),
    strategies=("top", "random"),
    random_state: int = 42,
) -> pd.DataFrame:
    """PR-AUC en el período tardío según cuántas etiquetas hay y cómo se eligieron.

    `ranking_scores` es el score no supervisado con el que se ordena la cola de revisión: es
    lo que el equipo tiene disponible **antes** de contar con una sola etiqueta.
    """
    rows = []
    for strategy in strategies:
        for budget in budgets:
            reviewed = select_for_review(ranking_scores, budget, strategy, random_state)
            n_fraud = int(np.asarray(y_pool)[reviewed].sum())

            for nombre, X_pool, X_test in (
                ("solo features", X_pool_raw, X_test_raw),
                ("features + scores", X_pool_stacked, X_test_stacked),
            ):
                X_train, y_train = build_training_set(X_pool, y_pool, reviewed,
                                                      random_state=random_state)
                model = fit_classifier(X_train, y_train, random_state)

                if model is None:
                    rows.append({
                        "estrategia": strategy, "presupuesto": budget,
                        "fraudes_etiquetados": n_fraud, "features": nombre,
                        "pr_auc": np.nan, "roc_auc": np.nan,
                    })
                    continue

                proba = model.predict_proba(X_test)[:, 1]
                rows.append({
                    "estrategia": strategy, "presupuesto": budget,
                    "fraudes_etiquetados": n_fraud, "features": nombre,
                    "pr_auc": average_precision_score(y_test, proba),
                    "roc_auc": roc_auc_score(y_test, proba),
                })
    return pd.DataFrame(rows)

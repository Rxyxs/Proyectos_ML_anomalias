"""Confirma que el benchmark prioriza PR-AUC sobre ROC-AUC bajo desbalance.

Con 99% inliers / 1% outliers, ROC-AUC puede ser optimista: promedia sobre TODOS los
pares negativo/positivo, así que un puñado de falsos positivos con score alto (que
arruinan la precisión justo donde un analista empezaría a revisar) apenas lo mueven.
PR-AUC sí lo nota, porque pondera la precisión en la cabeza del ranking. Esta prueba
no confía en la intuición: construye dos "detectores" sintéticos cuyo ROC-AUC y PR-AUC
apuntan a ganadores *distintos*, y confirma que `summary_table` elige el que gana en
PR-AUC -- no el que gana en ROC-AUC.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.unsupervised.benchmark import evaluate_all, summary_table
from src.unsupervised.train_unsupervised import evaluate

N_NEG, N_POS = 990, 10


def _y_test() -> np.ndarray:
    return np.array([0] * N_NEG + [1] * N_POS)


def _scores_alto_roc_bajo_pr() -> np.ndarray:
    """ROC-AUC alto: separa bien 'en promedio'. PR-AUC bajo: 43 negativos
    'impostores' se concentran muy por encima de los positivos, ocupando la
    cabeza del ranking -- son 43 de 990 negativos, así que apenas mueven un
    ROC-AUC que promedia sobre todos los pares, pero hunden la precisión en
    los primeros lugares."""
    rng = np.random.default_rng(2558)
    return np.concatenate([
        rng.normal(-0.74, 1.26, N_NEG - 43),
        rng.normal(4.89, 0.24, 43),
        rng.normal(1.81, 0.56, N_POS),
    ])


def _scores_bajo_roc_alto_pr() -> np.ndarray:
    """ROC-AUC más bajo: separación pareja y mediocre en todo el rango (muchos
    pares mal ordenados en el cuerpo de la distribución). PR-AUC alto: sin
    impostores concentrados en la cabeza, la precisión en los primeros lugares
    se mantiene alta."""
    rng = np.random.default_rng(1950)
    return np.concatenate([
        rng.normal(0.03, 0.38, N_NEG - 34),
        rng.normal(0.62, 0.10, 34),
        rng.normal(2.12, 2.49, N_POS),
    ])


def test_la_construccion_sintetica_realmente_diverge():
    """Chequeo previo: si esto no fuera cierto, el resto del archivo probaría
    algo vacío -- no que summary_table elige bien, sino que ambos criterios
    coinciden y da lo mismo cuál se use."""
    y = _y_test()
    roc_a = roc_auc_score(y, _scores_alto_roc_bajo_pr())
    roc_b = roc_auc_score(y, _scores_bajo_roc_alto_pr())
    pr_a = average_precision_score(y, _scores_alto_roc_bajo_pr())
    pr_b = average_precision_score(y, _scores_bajo_roc_alto_pr())

    assert roc_a > roc_b, "el detector A debe ganar en ROC-AUC"
    assert pr_b > pr_a, "el detector B debe ganar en PR-AUC"


def test_summary_table_ordena_por_pr_auc_no_por_roc_auc():
    y = _y_test()
    outputs = {
        "alto_roc_bajo_pr": {
            "scores": _scores_alto_roc_bajo_pr(), "fit_seconds": 0.0, "score_seconds": 0.0,
        },
        "bajo_roc_alto_pr": {
            "scores": _scores_bajo_roc_alto_pr(), "fit_seconds": 0.0, "score_seconds": 0.0,
        },
    }

    resultados = evaluate_all(outputs, y)
    tabla = summary_table(resultados)

    # Si summary_table ordenara por ROC-AUC, "alto_roc_bajo_pr" saldría primero.
    assert tabla.iloc[0]["pr_auc"] > tabla.iloc[1]["pr_auc"]
    assert tabla.iloc[0]["roc_auc"] < tabla.iloc[1]["roc_auc"], (
        "el ganador de la tabla tiene que ser el de mejor PR-AUC aunque tenga "
        "peor ROC-AUC -- si no, el benchmark estaría priorizando la métrica "
        "optimista bajo desbalance"
    )
    assert list(tabla["pr_auc"]) == sorted(tabla["pr_auc"], reverse=True)


def test_evaluate_incluye_precision_at_k_ademas_de_pr_auc():
    y = _y_test()
    metricas = evaluate(y, _scores_bajo_roc_alto_pr())

    assert "pr_auc" in metricas
    assert set(metricas["precision_recall_at_k"]) == {50, 100, 200}
    for k, (precision, recall) in metricas["precision_recall_at_k"].items():
        assert 0.0 <= precision <= 1.0, k
        assert 0.0 <= recall <= 1.0, k

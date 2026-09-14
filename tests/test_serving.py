"""Pruebas del paquete de scoring y la explicación de alertas (src.serving).

El modo de falla que más importa acá no produce una excepción: un DataFrame con las mismas
columnas en otro orden devuelve scores perfectamente plausibles y completamente equivocados.
Varias pruebas apuntan justo a eso.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.serving.explain import (
    attribution_frame,
    explain_rows,
    global_importance,
    occlusion_attribution,
)
from src.serving.package import DetectorPackage, build_package
from src.serving.predict import alert_rate, score_transactions, top_alerts
from src.unsupervised.families import GMMDensity

FEATURES = ["monto", "saldo", "error", "hora"]


@pytest.fixture
def paquete():
    """Paquete ajustado sobre una nube normal, con calibración retenida y disjunta."""
    rng = np.random.default_rng(42)
    entrenamiento = rng.normal(size=(1_500, 4))
    calibracion = rng.normal(size=(1_500, 4))

    scaler = RobustScaler().fit(entrenamiento)
    detector = GMMDensity(n_components=3).fit(scaler.transform(entrenamiento))

    return build_package(
        detector, scaler, scaler.transform(calibracion), FEATURES,
        alpha=0.01, detector_name="gmm_prueba",
    )


def _frame(filas) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(filas, dtype=float), columns=FEATURES)


# ---------------------------------------------------------------- contrato de columnas

def test_reordenar_columnas_no_cambia_el_resultado(paquete):
    """El paquete reordena por nombre, así que el orden de entrada es irrelevante.

    Sin esta garantía, un consumidor que arme las columnas en otro orden obtendría scores
    plausibles y equivocados, sin ninguna excepción que lo delate.
    """
    normal = _frame([[0.5, -0.2, 0.1, 0.3]])
    revuelto = normal[["hora", "error", "monto", "saldo"]]

    np.testing.assert_allclose(paquete.score(normal), paquete.score(revuelto))


def test_rechaza_una_columna_faltante(paquete):
    with pytest.raises(ValueError, match="Faltan columnas"):
        paquete.score(_frame([[0.5, -0.2, 0.1, 0.3]]).drop(columns=["saldo"]))


def test_rechaza_una_columna_inesperada(paquete):
    entrada = _frame([[0.5, -0.2, 0.1, 0.3]]).assign(extra=1.0)
    with pytest.raises(ValueError, match="Columnas inesperadas"):
        paquete.score(entrada)


def test_rechaza_un_arreglo_con_otra_cantidad_de_columnas(paquete):
    with pytest.raises(ValueError, match="Se esperaban 4 columnas"):
        paquete.score(np.zeros((3, 6)))


def test_acepta_un_arreglo_con_la_forma_correcta(paquete):
    assert paquete.score(np.zeros((3, 4))).shape == (3,)


# ---------------------------------------------------------------- scoring y umbral

def test_puntua_mas_alto_una_transaccion_extrema(paquete):
    scores = paquete.score(_frame([[0.0, 0.0, 0.0, 0.0], [12.0, 12.0, 12.0, 12.0]]))
    assert scores[1] > scores[0]


def test_el_p_valor_es_mas_bajo_cuanto_mas_anomalo(paquete):
    p = paquete.p_values(_frame([[0.0, 0.0, 0.0, 0.0], [12.0, 12.0, 12.0, 12.0]]))
    assert p[1] < p[0]
    assert (p > 0).all() and (p <= 1).all()


def test_el_umbral_reproduce_la_decision_por_p_valor(paquete):
    rng = np.random.default_rng(7)
    datos = _frame(rng.normal(size=(400, 4)))

    np.testing.assert_array_equal(
        paquete.predict(datos), paquete.score(datos) >= paquete.threshold
    )


def test_la_tasa_de_alerta_se_acerca_a_alpha_sin_deriva(paquete):
    """Sobre tráfico de la misma distribución, el paquete debe cumplir lo que promete."""
    rng = np.random.default_rng(99)
    legitimas = _frame(rng.normal(size=(4_000, 4)))

    assert alert_rate(paquete, legitimas) == pytest.approx(paquete.alpha, abs=0.01)


# ---------------------------------------------------------------- persistencia

def test_el_paquete_sobrevive_a_guardar_y_cargar(paquete, tmp_path):
    datos = _frame([[0.4, 0.1, -0.3, 0.2], [9.0, 9.0, 9.0, 9.0]])
    antes = paquete.score(datos)

    ruta = paquete.save(tmp_path / "paquete.joblib")
    recargado = DetectorPackage.load(ruta)

    np.testing.assert_allclose(recargado.score(datos), antes)
    assert recargado.feature_names == paquete.feature_names
    assert recargado.alpha == paquete.alpha


def test_el_paquete_guarda_su_procedencia(paquete):
    assert paquete.metadata["detector"] == "gmm_prueba"
    assert paquete.metadata["n_calibration"] == 1_500
    assert paquete.metadata["n_features"] == 4
    assert "created_at" in paquete.metadata


# ---------------------------------------------------------------- explicación

def test_la_oclusion_senala_la_feature_responsable(paquete):
    """Solo la tercera columna es anómala; su oclusión debe ser la que más baje el score."""
    punto = np.zeros((1, 4))
    punto[0, 2] = 10.0
    escalado = paquete.scaler.transform(punto)

    aporte = occlusion_attribution(paquete.score_from_scaled, escalado, np.zeros(4), FEATURES)
    assert int(np.argmax(aporte[0])) == 2


def test_la_explicacion_devuelve_las_top_k_ordenadas(paquete):
    punto = np.zeros((1, 4))
    punto[0, 1] = 9.0
    escalado = paquete.scaler.transform(punto)

    motivos = explain_rows(paquete.score_from_scaled, escalado, np.zeros(4), FEATURES, top_k=2)

    assert len(motivos) == 1 and len(motivos[0]) == 2
    assert motivos[0][0][0] == "saldo"
    assert motivos[0][0][1] >= motivos[0][1][1], "debe venir ordenado de mayor a menor"


def test_la_oclusion_rechaza_una_linea_de_base_de_otro_largo(paquete):
    with pytest.raises(ValueError, match="línea de base"):
        occlusion_attribution(paquete.score_from_scaled, np.zeros((2, 4)), np.zeros(7), FEATURES)


def test_la_oclusion_rechaza_nombres_de_otro_largo(paquete):
    with pytest.raises(ValueError, match="nombres"):
        occlusion_attribution(paquete.score_from_scaled, np.zeros((2, 4)), np.zeros(4), ["a", "b"])


def test_la_atribucion_en_dataframe_conserva_los_nombres(paquete):
    tabla = attribution_frame(paquete.score_from_scaled, np.zeros((3, 4)), np.zeros(4), FEATURES)
    assert list(tabla.columns) == FEATURES
    assert len(tabla) == 3


def test_la_importancia_global_ordena_por_aporte_absoluto(paquete):
    rng = np.random.default_rng(3)
    datos = rng.normal(size=(50, 4))
    datos[:, 0] *= 8.0  # la primera columna es la que manda

    tabla = global_importance(paquete.score_from_scaled, datos, np.zeros(4), FEATURES)

    assert list(tabla["aporte_absoluto_medio"]) == sorted(
        tabla["aporte_absoluto_medio"], reverse=True
    )
    assert tabla.iloc[0]["grupo"] == "monto"


# ---------------------------------------------------------------- API de alertas

def test_el_scoring_devuelve_una_fila_por_transaccion_con_motivo(paquete):
    datos = _frame([[0.1, 0.2, 0.0, -0.1], [11.0, 0.0, 0.0, 0.0]])
    salida = score_transactions(paquete, datos)

    assert list(salida.columns) == ["score", "p_valor", "alerta", "motivo"]
    assert len(salida) == 2
    assert salida["motivo"].str.len().gt(0).all()


def test_se_puede_saltear_la_explicacion(paquete):
    salida = score_transactions(paquete, _frame([[0.1, 0.2, 0.0, -0.1]]), explain=False)
    assert "motivo" not in salida.columns


def test_las_alertas_principales_vienen_ordenadas_por_p_valor(paquete):
    rng = np.random.default_rng(5)
    datos = _frame(np.vstack([rng.normal(size=(200, 4)), np.full((3, 4), 15.0)]))

    alertas = top_alerts(paquete, datos, n=5)

    assert len(alertas) == 5
    assert alertas["p_valor"].is_monotonic_increasing


def test_el_paquete_trae_su_propio_fondo(paquete):
    assert paquete.background is not None
    assert paquete.background.shape == (50, 4)
    assert paquete.metadata["n_background"] == 50


def test_la_oclusion_acepta_un_fondo_de_varias_filas(paquete):
    """Promediar sobre varias filas normales, no sobre un unico punto tipico."""
    rng = np.random.default_rng(0)
    fondo = rng.normal(size=(10, 4))
    punto = np.zeros((1, 4))
    punto[0, 2] = 10.0

    aporte = occlusion_attribution(paquete.score_from_scaled, punto, fondo, FEATURES)

    assert aporte.shape == (1, 4)
    assert np.isfinite(aporte).all()


def test_el_fondo_da_una_atribucion_distinta_que_un_punto_unico(paquete):
    """Es la correccion que vuelve util a la explicacion.

    Ocluir contra un unico punto tipico rompe las correlaciones entre columnas; sobre un
    modelo de densidad eso manda la transaccion a una region de probabilidad casi nula y
    produce aportes negativos gigantes que no explican nada. Promediar sobre filas normales
    reales mantiene combinaciones que si ocurren.
    """
    punto = np.zeros((1, 4))
    punto[0, 0] = 8.0

    con_punto = occlusion_attribution(paquete.score_from_scaled, punto, np.zeros(4), FEATURES)
    con_fondo = occlusion_attribution(
        paquete.score_from_scaled, punto, paquete.background, FEATURES
    )

    assert not np.allclose(con_punto, con_fondo)


def test_la_explicacion_de_una_alerta_usa_el_fondo_del_paquete(paquete):
    """Sin fondo la atribucion caeria al vector cero, y el resultado seria otro."""
    from src.serving.predict import _baseline_from_package

    fondo = _baseline_from_package(paquete)
    assert fondo.ndim == 2 and fondo.shape[0] == 50

    sin_fondo = DetectorPackage(
        detector=paquete.detector, scaler=paquete.scaler,
        calibration_scores=paquete.calibration_scores,
        feature_names=paquete.feature_names, alpha=paquete.alpha, background=None,
    )
    assert _baseline_from_package(sin_fondo).shape == (4,)


# ---------------------------------------------------------------- grupos dependientes

PAYSIM_COLS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "errorBalanceOrig", "errorBalanceDest", "origBalanceZero", "destBalanceZero",
]


def test_los_grupos_de_paysim_respetan_las_dependencias():
    """Monetario junto, dummies de tipo juntas, step solo.

    Las columnas monetarias van todas en un grupo porque `amount` aparece en las dos formulas
    de error y enlaza origen con destino; las dummies porque suman 1.
    """
    from src.serving.explain import infer_dependency_groups

    grupos = infer_dependency_groups(PAYSIM_COLS)
    por_tamano = {len(g): [PAYSIM_COLS[j] for j in g] for g in grupos}

    assert sorted(len(g) for g in grupos) == [1, 5, 9]
    assert por_tamano[1] == ["step"]
    assert all(c.startswith("type_") for c in por_tamano[5])
    assert "amount" in por_tamano[9] and "errorBalanceOrig" in por_tamano[9]


def test_los_grupos_cubren_cada_columna_exactamente_una_vez():
    from src.serving.explain import infer_dependency_groups

    cubiertas = sorted(j for g in infer_dependency_groups(PAYSIM_COLS) for j in g)
    assert cubiertas == list(range(len(PAYSIM_COLS)))


def test_las_columnas_de_un_grupo_comparten_atribucion(paquete):
    """Se movieron juntas, asi que repartir el efecto entre ellas seria inventarlo."""
    grupos = [[0, 1], [2], [3]]
    aporte = occlusion_attribution(
        paquete.score_from_scaled, np.full((1, 4), 5.0), paquete.background, FEATURES, grupos
    )
    assert aporte[0, 0] == aporte[0, 1]


def test_ocluir_por_grupo_da_un_resultado_distinto_que_de_a_una(paquete):
    """Es la correccion del modulo 8: de a una produce puntos imposibles."""
    punto = np.full((1, 4), 6.0)

    de_a_una = occlusion_attribution(
        paquete.score_from_scaled, punto, paquete.background, FEATURES
    )
    por_grupo = occlusion_attribution(
        paquete.score_from_scaled, punto, paquete.background, FEATURES, [[0, 1, 2], [3]]
    )
    assert not np.allclose(de_a_una, por_grupo)


def test_rechaza_grupos_que_no_cubren_todas_las_columnas(paquete):
    with pytest.raises(ValueError, match="exactamente una vez"):
        occlusion_attribution(
            paquete.score_from_scaled, np.zeros((2, 4)), np.zeros(4), FEATURES, [[0, 1]]
        )


def test_la_explicacion_por_grupo_no_repite_la_misma_columna(paquete):
    """Con grupos se devuelve una entrada por grupo, no tres veces el mismo numero."""
    motivos = explain_rows(
        paquete.score_from_scaled, np.full((1, 4), 7.0), paquete.background,
        FEATURES, top_k=3, groups=[[0, 1, 2], [3]],
    )
    etiquetas = [nombre for nombre, _ in motivos[0]]

    assert len(motivos[0]) == 2, "solo hay dos grupos, no puede devolver tres entradas"
    assert len(set(etiquetas)) == 2
    assert any("+" in e for e in etiquetas), "el grupo multiple debe nombrarse con sus columnas"

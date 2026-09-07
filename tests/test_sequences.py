"""Pruebas unitarias del detector secuencial por cuenta (src.deep.sequences).

El grueso de las pruebas apunta a `build_account_sequences`, que es donde vive el riesgo
real: si el relleno, el orden cronológico o el filtro de comercios están mal, el modelo
entrena sobre datos que no representan lo que dice representar y ninguna métrica lo delata.
"""
import numpy as np
import pandas as pd
import pytest

from src.deep.sequences import (
    FEATURE_NAMES,
    aggregate_step_errors,
    build_account_sequences,
    fraud_window_coverage,
    sequence_anomaly_score,
    train_sequence_autoencoder,
)


def hacer_transacciones(cuentas: dict[str, int], fraude_en: set[str] | None = None) -> pd.DataFrame:
    """Construye un DataFrame estilo PaySim: {nombre_cuenta: n_transacciones}."""
    fraude_en = fraude_en or set()
    filas = []
    for nombre, n in cuentas.items():
        for i in range(n):
            filas.append({
                "step": i * 2,
                "type": "TRANSFER",
                "amount": 100.0 + i,
                "nameDest": nombre,
                "oldbalanceDest": 50.0 * i,
                "newbalanceDest": 50.0 * i + 100.0,
                "isFraud": int(nombre in fraude_en and i == n - 1),
            })
    return pd.DataFrame(filas)


def test_descarta_cuentas_con_historia_insuficiente():
    df = hacer_transacciones({"C1": 6, "C2": 2, "C3": 5})
    sequences, mask, labels = build_account_sequences(df, min_length=5)

    # Solo C1 y C3 llegan a cinco transacciones.
    assert sequences.shape[0] == 2
    assert mask.shape[0] == 2
    assert labels.shape[0] == 2


def test_excluye_las_cuentas_de_comercio():
    # En PaySim el fraude nunca va a un comercio: dejarlos adentro le regala al modelo un
    # atajo trivial y una métrica inflada.
    df = hacer_transacciones({"C1": 6, "M1": 8})
    sequences, _, _ = build_account_sequences(df, min_length=5)
    assert sequences.shape[0] == 1


def test_rellena_al_inicio_y_deja_lo_mas_reciente_al_final():
    df = hacer_transacciones({"C1": 6})
    sequences, mask, _ = build_account_sequences(df, min_length=5, max_length=10)

    # Seis transacciones en diez ranuras: las cuatro primeras son relleno.
    np.testing.assert_array_equal(mask[0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert np.all(sequences[0, :4] == 0)
    # El monto crece con i, así que la última posición debe ser el monto mayor.
    montos = sequences[0, 4:, FEATURE_NAMES.index("log_amount")]
    assert montos[-1] == montos.max()


def test_conserva_las_transacciones_mas_recientes_al_truncar():
    df = hacer_transacciones({"C1": 12})
    sequences, mask, _ = build_account_sequences(df, min_length=5, max_length=4)

    assert sequences.shape[1] == 4
    assert mask[0].sum() == 4
    # Con 12 transacciones y max_length=4 deben quedar los montos 100+8 .. 100+11.
    montos = np.expm1(sequences[0, :, FEATURE_NAMES.index("log_amount")])
    np.testing.assert_allclose(montos, [108.0, 109.0, 110.0, 111.0], rtol=1e-5)


def test_etiqueta_la_cuenta_si_cualquier_transaccion_fue_fraude():
    df = hacer_transacciones({"C1": 6, "C2": 6}, fraude_en={"C2"})
    _, _, labels = build_account_sequences(df, min_length=5)
    assert labels.sum() == 1


def test_delta_step_mide_la_cadencia_entre_transacciones():
    # La primera transacción de una cuenta no tiene anterior: su delta es 0. El resto debe
    # reflejar la separación real en steps (2 en el generador).
    df = hacer_transacciones({"C1": 6})
    sequences, _, _ = build_account_sequences(df, min_length=5, max_length=6)

    deltas = sequences[0, :, FEATURE_NAMES.index("delta_step")]
    assert deltas[0] == 0.0
    np.testing.assert_allclose(deltas[1:], 2.0)


def test_ordena_cronologicamente_aunque_la_entrada_venga_desordenada():
    df = hacer_transacciones({"C1": 6}).sample(frac=1, random_state=0).reset_index(drop=True)
    sequences, _, _ = build_account_sequences(df, min_length=5, max_length=6)

    montos = sequences[0, :, FEATURE_NAMES.index("log_amount")]
    assert np.all(np.diff(montos) > 0), "la secuencia debe quedar ordenada por step"


def test_marca_el_tipo_de_transaccion_en_un_solo_indicador():
    df = hacer_transacciones({"C1": 6})
    sequences, _, _ = build_account_sequences(df, min_length=5, max_length=6)

    indicadores = sequences[0, :, [FEATURE_NAMES.index(n)
                                   for n in ("is_transfer", "is_cash_out", "is_other_type")]]
    np.testing.assert_allclose(indicadores.sum(axis=0), 1.0)


def test_falla_con_mensaje_claro_si_ninguna_cuenta_tiene_historia():
    df = hacer_transacciones({"C1": 2})
    with pytest.raises(ValueError, match="Ninguna cuenta destino"):
        build_account_sequences(df, min_length=5)


def test_el_autoencoder_ignora_el_relleno_al_puntuar():
    # Dos cuentas con la misma historia real pero distinto largo de relleno deben recibir
    # scores comparables: si el relleno contara, la más corta saldría artificialmente rara.
    df = hacer_transacciones({"C1": 5, "C2": 5})
    sequences, mask, _ = build_account_sequences(df, min_length=5, max_length=20)

    modelo = train_sequence_autoencoder(sequences, mask, epochs=3)
    scores_largos = sequence_anomaly_score(modelo, sequences, mask)

    # Mismos datos en un tensor con menos ranuras de relleno.
    sequences_cortas, mask_cortas, _ = build_account_sequences(df, min_length=5, max_length=5)
    scores_cortos = sequence_anomaly_score(modelo, sequences_cortas, mask_cortas)

    assert np.isfinite(scores_largos).all() and np.isfinite(scores_cortos).all()
    np.testing.assert_allclose(scores_largos, scores_cortos, rtol=0.5)


def test_el_score_secuencial_devuelve_un_valor_por_cuenta():
    df = hacer_transacciones({f"C{i}": 6 for i in range(12)})
    sequences, mask, _ = build_account_sequences(df, min_length=5)

    modelo = train_sequence_autoencoder(sequences, mask, epochs=2)
    scores = sequence_anomaly_score(modelo, sequences, mask, batch_size=5)

    assert scores.shape == (12,)
    assert np.isfinite(scores).all()


def test_cobertura_del_truncado_cuenta_el_fraude_dentro_de_la_ventana():
    # C1 recibe fraude en su transacción más reciente (posición 0 desde el final): entra en
    # cualquier ventana. Es la comprobación que permite descartar el truncado como
    # explicación de un mal resultado del detector.
    df = hacer_transacciones({"C1": 8, "C2": 8}, fraude_en={"C1"})
    cobertura = fraud_window_coverage(df, min_length=5, max_length=20)

    assert cobertura["fraud_transactions"] == 1
    assert cobertura["inside_window"] == 1
    assert cobertura["coverage"] == 1.0


def test_cobertura_detecta_el_fraude_que_queda_fuera_de_la_ventana():
    # Con una ventana corta, el fraude al inicio de una historia larga queda fuera: es el
    # caso que invalidaría cualquier lectura del resultado del detector.
    df = hacer_transacciones({"C1": 8})
    df.loc[0, "isFraud"] = 1
    df.loc[7, "isFraud"] = 0

    dentro = fraud_window_coverage(df, min_length=5, max_length=20)
    fuera = fraud_window_coverage(df, min_length=5, max_length=3)

    assert dentro["coverage"] == 1.0
    assert fuera["coverage"] == 0.0


def test_cobertura_ignora_cuentas_sin_historia_suficiente():
    df = hacer_transacciones({"C1": 3}, fraude_en={"C1"})
    assert fraud_window_coverage(df, min_length=5)["fraud_transactions"] == 0


def test_agregaciones_resumen_los_errores_por_paso_de_forma_distinta():
    step_errors = np.array([[0.0, 0.0, 1.0, 9.0]])
    mask = np.array([[0.0, 1.0, 1.0, 1.0]])

    # media sobre los 3 pasos reales = 10/3; máximo = 9; top3 = (0+1+9)/3; último = 9.
    np.testing.assert_allclose(aggregate_step_errors(step_errors, mask, "mean"), [10 / 3])
    np.testing.assert_allclose(aggregate_step_errors(step_errors, mask, "max"), [9.0])
    np.testing.assert_allclose(aggregate_step_errors(step_errors, mask, "top3"), [10 / 3])
    np.testing.assert_allclose(aggregate_step_errors(step_errors, mask, "last"), [9.0])


def test_la_media_diluye_un_unico_paso_anomalo_y_el_maximo_no():
    # Dos cuentas con el mismo pico anómalo pero distinta cantidad de pasos normales: la
    # media las separa (efecto de dilución), el máximo las iguala.
    step_errors = np.array([
        [0.0, 0.0, 0.0, 9.0],
        [9.0, 0.0, 0.0, 0.0],
    ])
    mask = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    medias = aggregate_step_errors(step_errors, mask, "mean")
    maximos = aggregate_step_errors(step_errors, mask, "max")

    assert medias[0] > medias[1]
    np.testing.assert_allclose(maximos[0], maximos[1])


def test_rechaza_una_agregacion_desconocida():
    with pytest.raises(ValueError, match="Agregación desconocida"):
        aggregate_step_errors(np.zeros((2, 3)), np.ones((2, 3)), how="promedio")

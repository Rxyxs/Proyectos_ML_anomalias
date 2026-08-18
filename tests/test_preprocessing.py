"""Pruebas unitarias para src.data.preprocessing."""
import pandas as pd

from src.data.preprocessing import clean_data, drop_identifier_columns, encode_transaction_type


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["PAYMENT", "TRANSFER"],
            "amount": [100.0, 200.0],
            "nameOrig": ["C1", "C2"],
            "oldbalanceOrg": [500.0, 300.0],
            "newbalanceOrig": [400.0, 100.0],
            "nameDest": ["M1", "M2"],
            "oldbalanceDest": [0.0, 0.0],
            "newbalanceDest": [100.0, 200.0],
            "isFraud": [0, 1],
            "isFlaggedFraud": [0, 0],
        }
    )


def test_drop_identifier_columns_removes_expected_columns():
    df = _sample_df()
    result = drop_identifier_columns(df)

    assert "nameOrig" not in result.columns
    assert "nameDest" not in result.columns
    assert "isFlaggedFraud" not in result.columns
    assert "amount" in result.columns


def test_drop_identifier_columns_ignores_missing_columns():
    df = _sample_df().drop(columns=["nameOrig"])
    result = drop_identifier_columns(df)

    assert "nameDest" not in result.columns
    assert "isFlaggedFraud" not in result.columns


def test_encode_transaction_type_creates_dummies():
    df = _sample_df()
    result = encode_transaction_type(df)

    assert "type" not in result.columns
    assert "type_PAYMENT" in result.columns
    assert "type_TRANSFER" in result.columns
    assert result.loc[0, "type_PAYMENT"] == 1
    assert result.loc[0, "type_TRANSFER"] == 0


def test_clean_data_drops_duplicates_and_identifier_columns():
    df = pd.concat([_sample_df(), _sample_df().iloc[[0]]], ignore_index=True)
    result = clean_data(df)

    assert len(result) == 2
    assert "nameOrig" not in result.columns
    assert "nameDest" not in result.columns
    assert "isFlaggedFraud" not in result.columns
    assert "type" not in result.columns
    assert "isFraud" in result.columns

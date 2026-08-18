"""Pruebas unitarias para src.features.build_features."""
import pandas as pd

from src.features.build_features import add_balance_deltas, add_zero_balance_flags, build_features


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [100.0, 50.0],
            "oldbalanceOrg": [500.0, 0.0],
            "newbalanceOrig": [400.0, 0.0],
            "oldbalanceDest": [0.0, 200.0],
            "newbalanceDest": [100.0, 250.0],
        }
    )


def test_add_balance_deltas_computes_expected_discrepancies():
    df = _sample_df()
    result = add_balance_deltas(df)

    # errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
    assert result.loc[0, "errorBalanceOrig"] == 400.0 + 100.0 - 500.0
    assert result.loc[1, "errorBalanceOrig"] == 0.0 + 50.0 - 0.0

    # errorBalanceDest = oldbalanceDest + amount - newbalanceDest
    assert result.loc[0, "errorBalanceDest"] == 0.0 + 100.0 - 100.0
    assert result.loc[1, "errorBalanceDest"] == 200.0 + 50.0 - 250.0


def test_add_balance_deltas_does_not_mutate_input():
    df = _sample_df()
    add_balance_deltas(df)

    assert "errorBalanceOrig" not in df.columns


def test_add_zero_balance_flags_marks_zero_balances():
    df = _sample_df()
    result = add_zero_balance_flags(df)

    assert result.loc[0, "origBalanceZero"] == 0
    assert result.loc[1, "origBalanceZero"] == 1
    assert result.loc[0, "destBalanceZero"] == 1
    assert result.loc[1, "destBalanceZero"] == 0


def test_build_features_adds_all_expected_columns():
    df = _sample_df()
    result = build_features(df)

    expected_new_columns = {"errorBalanceOrig", "errorBalanceDest", "origBalanceZero", "destBalanceZero"}
    assert expected_new_columns.issubset(result.columns)
    assert len(result) == len(df)

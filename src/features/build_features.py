"""Ingeniería de características para el modelo de detección de fraude."""
from __future__ import annotations

import pandas as pd


def add_balance_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de discrepancia entre saldos esperados y reales."""
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def add_zero_balance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores binarios de saldo en cero antes/después de la transacción."""
    df = df.copy()
    df["origBalanceZero"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["destBalanceZero"] = (df["oldbalanceDest"] == 0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline principal de construcción de features sobre datos ya limpios."""
    df = add_balance_deltas(df)
    df = add_zero_balance_flags(df)
    return df

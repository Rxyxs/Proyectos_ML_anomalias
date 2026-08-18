"""Limpieza y transformación de los datos crudos de PaySim."""
from __future__ import annotations

import pandas as pd

COLUMNS_TO_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]


def drop_identifier_columns(df: pd.DataFrame, columns: list[str] = COLUMNS_TO_DROP) -> pd.DataFrame:
    """Elimina columnas identificadoras/irrelevantes para el modelado."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def encode_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna categórica 'type' en variables dummy."""
    return pd.get_dummies(df, columns=["type"], prefix="type")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza principal aplicado sobre el dataset crudo."""
    df = df.drop_duplicates()
    df = drop_identifier_columns(df)
    df = encode_transaction_type(df)
    return df

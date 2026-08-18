"""Descarga y carga del dataset PaySim (ealaxi/paysim1) desde Kaggle."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"
KAGGLE_DATASET = "ealaxi/paysim1"


def download_dataset(target_path: str | Path = RAW_DATA_PATH) -> Path:
    """Descarga el dataset PaySim vía kagglehub si no existe en target_path.

    Copia el .csv descargado al destino esperado dentro de data/raw/.
    """
    target_path = Path(target_path)
    if target_path.exists():
        return target_path

    import kagglehub

    target_path.parent.mkdir(parents=True, exist_ok=True)
    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    csv_files = sorted(download_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .csv en el dataset descargado: {download_dir}"
        )

    shutil.copy2(csv_files[0], target_path)
    return target_path


def load_raw_data(filepath: str | Path = "data/raw/paysim.csv") -> pd.DataFrame:
    """Carga el dataset PaySim crudo, descargándolo primero si es necesario."""
    filepath = Path(filepath)
    if not filepath.exists():
        filepath = download_dataset(filepath)
    return pd.read_csv(filepath)


if __name__ == "__main__":
    df = load_raw_data()

    print(f"Dimensiones del dataset: {df.shape}")
    print("\nPrimeras 5 filas:")
    print(df.head())

    fraud_pct = df["isFraud"].value_counts(normalize=True) * 100
    print("\nDesglose porcentual de isFraud:")
    for label, pct in fraud_pct.sort_index().items():
        tag = "Fraude" if label == 1 else "No fraude"
        print(f"  {tag} ({label}): {pct:.4f}%")

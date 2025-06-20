# tests/test_dataset_loader.py
import os

import pytest
from pathlib import Path
import pandas as pd

from utils.dataset_loader import (
    load_iemocap_metadata,
    get_fold,
)
from config import EMOTIONS, N_FOLDS


@pytest.fixture(scope="module")
def metadata_df():
    # Cargamos una sola vez todo el metadata
    df = load_iemocap_metadata()
    return df

def test_non_empty(metadata_df):
    # Debe haber al menos un utterance tras el filtrado
    assert len(metadata_df) > 0, "El DataFrame no debería estar vacío"

def test_labels_subset(metadata_df):
    # Todas las etiquetas resultantes deben estar en EMOTIONS
    assert set(metadata_df["label"].unique()).issubset(set(EMOTIONS))

def test_sessions_range(metadata_df):
    # Debemos encontrar sesiones 1..N_FOLDS
    found = set(metadata_df["session"].unique())
    expected = set(range(1, N_FOLDS + 1))
    assert found == expected, f"Sesiones esperadas {expected}, encontradas {found}"

def test_get_fold_splits(metadata_df):
    # Verifica que get_fold separa correctamente train vs test
    for fold in range(N_FOLDS):
        train_df, test_df = get_fold(metadata_df, fold)
        # Test solo contiene la sesion fold+1
        assert set(test_df["session"].unique()) == {fold + 1}
        # Train no contiene esa sesion
        assert fold + 1 not in set(train_df["session"].unique())
        # La unión de train + test cubre todo
        combined = pd.concat([train_df, test_df], ignore_index=True)
        assert set(combined.index) == set(metadata_df.index)

def test_path_existence(metadata_df):
    # Comprueba que al menos el 99% de las rutas realmente exista
    paths = metadata_df["path"].tolist()
    missing = [p for p in paths if not Path(p).exists()]
    assert len(missing) / len(paths) < 0.01, f"Muchas rutas faltantes: {len(missing)}"

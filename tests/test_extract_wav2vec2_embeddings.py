# tests/test_extract_wav2vec2_embeddings.py

import numpy as np
import soundfile as sf
import pytest
import torch
from pathlib import Path
import importlib

import config
import extractors  # para stubbar get_extractor

@pytest.fixture
def mini_dataset(tmp_path, monkeypatch):
    """
    Crea mini-IEMOCAP con una sola utterance:
      tmp_path/IEMOCAP/
        Session1/dialog/EmoEvaluation/mini.txt
        Session1/sentences/wav/mini/mini_0001.wav
    """
    root     = tmp_path / "IEMOCAP"
    session1 = root / "Session1"
    (session1 / "dialog/EmoEvaluation").mkdir(parents=True)
    (session1 / "sentences/wav/mini").mkdir(parents=True)

    # 1s de seno a SAMPLE_RATE
    sr    = config.SAMPLE_RATE
    t     = np.linspace(0, 1, sr, endpoint=False)
    wave  = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = session1 / "sentences/wav/mini/mini_0001.wav"
    sf.write(str(wav_path), wave, sr)

    # Anotación raw 'ang'
    txt = session1 / "dialog/EmoEvaluation/mini.txt"
    txt.write_text(f"[0.000 - 1.000]\tmini_0001\tang\n")

    # Parchea DATA_ROOT para que loader use este mini-dataset
    monkeypatch.setattr(config, "DATA_ROOT", root)
    return {"wav_path": str(wav_path)}

def test_extract_script_generates_npy_fast(mini_dataset, monkeypatch):
    # Stub: reemplaza get_extractor para evitar cargar el modelo real
    def dummy_get_extractor(name, device=None):
        class Dummy:
            def extract_batch(self, speeches, srs):
                return np.zeros((len(speeches), config.EMB_DIM), dtype=np.float32)
        return Dummy()
    monkeypatch.setattr(extractors, "get_extractor", dummy_get_extractor)

    # Importa y recarga el modulo *después* de stubbar
    script_mod = importlib.reload(importlib.import_module("scripts.extract_wav2vec2_embeddings"))

    # Ejecuta la funcion principal
    script_mod.main()

    # Comprueba que los archivos .npy se crearon
    out = Path(config.OUT_DIR)
    emb_file    = out / "all_w2v2_emb.npy"
    labels_file = out / "all_labels.npy"

    assert emb_file.exists(),    "No se crea all_w2v2_emb.npy"
    assert labels_file.exists(), "No se crea all_labels.npy"

    embs   = np.load(emb_file)
    labels = np.load(labels_file)

    # Debe haber exactamente 1 embedding con EMB_DIM dimensiones
    assert embs.shape == (1, config.EMB_DIM)

    # raw 'ang' mapea a canonical 'angry'
    expected = config.LABEL_MAP["ang"]
    assert labels.tolist() == [expected]

# tests/test_extractor_wav2vec2.py

import numpy as np
import torch
import pytest

from extractors.wav2vec2_extractor import Wav2Vec2Extractor
from config import SAMPLE_RATE, EMB_DIM

@pytest.fixture(scope="module")
def sine_wave():
    """Seno de 440 Hz, 1 s, 16 kHz, float32."""
    sr = SAMPLE_RATE
    t  = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    return wave.astype(np.float32), sr

def test_extract_single(sine_wave):
    speech, sr = sine_wave
    extractor = Wav2Vec2Extractor(device=torch.device("cpu"))
    emb = extractor.extract(speech, sr)

    # 1) Tipo y forma
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (EMB_DIM,)
    # 2) Dtype correcto
    assert emb.dtype == np.float32
    # 3) No es todo ceros
    assert np.any(emb != 0.0)

def test_extract_batch(sine_wave):
    wave, sr = sine_wave
    extractor = Wav2Vec2Extractor(device=torch.device("cpu"))

    # Duplicamos la señal para formar un batch de 2
    batch = [wave, wave]
    srs   = [sr, sr]
    embs  = extractor.extract_batch(batch, srs)

    # Chequeos basicos
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, EMB_DIM)
    assert embs.dtype == np.float32
    # Asegurarnos de que los dos vectores sean “similares” (sin ser iguales exactos)
    np.testing.assert_allclose(embs[0], embs[1], rtol=1e-5, atol=1e-6)

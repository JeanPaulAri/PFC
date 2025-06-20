# tests/test_dataset_parser.py

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

import config
from utils.dataset_loader import load_iemocap_metadata


@pytest.fixture
def mini_iemocap(tmp_path, monkeypatch):
    """
    Crea una mini-estructura IEMOCAP:
      tmp_path/IEMOCAP_full_release/
        Session1/dialog/EmoEvaluation/Ses01F_impro01.txt
        Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
    Parchea config.DATA_ROOT para que el loader use este directorio.
    """
    # 1) Directorio raíz simulando el release
    root     = tmp_path / "IEMOCAP_full_release"
    session1 = root / "Session1"
    eval_dir = session1 / "dialog" / "EmoEvaluation"
    # subcarpeta de diálogo
    dialog_dir = session1 / "sentences" / "wav" / "Ses01F_impro01"

    # 2) Crear carpetas
    eval_dir .mkdir(parents=True, exist_ok=True)
    dialog_dir.mkdir(parents=True, exist_ok=True)

    # 3) Generar un .wav dummy de 1s a 16kHz
    sr   = config.SAMPLE_RATE
    t    = np.linspace(0, 1.0, sr, endpoint=False)
    wave = 0.01 * np.sin(2 * np.pi * 440 * t)
    # Aquí incluimos canal "_F000" en el nombre
    utt_id   = "Ses01F_impro01_F000"
    wav_file = dialog_dir / f"{utt_id}.wav"
    sf.write(str(wav_file), wave.astype(np.float32), sr)

    # 4) Escribir la anotación en el TXT (usa exactamente el mismo utt_id)
    txt = eval_dir / "Ses01F_impro01.txt"
    txt.write_text(f"[0.000 - 1.000]\t{utt_id}\tneu\n")

    # 5) Monkeypatch DATA_ROOT (y si existe IEMOCAP_PATH)
    monkeypatch.setattr(config, "DATA_ROOT", root)
    if hasattr(config, "IEMOCAP_PATH"):
        monkeypatch.setattr(config, "IEMOCAP_PATH", str(root))

    return {"wav_path": str(wav_file)}


def test_loader_parsing_and_wav_existence(mini_iemocap):
    df = load_iemocap_metadata()
    # Solo la nuestra
    assert len(df) == 1, f"Se esperaba 1 registro, obtuvo {len(df)}"

    row = df.iloc[0]
    # raw_label en el TXT
    assert row["raw_label"] == "neu"
    # mapeo → 'neutral'
    assert row["label"] == "neutral"
    # ruta al WAV coincide
    assert row["path"] == mini_iemocap["wav_path"]
    # sesión = 1
    assert row["session"] == 1

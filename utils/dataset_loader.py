import os

import pandas as pd
from pathlib import Path

import config

# EMOTION_MAP = {
#     "ang": "angry",
#     "hap": "happy",
#     "neu": "neutral",
#     "sad": "sad",
#     "exc": "excited",
#     "fru": "frustrated",
#     "sur": "surprised",
#     "fea": "fearful",
#     "dis": "disgusted",
#     "xxx": None,  # invalid or undefined
#     "oth": None
#     # emociones bases para el proyecto
# }

def load_iemocap_metadata() -> pd.DataFrame:
    """
    Lee IEMOCAP desde DATA_ROOT, mapea etiquetas con LABEL_MAP,
    filtra solo las de interest (EMOTIONS) y devuelve un DataFrame con:
      - path (str)
      - raw_label (str)
      - label (str)
      - session (int)
      - gender (optional, aquí None)
    """
    records = []
    raw_emotions_found = set()

    DATA_ROOT = Path(config.DATA_ROOT)
    LABEL_MAP = config.LABEL_MAP
    EMOTIONS  = set(config.EMOTIONS)
    N_FOLDS   = config.N_FOLDS
    # Itera sesiones 1..5
    for session in range(1, 6):
        session_dir = DATA_ROOT / f"Session{session}"
        eval_dir    = session_dir / "dialog"     / "EmoEvaluation"
        wav_dir     = session_dir / "sentences"  / "wav"

        if not eval_dir.is_dir():
            continue
        # Cada archivo .txt en la carpeta de anotaciones
        for txt_path in eval_dir.glob("*.txt"):
            for line in txt_path.open("r"):
                if not line.startswith("["):  
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 3:  
                    continue

                utt_id       = parts[1]
                raw          = parts[2].lower().strip()
                raw_emotions_found.add(raw)

                # Mapea crudo -> canonical; descarta si no esta o es None
                canonical = LABEL_MAP.get(raw)
                if canonical is None or canonical not in EMOTIONS:
                    continue

                # Construye la ruta al .wav
                subdir  = utt_id.rsplit("_", 1)[0]
                wav_path = wav_dir / subdir / f"{utt_id}.wav"
                if not wav_path.exists():
                    continue

                # _TODO: extrae genero del hablante si es necesario:
                gender = None

                records.append({
                    "path":   str(wav_path),
                    "raw_label": raw,
                    "label":  canonical,
                    "session": session,
                    "gender": gender
                })

    df = pd.DataFrame(records)
    # Informacion de diagnostico
    print(f"[INFO] Emociones crudas encontradas: {sorted(raw_emotions_found)}")
    print(f"[INFO] Total utts filtradas: {len(df)}")
    return df

def get_fold(df: pd.DataFrame, fold: int):
    """
    fold ∈ [0..N_FOLDS-1]; usa leave-one-session-out.
    Devuelve (train_df, test_df) donde test_df.session == fold+1.
    """
    test_s = fold + 1
    train_df = df[df.session != test_s].reset_index(drop=True)
    test_df  = df[df.session == test_s].reset_index(drop=True)
    return train_df, test_df

import pandas as pd
import os

from config import IEMOCAP_PATH

EMOTION_MAP = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    "exc": "excited",
    "fru": "frustrated",
    "sur": "surprised",
    "fea": "fearful",
    "dis": "disgusted",
    "xxx": None,  # invalid or undefined
    "oth": None
    # emociones bases para el proyecto
}

def load_iemocap_metadata(emotion_set=("happy", "sad", "angry", "neutral")):
    """
    Carga metadatos de IEMOCAP con etiquetas emocionales discretas.

    Args:
        emotion_set (tuple): Emociones que se desean filtrar.

    Returns:
        pd.DataFrame: DataFrame con columnas [path, emotion, session, gender]
    """
    data = []
    total_lines = 0
    matched_lines = 0
    raw_emotions_found = set()
    for session in range(1, 6):
        session_id = f"Session{session}"
        eval_dir = os.path.join(IEMOCAP_PATH, session_id, "dialog", "EmoEvaluation")
        wav_dir = os.path.join(IEMOCAP_PATH, session_id, "sentences", "wav")
        
        for root, _, files in os.walk(eval_dir):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r") as f:
                        for line in f:
                            total_lines += 1
                            if line.startswith("["):
                                parts = line.strip().split("\t")

                                # Depuracion:
                                print("LINEA:", line.strip())
                                print("PARTES:", parts)

                                if len(parts) >= 3:
                                    utt_id = parts[1]
                                    raw_emotion = parts[2]
                                    raw_emotions_found.add(raw_emotion)

                                    if raw_emotion in EMOTION_MAP:
                                        emotion = EMOTION_MAP[raw_emotion]
                                        if emotion in emotion_set and emotion is not None:
                                            matched_lines += 1
                                            session_path = os.path.join(
                                                wav_dir,
                                                utt_id[:utt_id.rfind('_')],
                                                utt_id + ".wav"
                                            )
                                            gender = utt_id[0]  # M o F
                                            data.append({
                                                "path": session_path,
                                                "emotion": emotion,
                                                "session": session_id,
                                                "gender": gender
                                            })

    print(f"[INFO] Lineas totales: {total_lines}")
    print(f"[INFO] Lineas con emocion mapeada: {matched_lines}")
    print(f"[INFO] Emociones encontradas: {sorted(raw_emotions_found)}")
    return pd.DataFrame(data)

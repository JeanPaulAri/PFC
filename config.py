# config.py
"""
config.py  ·  Parametros globales para comparar embeddings
wav2vec 2.0  vs  emotion2vec en IEMOCAP

Modificar SOLO este archivo para:
- cambiar la ubicacion del dataset
- variar el numero de clases
- alterar semillas o batch size, etc
"""
from pathlib import Path
# Ruta al dataset IEMOCAP (se debe ajustar segun la ubicacion real del dataset)
IEMOCAP_PATH = "C:\IEMOCAP_full_release"

#---- paths de salida y directorios de trabajo ----
DATA_ROOT = Path(IEMOCAP_PATH)
OUT_DIR = Path("embeddings")
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")


# --- Audio & batching --------------------------------------------------
SAMPLE_RATE = 16000
BATCH_SIZE  = 8
EMB_DIM     = 768 # Dimensiones de los embeddings (wav2vec2 y emotion2vec)

# --- Experiment design -------------------------------------------------
SEED        = 42
N_FOLDS     = 5            # leave-one-session-out
EMOTIONS    = ["angry", "happy", "sad", "neutral"]  

LABEL_MAP = {
    # ─ abreviaturas oficiales (3 letras) ─
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",          # excited se fusiona con happy
    "sad": "sad",
    "neu": "neutral",
    "fru": "frustration",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "oth": None,             # descartar “other / no agreement”

    # ─ nombres en texto completo ─
    "angry": "angry",
    "happiness": "happy",
    "excited": "happy",
    "sadness": "sad",
    "sad": "sad",
    "neutral": "neutral",
    "frustration": "frustration",
    "fearful": "fear",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "other": None,
}
# ─ crear directorios de salida ─
for _p in (OUT_DIR, MODEL_DIR, PLOTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ─ etiquetas activas ─
ACTIVE_LABELS = set(EMOTIONS)
"""
    extract_wav2vec2_embeddings.py

    Este script recorre todos los segmentos de audio del dataset IEMOCAP, 
    utiliza el modelo preentrenado `facebook/wav2vec2-base-960h` disponible en Hugging Face 
    para extraer embeddings de cada archivo de voz y guarda estos vectores 
    en formato `.npy` dentro del directorio `embeddings/wav2vec2/`.

    Uso previsto:
        python scripts/extract_wav2vec2_embeddings.py

    Requiere:
    - Hugging Face Transformers
    - soundfile
    - torch
    - utils.dataset_loader
    - extractors.wav2vec2_extractor

    El resultado es un conjunto de vectores de caracteristicas por segmento, 
    listos para tareas de clasificacion, regresion o visualizacion emocional.
"""

import os

import numpy as np
import torch
import soundfile as sf

from tqdm import tqdm
from pathlib import Path

from config import (
    DATA_ROOT,
    OUT_DIR,
    SAMPLE_RATE,
    BATCH_SIZE,
    SEED,
    EMB_DIM,
)

from utils.dataset_loader import load_iemocap_metadata
from extractors import get_extractor


def main():
    """
        Extrae embeddings de audio usando el modelo wav2vec2 preentrenado.
        Recorre los archivos de audio del dataset IEMOCAP y guarda los embeddings
        en formato .npy en el directorio OUTPUT_DIR.
    """

    # 1) Prepara GPU/CPU y el extractor
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = get_extractor("w2v2", device)
    torch.manual_seed(SEED)


     # 2) Carga metadata (path, label, session…)
    df = load_iemocap_metadata()
    if df.empty:
        print("[ERROR] No se encontraron muestras con las emociones seleccionadas.")
        return
    
    N  = len(df)

    # 3) Inicializa contenedores
    all_embs   = np.zeros((N, EMB_DIM), dtype=np.float32)  # (N,768)
    all_labels = []

    # 4) Recorre en bloques para aprovechar la GPU sin saturar
    for start in range(0, N, BATCH_SIZE):
        end     = min(start + BATCH_SIZE, N)
        batch   = df.iloc[start:end]
        speeches, srs, labels = [], [], []

        for _, row in batch.iterrows():
            speech, sr = sf.read(row["path"])
            speeches.append(speech)
            srs.append(sr)
            labels.append(row["label"])

        # extrae lote (shape = (B, 768))
        batch_emb = extractor.extract_batch(speeches, srs)

        bs = batch_emb.shape[0]
        all_embs[start:start+bs] = batch_emb
        all_labels.extend(labels)
    # 5) Guarda artefactos
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "all_w2v2_emb.npy", all_embs)
    np.save(OUT_DIR / "all_labels.npy",   np.array(all_labels))

    print(f"[Done] Guardados {all_embs.shape[0]} embeddings de wav2vec2 en '{OUT_DIR}'")

if __name__ == "__main__":
    main()

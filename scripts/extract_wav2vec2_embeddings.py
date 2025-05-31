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
from tqdm import tqdm
import soundfile as sf

from utils.dataset_loader import load_iemocap_metadata
from extractors.wav2vec2_extractor import Wav2Vec2Extractor

OUTPUT_DIR = "embeddings/wav2vec2"

def main():
    """
        Extrae embeddings de audio usando el modelo wav2vec2 preentrenado.
        Recorre los archivos de audio del dataset IEMOCAP y guarda los embeddings
        en formato .npy en el directorio OUTPUT_DIR.
    """
    df = load_iemocap_metadata()
    if df.empty:
        print("[ERROR] No se encontraron muestras con las emociones seleccionadas.")
        return

    extractor = Wav2Vec2Extractor()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["path"]
        utt_id = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(OUTPUT_DIR, utt_id + ".npy")

        if not os.path.exists(audio_path):
            print(f"[WARN] No se encontro el audio: {audio_path}")
            continue

        try:
            embedding = extractor.extract(audio_path)
            np.save(out_path, embedding.cpu().numpy())
        except Exception as e:
            print(f"[ERROR] {utt_id}: {e}")



if __name__ == "__main__":
    main()

"""
    extract_emotion2vec_embeddings.py

    Este script recorre los archivos de audio del dataset IEMOCAP y extrae embeddings
    utilizando el modelo preentrenado emotion2vec (emotion2vec/emotion2vec_base) disponible en Hugging Face.

    Los embeddings se almacenan como archivos .npy en la carpeta embeddings/emotion2vec.

    Requiere:
    - IEMOCAP descargado y organizado correctamente
    - config.py con la ruta a IEMOCAP
    - extractor definido en extractors/emotion2vec_extractor.py

    Uso:
        python scripts/extract_emotion2vec_embeddings.py
"""
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf

from utils.dataset_loader import load_iemocap_metadata
from extractors.emotion2vec_extractor import Emotion2VecExtractor

OUTPUT_DIR = "embeddings/emotion2vec"

def main():
    df = load_iemocap_metadata()
    if df.empty:
        print("[ERROR] No se encontraron muestras con las emociones seleccionadas.")
        return
    
    extractor = Emotion2VecExtractor()

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

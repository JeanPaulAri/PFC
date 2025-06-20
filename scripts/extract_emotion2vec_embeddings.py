#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf

import config
from utils.dataset_loader import load_iemocap_metadata
from extractors import get_extractor


def main():
    """
    Extrae embeddings de emotion2vec para todo IEMOCAP y guarda:
      - embeddings/all_e2v_emb.npy
      - embeddings/all_labels.npy
    """
    # Inicializa extractor (usa ModelScope internamente)
    extractor = get_extractor("emotion2vec")

    # Carga metadata
    df = load_iemocap_metadata()
    N  = len(df)

    # Prealoca arrays
    all_embs   = np.zeros((N, config.EMB_DIM), dtype=np.float32)
    all_labels = []

    paths  = df['path'].tolist()
    labels = df['label'].tolist()

    # Extracción por batches
    for start in tqdm(range(0, N, config.BATCH_SIZE), desc="e2v-extract"):
        batch_paths  = paths[start:start+config.BATCH_SIZE]
        batch_labels = labels[start:start+config.BATCH_SIZE]

        # Llamada al extractor por lote
        batch_embs = extractor.extract_batch(batch_paths)
        bs = batch_embs.shape[0]

        all_embs[start:start+bs] = batch_embs
        all_labels.extend(batch_labels)

    # Guarda resultados
    out_dir = config.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "all_e2v_emb.npy", all_embs)
    np.save(out_dir / "all_labels.npy",  np.array(all_labels))

    print(f"[Done] Guardados {N} embeddings de emotion2vec en '{out_dir}'")


if __name__ == "__main__":
    main()

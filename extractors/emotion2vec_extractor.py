# extractors/emotion2vec_extractor.py

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np

class Emotion2VecExtractor:
    """
    Extrae embeddings 768-D de emotion2vec_base mediante ModelScope Pipeline.

    Uso:
        extractor = Emotion2VecExtractor()
        emb = extractor.extract("ruta/a/audio.wav")  # np.ndarray (768,)
    """
    def __init__(self,
                 model_name: str = "iic/emotion2vec_base",
                 device_id: int = 0):
        # Inicializa el pipeline de reconocimiento de emoción
        # extract_embedding=True puede no funcionar; usaremos feats en lugar
        self.pipe = pipeline(
            task=Tasks.emotion_recognition,
            model=model_name,
            device_id=device_id
        )

    def extract(self, audio_path: str) -> np.ndarray:
        """
        Extrae un embedding de 768-D dado un archivo de audio.

        Args:
            audio_path: ruta al .wav de la utterance
        Returns:
            np.ndarray con shape (768,)
        """
        res = self.pipe(
            audio_path,
            granularity='utterance',
            extract_embedding=True
        )

        # ModelScope puede devolver list o dict
        if isinstance(res, list):
            res = res[0]

        emb = None
        # Claves candidatas
        if isinstance(res, dict):
            # 'feats' contiene a menudo la representación final
            if 'feats' in res:
                feats = res['feats']
                # puede ser lista 2D o array; convertimos np.array
                arr = np.array(feats, dtype=np.float32)
                # si es temporal (ndim==2), hacemos mean pooling
                if arr.ndim == 2:
                    emb = arr.mean(axis=0)
                else:
                    emb = arr
            # otras claves
            for key in ('embeddings','embedding','utt_embedding','emotion_representation'):
                if key in res:
                    emb = np.array(res[key], dtype=np.float32)
                    break

        if emb is None:
            raise ValueError(f"No se encontró embedding. Claves disponibles: {list(res.keys())}")

        return emb

    def extract_batch(self, audio_paths: list) -> np.ndarray:
        """
        Extrae embeddings para un lote de archivos.

        Args:
            audio_paths: lista de rutas a archivos .wav
        Returns:
            np.ndarray con shape (B, 768)
        """
        embs = []
        for p in audio_paths:
            embs.append(self.extract(p))
        return np.vstack(embs)

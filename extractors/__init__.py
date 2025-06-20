# extractors/__init__.py

from .wav2vec2_extractor     import Wav2Vec2Extractor
from .emotion2vec_extractor  import Emotion2VecExtractor
import torch

def get_extractor(name: str, device: torch.device = None):
    """
    Factory que, dado el nombre del extractor, devuelve
    la clase adecuada inicializada en el device indicado.
    
    Params:
      name (str): "w2v2" o "emotion2vec" (o sus alias)
      device (torch.device): cuda o cpu; si es None, lo detecta automáticamente.

    Returns:
      instancia de Wav2Vec2Extractor o Emotion2VecExtractor
    """
    key = name.strip().lower()
    if key in ("w2v2", "wav2vec2"):
        # Para wav2vec2 podemos especificar el device
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return Wav2Vec2Extractor(device=device)
    elif key in ("e2v", "emotion2vec"):
        # Emotion2VecExtractor usa ModelScope y gestiona device internamente
        return Emotion2VecExtractor()
    else:
        raise ValueError(f"Extractor desconocido: {name}. Usa 'w2v2' o 'e2v'.")


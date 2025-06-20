"""
wav2vec2_extractor.py

Este modulo define la clase Wav2Vec2Extractor, que permite extraer embeddings de audio
usando el modelo wav2vec2 preentrenado desde Hugging Face (facebook/wav2vec2-base-960h).

El vector de salida corresponde al promedio de los embeddings de las capas ocultas del modelo.

Requiere:
- paquetes transformers, torch y soundfile

Uso:
    extractor = Wav2Vec2Extractor()
    embedding = extractor.extract("ruta/al/audio.wav")
"""

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import numpy as np
import soundfile as sf

from config import SAMPLE_RATE

class Wav2Vec2Extractor:
    """
        Clase que permite extraer embeddings de audio usando el modelo wav2vec2
        de Hugging Face (facebook/wav2vec2-base-960h).
    """
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            device: torch.device = None
            ):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name) \
                          .to(self.device) \
                          .eval()

    def extract(self, speech: np.ndarray, sr: int) -> np.ndarray:
        """
            Extrae el embedding promedio de un archivo de audio.

            Args:
                audio_path (str): Ruta al archivo .wav en formato mono y 16kHz.

            Returns:
                torch.Tensor: Vector de "embedding" promedio (1D tensor). 
                embedding: vector 1D np.float32 de longitud 768.
        """
        # re-muestreo si es necesario
        if sr != SAMPLE_RATE:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # tokenización + padding
        inputs = self.processor(
            speech,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(self.device)  # (1, L)

        # inferencia sin gradiente
        with torch.no_grad():
            outputs = self.model(input_values)
            # last_hidden_state: (1, T, 768)
            emb = outputs.last_hidden_state.mean(dim=1)      # (1, 768)

        return emb.squeeze(0).cpu().numpy()  # (768,)
    def extract_batch(self, speeches: list, srs: list) -> np.ndarray:
            """
            Version por lotes:
            speeches: lista de np.ndarray
            srs: lista de sample rates correspondientes
            Devuelve arreglo (B, 768).
            """
            # remuestrear en lote si hace falta
            proc_speeches = []
            for speech, sr in zip(speeches, srs):
                if sr != SAMPLE_RATE:
                    import librosa
                    speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
                proc_speeches.append(speech)

            # procesar lote
            inputs = self.processor(
                proc_speeches,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(self.device)  # (B, L)

            with torch.no_grad():
                outputs = self.model(input_values)
                # last_hidden_state: (B, T, 768)
                embs = outputs.last_hidden_state.mean(dim=1)     # (B, 768)

            return embs.cpu().numpy()
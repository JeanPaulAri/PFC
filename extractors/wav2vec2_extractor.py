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
import soundfile as sf

class Wav2Vec2Extractor:
    """
        Clase que permite extraer embeddings de audio usando el modelo wav2vec2
        preentrenado de Hugging Face (facebook/wav2vec2-base-960h).
    """
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """
            Extrae el embedding promedio de un archivo de audio.

            Args:
                audio_path (str): Ruta al archivo .wav en formato mono y 16kHz.

            Returns:
                torch.Tensor: Vector de embedding promedio (1D tensor).
        """
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def extract(self, audio_path):
        """
            Extrae el embedding promedio de un archivo de audio.

            Args:
                audio_path (str): Ruta al archivo .wav en formato mono y 16kHz.

            Returns:
                torch.Tensor: Vector de embedding promedio (1D tensor).
        """
        speech, sr = sf.read(audio_path)
        if sr != 16000:
            raise ValueError("El audio debe estar a 16 kHz.")
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

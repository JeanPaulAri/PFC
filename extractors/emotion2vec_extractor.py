"""
emotion2vec_extractor_local.py

Este modulo define una clase que permite cargar el modelo emotion2vec_base
desde Hugging Face usando Wav2Vec2Model y extraer embeddings de archivos WAV
sin depender de FeatureExtractor ni Processor.
"""

import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Emotion2VecExtractorLocal:
    """
    Extrae embeddings desde archivos .wav usando el modelo emotion2vec_base.
    """

    def __init__(self, model_name="emotion2vec/emotion2vec_base", device=None):
        """
        Inicializa el modelo y asigna dispositivo.

        Args:
            model_name (str): Ruta local o nombre Hugging Face del modelo.
            device (str): "cuda" o "cpu". Por defecto detecta automaticamente.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        config = Wav2Vec2Config.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name, config=config)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, audio_path):
        """
        Extrae el embedding promedio del archivo WAV.

        Args:
            audio_path (str): Ruta al archivo WAV (mono, 16 kHz)

        Returns:
            torch.Tensor: Vector de embedding promedio
        """
        waveform, sr = torchaudio.load(audio_path)

        if sr != 16000:
            raise ValueError("El archivo debe tener una tasa de muestreo de 16 kHz")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # convertir a mono

        input_values = waveform.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

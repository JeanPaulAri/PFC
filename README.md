# 🎧 Análisis de Voz para visualizacion emocional

## 📌 Descripción general

Este proyecto tiene como objetivo comparar dos enfoques de extracción de representaciones de voz para análisis emocional, utilizando modelos de aprendizaje profundo:

- **Wav2Vec2**: Un modelo auto-supervisado preentrenado por Facebook AI para tareas de reconocimiento de voz.
- **Emotion2Vec**: Una variante basada en Wav2Vec2 entrenada para generar embeddings que capturen emociones humanas.

Se procesan archivos de audio del dataset IEMOCAP y se generan embeddings `.npy` que luego pueden visualizarse o usarse en tareas de clasificación o regresión emocional.

---

## 📦 Repositorios y modelos utilizados

- 🔗 Wav2Vec2 (Hugging Face): https://huggingface.co/facebook/wav2vec2-base-960h
- 🔗 Emotion2Vec (GitHub): https://github.com/ddlBoJack/emotion2vec
- 🔗 Emotion2Vec (Hugging Face): https://huggingface.co/emotion2vec/emotion2vec_base

---

## 📂 Estructura del proyecto

```
PFC/
├── config.py                        # Ruta a IEMOCAP_PATH
├── scripts/
│   ├── extract_wav2vec2_embeddings.py
│   └── extract_emotion2vec_embeddings.py
├── extractors/
│   ├── wav2vec2_extractor.py
│   └── emotion2vec_extractor_local.py
├── utils/
│   └── dataset_loader.py
├── embeddings/                     # Aquí se guardan los .npy
└── README.md
```

---

## ⚙️ Cómo ejecutar el proyecto

1. **Clona este repositorio y crea entorno virtual**:
   ```bash
   git clone <este-repo>
   cd PFC
   python -m venv venv
   source venv/bin/activate  # o venv\Scripts\activate en Windows
   ```

2. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configura la ruta al dataset IEMOCAP**:
   Edita el archivo `config.py`:
   ```python
   IEMOCAP_PATH = "C:/ruta/a/IEMOCAP_full_release"
   ```

4. **Ejecuta el script deseado**:
   ```bash
   python -m scripts.extract_wav2vec2_embeddings
   python -m scripts.extract_emotion2vec_embeddings
   ```

5. **Visualiza o analiza los embeddings desde `embeddings/`**.

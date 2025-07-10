#!/usr/bin/env python
import argparse
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import joblib
from extractors import Emotion2VecExtractor, Wav2Vec2Extractor
import config
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import json


def infer_e2v(audio_path, model_paths):
    """
    Inferencia usando emotion2vec + clasificadores entrenados.
    """
    extractor = Emotion2VecExtractor()
    emb = extractor.extract(audio_path).reshape(1, -1)
    prob_list = []
    for mp in model_paths:
        clf = joblib.load(mp)
        prob = clf.predict_proba(emb)
        prob_list.append(prob)
    avg_prob = np.mean(prob_list, axis=0)[0]
    classes = clf.classes_
    idx = int(np.argmax(avg_prob))
    return classes[idx], avg_prob, classes


def infer_w2v2(audio_path, model_paths):
    """
    Inferencia usando embeddings de wav2vec2 + clasificadores entrenados.
    """
    # Leer y re-muestrear si es necesario
    speech, sr = sf.read(audio_path)
    if sr != config.SAMPLE_RATE:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        sr = config.SAMPLE_RATE
    # Extraer embedding
    extractor = Wav2Vec2Extractor()
    emb = extractor.extract(speech, sr).reshape(1, -1)
    prob_list = []
    for mp in model_paths:
        clf = joblib.load(mp)
        prob = clf.predict_proba(emb)
        prob_list.append(prob)
    avg_prob = np.mean(prob_list, axis=0)[0]
    classes = clf.classes_
    idx = int(np.argmax(avg_prob))
    return classes[idx], avg_prob, classes


def plot_and_save_confusion(true_label, pred_label, classes, prefix, method):
    cm = confusion_matrix([true_label], [pred_label], labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"CM {method}")
    out_img = Path('plots') / f"{prefix}cm_{method}.png"
    fig.savefig(out_img)
    plt.close(fig)
    return cm, str(out_img)


def save_metrics(true_label, pred_label, classes, probs, prefix, method):
    y_true = [true_label]
    y_pred = [pred_label]
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    prob_dict = dict(zip(classes, probs.tolist()))
    out_json = Path('plots') / f"{prefix}metrics_{method}.json"
    with open(out_json, 'w') as f:
        json.dump({'metrics': metrics, 'probabilities': prob_dict}, f, indent=2)
    return metrics, str(out_json)


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia y métricas para un audio con e2v y w2v2"
    )
    parser.add_argument('audio_path', type=str, help="Path a WAV 16kHz")
    parser.add_argument(
        '--label', type=str, default=None,
        help="Etiqueta verdadera (opcional). Si no se pasa, solo se mostrará predicción y probabilidades."
    )
    args = parser.parse_args()

    # Modelos entrenados guardados como pickles
    e2v_models = sorted(Path('models').glob('e2v_fold*.pkl'))
    w2v2_models = sorted(Path('models').glob('w2v2_fold*.pkl'))

    # Inferencias
    e2v_pred, e2v_prob, e2v_classes = infer_e2v(
        args.audio_path, [str(p) for p in e2v_models]
    )
    w2v2_pred, w2v2_prob, w2v2_classes = infer_w2v2(
        args.audio_path, [str(p) for p in w2v2_models]
    )

    # Mostrar resultados
    print("=== emotion2vec ===")
    print(f"Pred: {e2v_pred}")
    print("Probabilidades:")
    for lab, pr in zip(e2v_classes, e2v_prob.tolist()):
        print(f"  {lab}: {pr:.3f}")

    print("\n=== wav2vec2 ===")
    print(f"Pred: {w2v2_pred}")
    print("Probabilidades:")
    for lab, pr in zip(w2v2_classes, w2v2_prob.tolist()):
        print(f"  {lab}: {pr:.3f}")

    # Si se proporciono etiqueta real, guardar metricas y matrices
    if args.label is not None:
        Path('plots').mkdir(exist_ok=True)
        prefix = 'inf_'
        plot_and_save_confusion(args.label, e2v_pred, e2v_classes, prefix, 'e2v')
        plot_and_save_confusion(args.label, w2v2_pred, w2v2_classes, prefix, 'w2v2')
        save_metrics(args.label, e2v_pred, e2v_classes, e2v_prob, prefix, 'e2v')
        save_metrics(args.label, w2v2_pred, w2v2_classes, w2v2_prob, prefix, 'w2v2')
    else:
        print("No se proporciono una etiqueta real; solo se muestra prediccion y probabilidades.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

import config
from utils.dataset_loader import load_iemocap_metadata, get_fold


def train_classifier(embed: str, use_mlp: bool):
    """
    Entrena un clasificador ligero (LogReg o MLP) sobre embeddings preextraidos.

    Args:
        embed (str): 'w2v2' o 'e2v'
        use_mlp (bool): si True, usar MLPClassifier, si False, LogisticRegression
    """
    # Carga metadata y embeddings
    df = load_iemocap_metadata()
    emb_file = Path(config.OUT_DIR) / f"all_{embed}_emb.npy"
    labels_file = Path(config.OUT_DIR) / "all_labels.npy"

    X = np.load(emb_file)
    y = np.load(labels_file)

    results = []
    metrics_out = Path(config.MODEL_DIR)
    metrics_out.mkdir(parents=True, exist_ok=True)
    cm_dir = Path(config.PLOTS_DIR)
    cm_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(config.N_FOLDS):
        # Split leave-one-session-out
        train_df, test_df = get_fold(df, fold)
        idx_tr = train_df.index.to_list()
        idx_te = test_df.index.to_list()

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_te, y_te = X[idx_te], y[idx_te]

        # Escoger clasificador
        if use_mlp:
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=50,
                early_stopping=True,
                n_iter_no_change=5,
                random_state=config.SEED
            )
        else:
            clf = LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                random_state=config.SEED
            )

        # Entrenamiento
        clf.fit(X_tr, y_tr)

        # Guardar modelo
        model_path = metrics_out / f"{embed}_fold{fold}.pkl"
        joblib.dump(clf, model_path)

        # Predicción y métricas
        y_pred = clf.predict(X_te)
        wa = accuracy_score(y_te, y_pred)
        ua = recall_score(y_te, y_pred, average='macro')
        f1 = f1_score(y_te, y_pred, average='macro')

        # Matriz de confusión
        cm = confusion_matrix(y_te, y_pred, labels=config.EMOTIONS)
        np.save(cm_dir / f"cm_{embed}_fold{fold}.npy", cm)

        results.append({
            'embed': embed,
            'fold': fold,
            'WA': wa,
            'UA': ua,
            'F1': f1
        })
        print(f"Fold {fold}: WA={wa:.3f}, UA={ua:.3f}, F1={f1:.3f}")

    # Guardar resultados agregados
    summary_path = metrics_out / f"{embed}_results.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('embed', choices=['w2v2', 'e2v'], help="Embedding a usar: 'w2v2' o 'e2v'")
    parser.add_argument('--mlp', action='store_true', help='Usar MLPClassifier en lugar de LogisticRegression')
    args = parser.parse_args()
    train_classifier(args.embed, use_mlp=args.mlp)

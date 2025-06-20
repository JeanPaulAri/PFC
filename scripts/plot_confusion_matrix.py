#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import config


def plot_cms(embed):
    cms = []
    for fold in range(config.N_FOLDS):
        cm_path = f"plots/cm_{embed}_fold{fold}.npy"
        cms.append(np.load(cm_path))

    fig, axes = plt.subplots(1, config.N_FOLDS, figsize=(4*config.N_FOLDS, 4))
    for i, ax in enumerate(axes):
        disp = ConfusionMatrixDisplay(confusion_matrix=cms[i], display_labels=config.EMOTIONS)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{embed} fold {i}")

    plt.tight_layout()
    out_path = f"plots/cms_{embed}.png"
    plt.savefig(out_path)
    print(f"Saved confusion matrix plot to '{out_path}'")


def main():
    for emb in ["w2v2", "e2v"]:
        plot_cms(emb)

if __name__ == '__main__':
    main()

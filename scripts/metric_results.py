#!/usr/bin/env python
import json
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

# Carga resultados de WA, UA y F1 para cada embedding

def load_metrics(embed):
    path = f"models/{embed}_results.json"
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def main():
    embeds = ["w2v2", "e2v"]
    dfs = {emb: load_metrics(emb) for emb in embeds}

    # Print medias y desviaciones
    summary = pd.concat(dfs.values(), keys=dfs.keys(), names=["embed"])                              
    stats = summary.groupby(level="embed")[['WA','UA','F1']].agg(['mean','std'])
    print("Medias y desviaciones por embedding:")
    print(stats)

    # Test pareado sobre UA
    ua_w2v2 = dfs['w2v2']['UA']
    ua_e2v  = dfs['e2v']['UA']
    stat, p = ttest_rel(ua_w2v2, ua_e2v)
    print(f"\nPaired t-test on UA: stat={stat:.4f}, p={p:.4f}")

    # Grafico de barras para UA
    means = summary['UA'].groupby(level='embed').mean()
    stds  = summary['UA'].groupby(level='embed').std()
    ax = means.plot.bar(yerr=stds, rot=0)
    ax.set_ylabel('Unweighted Accuracy (UA)')
    plt.tight_layout()
    plt.savefig('plots/ua_comparison.png')
    print("Grafico guardado en 'plots/ua_comparison.png'")

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs

def visualize_datasets(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for i, (name, (X, y)) in enumerate(datasets.items()):
        axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
        axes[i].set_title(name)
        axes[i].set_xlabel('x1')
        axes[i].set_ylabel('x2')
    plt.tight_layout()
    plt.show()

def read_datasets_from_csv():
    datasets = {}
    for name in ['linear', 'moons', 'circles', 'blobs']:
        df = pd.read_csv(f'dataset_{name}.csv')
        X = df[['x1', 'x2']].values
        y = df['label'].values
        datasets[name] = (X, y)
    return datasets

datasets = read_datasets_from_csv()
visualize_datasets(datasets)

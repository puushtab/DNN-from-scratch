import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs

def generate_datasets():
    datasets = {}

    # # Linearly separable data
    # X_linear, y_linear = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=2.0, random_state=42)
    # datasets['linear'] = (X_linear, y_linear)

    # # Moons (non-linear separation)
    # X_moons, y_moons = make_moons(n_samples=1000, noise=0.1, random_state=42)
    # datasets['moons'] = (X_moons, y_moons)

    # # Circles (non-linear separation)
    # X_circles, y_circles = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    # datasets['circles'] = (X_circles, y_circles)

    # # Blobs (multiple clusters)
    # X_blobs, y_blobs = make_blobs(n_samples=1000, centers=2, cluster_std=1.0, random_state=42)
    # datasets['blobs'] = (X_blobs, y_blobs)

    # Blobs with 2 categories, close and medium variance
    X_blobs_close, y_blobs_close = make_blobs(n_samples=1000, centers=2, cluster_std=1.2, random_state=12)
    datasets['blobs_close'] = (X_blobs_close, y_blobs_close)

    return datasets

def save_datasets_to_csv(datasets):
    for name, (X, y) in datasets.items():
        df = pd.DataFrame(np.column_stack((X, y)), columns=['x1', 'x2', 'label'])
        df.to_csv(f'dataset_{name}.csv', index=False)
        print(f"Saved dataset '{name}' to dataset_{name}.csv")

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

datasets = generate_datasets()
save_datasets_to_csv(datasets)
visualize_datasets(datasets)

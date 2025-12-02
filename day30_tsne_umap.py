import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import umap

# -------------------------
# Dataset
# -------------------------

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# t-SNE
# -------------------------

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_tsne[y==i,0], X_tsne[y==i,1], label=target_name)
plt.title("t-SNE Visualization of Iris Dataset")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.show()

# -------------------------
# UMAP
# -------------------------

umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_umap[y==i,0], X_umap[y==i,1], label=target_name)
plt.title("UMAP Visualization of Iris Dataset")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.show()


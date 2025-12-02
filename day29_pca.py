import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------
# Dataset
# -------------------------

data = {
    "Age": [14, 17, 18, 20, 22, 25, 30, 35, 40, 45],
    "Income": [20000, 25000, 27000, 30000, 32000, 35000, 50000, 60000, 65000, 80000],
    "Expenses": [1000, 1500, 1200, 1800, 2000, 2200, 3000, 3500, 4000, 4500]
}

df = pd.DataFrame(data)

# -------------------------
# Standardize features
# -------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -------------------------
# PCA Transformation
# -------------------------

pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Original Shape:", X_scaled.shape)
print("Transformed Shape:", X_pca.shape)

# -------------------------
# Visualization
# -------------------------

plt.scatter(X_pca[:,0], X_pca[:,1], c='blue', s=100)
for i, txt in enumerate(df.index):
    plt.annotate(txt, (X_pca[i,0]+0.1, X_pca[i,1]+0.1))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Projection")
plt.show()

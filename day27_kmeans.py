import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------
# Dataset
# -------------------------

data = {
    "Age": [14, 17, 18, 20, 22, 25, 30, 35, 40, 45],
    "Income": [20000, 25000, 27000, 30000, 32000, 35000, 50000, 60000, 65000, 80000]
}

df = pd.DataFrame(data)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -------------------------
# K-Means Model
# -------------------------

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Cluster labels
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster labels:", labels)
print("Centroids:\n", centroids)

# -------------------------
# Visualization
# -------------------------

plt.scatter(df["Age"], df["Income"], c=labels, cmap="viridis", s=100, label="Data Points")
plt.scatter(centroids[:,0]*df["Age"].std()+df["Age"].mean(),
            centroids[:,1]*df["Income"].std()+df["Income"].mean(),
            color='red', s=200, marker='X', label='Centroids')
plt.xlabel("Age")
plt.ylabel("Income")
plt.title(f"K-Means Clustering (k={k})")
plt.legend()
plt.show()

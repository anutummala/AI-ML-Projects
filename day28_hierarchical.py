import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

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
# Linkage and Dendrogram
# -------------------------

linked = linkage(X_scaled, method='ward')  # Ward minimizes variance

plt.figure(figsize=(10,5))
dendrogram(linked,
           labels=np.arange(1, len(df)+1),
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# -------------------------
# Form Clusters
# -------------------------

k = 3
clusters = fcluster(linked, k, criterion='maxclust')
df['Cluster'] = clusters
print(df)

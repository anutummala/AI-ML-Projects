# day11_numpy_mlprep.py
# NumPy basics for ML data preparation

import numpy as np

# --------------------------
# 1. Create NumPy arrays
ages = np.array([22, 25, 25, 19, 14, 25])
name_lengths = np.array([5, 3, 4, 4, 4, 4])  # Example lengths of student names

print("Ages:", ages)
print("Name lengths:", name_lengths)

# --------------------------
# 2. Combine arrays into feature matrix
X = np.column_stack((ages, name_lengths))  # Shape: (6, 2)
print("Feature matrix X:\n", X)

# --------------------------
# 3. Normalize the features (Min-Max scaling)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)
print("Normalized feature matrix X_norm:\n", X_norm)

# --------------------------
# 4. Create target labels (categories)
y = np.array([1, 1, 1, 1, 0, 1])  # 1 = Young, 0 = Minor
print("Target labels y:", y)

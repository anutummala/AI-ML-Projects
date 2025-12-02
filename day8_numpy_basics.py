# day8_numpy_basics.py
# Python NumPy Basics for AI/ML

import numpy as np

# --------------------------
# 1. Creating arrays
ages = np.array([22, 25, 25, 19, 14, 25])
print("Ages array:", ages)

# Basic statistics
print("Mean:", np.mean(ages))
print("Sum:", np.sum(ages))
print("Max:", np.max(ages))
print("Min:", np.min(ages))
print("Standard deviation:", np.std(ages))

# --------------------------
# 2. Array operations
print("Ages + 2:", ages + 2)
print("First 3 ages:", ages[:3])
print("Last 2 ages:", ages[-2:])

# --------------------------
# 3. 2D arrays (Matrices)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("Matrix A:\n", A)
print("Transpose:\n", A.T)

B = np.array([[1, 1],
              [2, 1],
              [3, 2]])
dot_product = A.dot(B)
print("Dot product A.B:\n", dot_product)

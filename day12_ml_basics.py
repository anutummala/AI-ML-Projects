# day12_ml_basics.py
# Basic Machine Learning using scikit-learn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. Prepare dataset
# Features: Age and Name length
X = np.array([[22,5], [25,3], [25,4], [19,4], [14,4], [25,4]])
# Target labels: 1 = Young, 0 = Minor
y = np.array([1,1,1,1,0,1])

# --------------------------
# 2. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# --------------------------
# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 4. Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# --------------------------
# 5. Evaluate model
accuracy = knn.score(X_test_scaled, y_test)
print("Model Accuracy:", accuracy)

# --------------------------
# 6. Predict categories for test set
predictions = knn.predict(X_test_scaled)
print("Predicted Categories:", predictions)

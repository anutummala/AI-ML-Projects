# day15_nn_inspection.py
# Understanding Neural Network outputs and activations

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# --------------------------
# 1. Prepare dataset
X = np.array([[22,5], [25,3], [25,4], [19,4], [14,4], [25,4]])
y = np.array([1,1,1,1,0,2])  # Multi-class labels

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1,1))

# --------------------------
# 2. Build simple neural network
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(2,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(y_encoded.shape[1], activation='softmax'))  # Multi-class output

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_encoded, epochs=100, verbose=0)

# --------------------------
# 3. Evaluate model
loss, accuracy = model.evaluate(X_scaled, y_encoded, verbose=0)
print("Model Accuracy:", accuracy)

# --------------------------
# 4. Inspect predictions
predictions = model.predict(X_scaled)
print("\nPredicted probabilities for each class:")
print(predictions)

predicted_classes = np.argmax(predictions, axis=1)
print("\nPredicted Classes:", predicted_classes)

# --------------------------
# 5. Compare with actual labels
print("\nActual labels:", y)

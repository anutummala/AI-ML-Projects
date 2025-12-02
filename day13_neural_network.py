# day13_neural_network.py
# Simple Neural Network with TensorFlow/Keras

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. Prepare dataset
X = np.array([[22,5], [25,3], [25,4], [19,4], [14,4], [25,4]])
y = np.array([1,1,1,1,0,1])  # 1 = Young, 0 = Minor

# --------------------------
# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 3. Build neural network
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(2,)))  # Input layer + first hidden layer
model.add(Dense(8, activation='relu'))  # Second hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# --------------------------
# 4. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------
# 5. Train the model
model.fit(X_scaled, y, epochs=50, verbose=0)

# --------------------------
# 6. Evaluate the model
loss, accuracy = model.evaluate(X_scaled, y, verbose=0)
print("Neural Network Accuracy:", accuracy)

# --------------------------
# 7. Make predictions
predictions = model.predict(X_scaled)
predicted_classes = (predictions > 0.5).astype(int).flatten()
print("Predicted Categories:", predicted_classes)

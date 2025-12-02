# day14_multiclass_nn.py
# Multi-class Neural Network with TensorFlow/Keras

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# --------------------------
# 1. Prepare dataset
# Features: Age and Name length
X = np.array([[22,5], [25,3], [25,4], [19,4], [14,4], [25,4]])
# Target labels: 0 = Minor, 1 = Young, 2 = Adult (example)
y = np.array([1,1,1,1,0,2])

# --------------------------
# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 3. One-hot encode target labels for multi-class classification
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# --------------------------
# 4. Build neural network
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(2,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(y_encoded.shape[1], activation='softmax'))  # Output layer with softmax

# --------------------------
# 5. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------------
# 6. Train the model
model.fit(X_scaled, y_encoded, epochs=100, verbose=0)

# --------------------------
# 7. Evaluate the model
loss, accuracy = model.evaluate(X_scaled, y_encoded, verbose=0)
print("Multi-class Neural Network Accuracy:", accuracy)

# --------------------------
# 8. Make predictions
predictions = model.predict(X_scaled)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted Categories:", predicted_classes)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# -------------------------
# Dataset (XOR)
# -------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([0,1,1,0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -------------------------
# Build Model
# -------------------------
model = Sequential()
model.add(Dense(4, input_dim=2, activation='tanh'))  # Hidden layer with Tanh
model.add(Dense(3, activation='relu'))               # Hidden layer with ReLU
model.add(Dense(1, activation='sigmoid'))            # Output layer with Sigmoid

# -------------------------
# Compile with Adam optimizer
# -------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# -------------------------
# Train Model
# -------------------------
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, verbose=0)

# -------------------------
# Evaluate & Predict
# -------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

predictions = model.predict(X_test)
print("Predictions (rounded):", np.round(predictions))

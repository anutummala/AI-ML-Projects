import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------
# Sample Dataset
# -------------------------

# Input features (2 features per sample)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Labels (XOR problem)
y = np.array([0, 1, 1, 0])

# -------------------------
# Build Model
# -------------------------

model = Sequential()

# Hidden layer 1
model.add(Dense(4, input_dim=2, activation='relu'))

# Hidden layer 2
model.add(Dense(3, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# -------------------------
# Compile Model
# -------------------------

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# -------------------------
# Train Model
# -------------------------

model.fit(X, y, epochs=500, verbose=1)

# -------------------------
# Make Predictions
# -------------------------

predictions = model.predict(X)
print("\nPredictions:")
print(predictions.round())

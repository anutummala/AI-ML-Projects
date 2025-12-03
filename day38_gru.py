import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# -------------------------
# Generate Sequence Data
# -------------------------
t = np.linspace(0, 20, 200)
data = np.sin(t)

# Prepare sequences
seq_length = 20
X = []
y = []

for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])

X = np.array(X)
y = np.array(y)

# Reshape for GRU: (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------
# Build GRU Model
# -------------------------
model = Sequential()
model.add(GRU(50, activation='tanh', input_shape=(seq_length,1)))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mse')

# -------------------------
# Train Model
# -------------------------
history = model.fit(X, y, epochs=300, verbose=0)

# -------------------------
# Predict
# -------------------------
predictions = model.predict(X)

# -------------------------
# Plot Results
# -------------------------
plt.figure(figsize=(10,4))
plt.plot(range(len(data)), data, label='Original Sine Wave')
plt.plot(range(seq_length, len(data)), predictions, label='GRU Predictions')
plt.title("GRU Sequence Prediction")
plt.legend()
plt.show()

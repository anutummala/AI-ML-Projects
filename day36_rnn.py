import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# -------------------------
# Generate Sequence Data
# -------------------------
t = np.linspace(0, 10, 100)
data = np.sin(t)

# Prepare input sequences
X = []
y = []
seq_length = 10

for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])

X = np.array(X)
y = np.array(y)

# Reshape for RNN: (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------
# Build RNN Model
# -------------------------
model = Sequential()
model.add(SimpleRNN(20, activation='tanh', input_shape=(seq_length, 1)))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mse')

# -------------------------
# Train Model
# -------------------------
history = model.fit(X, y, epochs=200, verbose=0)

# -------------------------
# Predict
# -------------------------
predictions = model.predict(X)

# -------------------------
# Plot Results
# -------------------------
plt.figure(figsize=(10,4))
plt.plot(range(len(data)), data, label='Original Sine Wave')
plt.plot(range(seq_length, len(data)), predictions, label='RNN Predictions')
plt.title("RNN Sequence Prediction")
plt.legend()
plt.show()
